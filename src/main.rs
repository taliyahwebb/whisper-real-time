use core::panic;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::time::Instant;

use clap::{arg, Parser};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleRate;
use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};
use wav_io::writer::Writer;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// path to the whisper.cpp model to be used
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,

    /// path to the whisper.cpp binary
    #[arg(short, long, value_name = "FILE")]
    whisper_cpp: Option<PathBuf>,

    /// path to a file to transcribe
    ///
    /// Transcribes the file instead of the microphone stream
    #[arg(short, long, value_name = "FILE")]
    file: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    whisper(args);
}

const SAMPLE_RATE: usize = 16000;
fn whisper(args: Args) {
    if let Some(file) = args.file {
        let (_header, waveform) =
            wav_io::read_from_file(File::open(file).expect("file doesnt exist"))
                .expect("invalid wav file");
        let samples = wav_io::convert_samples_f32_to_i16(&waveform);
        let now = Instant::now();
        match args.whisper_cpp.clone() {
            Some(bin) => decode_bin(args.model.clone(), bin, &samples),
            None => decode_first(&mut Whisper::new(args.model), &samples),
        }
        println!("\t@{:?}", now.elapsed());
        return;
    }
    let mut whisper = Whisper::new(&args.model);
    let host = cpal::default_host();
    let mic = host.default_input_device().expect("no mic");
    eprintln!("using audio: '{}'", mic.name().unwrap());
    let (tx, rx) = channel::<Vec<_>>();
    const VAD_FRAME: usize = 480;
    let mut next_vad_frame = Vec::with_capacity(VAD_FRAME);
    let config = mic
        .supported_input_configs()
        .expect("should be able to fetch input device config")
        .next()
        .expect("should have a device config")
        .with_sample_rate(SampleRate(SAMPLE_RATE as u32))
        .config();
    let stream = mic
        .build_input_stream(
            &config,
            move |data: &[i16], _info| {
                let mut head = 0;
                let mut available = data.len();
                while available > 0 {
                    let free = 480 - next_vad_frame.len();
                    let used = free.min(available);
                    next_vad_frame.extend_from_slice(&data[head..head + used]);
                    head += used;
                    available -= used;
                    if next_vad_frame.len() == 480 {
                        tx.send(next_vad_frame.clone())
                            .expect("should be able to send vad frame");
                        next_vad_frame.clear();
                    }
                }
            },
            move |err| {
                eprintln!("error: {err}");
            },
            None,
        )
        .expect("config should be able to work");
    stream.play().expect("could not listen to microphone");

    let mut vad = VoiceActivityDetector::new_with_model(
        VoiceActivityModel::ES_ALPHA,
        VoiceActivityProfile::VERY_AGGRESSIVE,
    );

    let mut speech_segment = None;
    let mut segment = 0;

    let mut accumulator = [0i16; SAMPLE_RATE * 30];
    const HEAD_ZERO: usize = 1600 * 7; // prepend 700ms of silence so single word statements get picked up
    let mut head = HEAD_ZERO;

    let dispatch = |whisper: &mut Whisper,
                    accumulator: &mut [i16; SAMPLE_RATE * 30],
                    head: &mut usize,
                    speech_segment: &mut Option<u64>,
                    segment: &mut u64| {
        let now = Instant::now();
        match args.whisper_cpp.clone() {
            Some(bin) => decode_bin(args.model.clone(), bin, &accumulator[..*head]),
            None => decode_first(whisper, &accumulator[..*head]),
        }
        eprintln!("\t@{:?}", now.elapsed());
        accumulator.fill(0);
        *head = HEAD_ZERO;
        *speech_segment = None;
        *segment = 0;
    };

    while let Ok(partial) = rx.recv() {
        if speech_segment.is_some() {
            segment += 1;
        }
        let predict = vad
            .predict_16khz(&partial)
            .expect("should be able to predict audio");
        if !predict {
            if speech_segment.is_some_and(|last| segment - last <= 3) {
                // allow 3 more segments after we stop detecting speech (~90ms)
            } else if speech_segment.is_some_and(|last| segment - last > 8) {
                // if we have 8 segments of non speech stop listening and dispatch (~240ms)
                dispatch(
                    &mut whisper,
                    &mut accumulator,
                    &mut head,
                    &mut speech_segment,
                    &mut segment,
                );
                continue; // skip the sample
            } else {
                continue; // skip the sample
            }
        }
        if speech_segment.is_none() {
            eprintln!("started speech");
        }
        if predict {
            // only update the latest segment id if voice was detected
            speech_segment = Some(segment);
        }
        if head < accumulator.len() {
            let overflow = (head + partial.len()).saturating_sub(accumulator.len());
            let max = partial.len() - overflow;
            let (used, leftover) = partial
                .split_at_checked(max)
                .expect("should be able to split ");
            accumulator[head..(head + max)].copy_from_slice(used);
            head += max;
            if max < partial.len() {
                dispatch(
                    &mut whisper,
                    &mut accumulator,
                    &mut head,
                    &mut speech_segment,
                    &mut segment,
                );
                accumulator[head..head + leftover.len()].copy_from_slice(leftover);
                head += leftover.len();
                speech_segment = Some(segment);
            }
            continue;
        }
        panic!("should not be reachable")
    }
}

fn decode_bin(model: PathBuf, binary: PathBuf, samples: &[i16]) {
    let header = wav_io::new_header(SAMPLE_RATE as u32, 16, false, true);
    let mut writer = Writer::new();
    writer
        .from_scratch_i16(&header, &samples.to_vec())
        .expect("could not turn into wav file");
    let bytes = writer.to_bytes();

    let mut out = Command::new(binary)
        .arg("--no-prints")
        .arg("--no-timestamps")
        .arg("-f")
        .arg("-") // read from stdin
        .arg("-m")
        .arg(model.clone().into_os_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("could not execute child");
    out.stdin
        .as_mut()
        .expect("child should have stdin")
        .write_all(&bytes)
        .expect("could not pipe audio");
    out.wait().expect("whisper.cpp should not error");
    let mut lines = String::new();
    out.stdout
        .take()
        .expect("whisper.cpp should have an stdout")
        .read_to_string(&mut lines)
        .expect("whisper.cpp output should be utf8");
    let stdout = lines
        .strip_prefix('\n')
        .expect("output from this tool should have a leading newline");
    if !stdout.is_empty() {
        print!(
            "{}",
            &stdout
                .strip_prefix(' ')
                .expect("output from this tool should have a leading space")
        );
    }
}

struct Whisper {
    state: WhisperState,
    params: FullParams<'static, 'static>,
}

impl Whisper {
    pub fn new(model: impl AsRef<Path>) -> Whisper {
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(
            model.as_ref().to_str().expect("path should be utf8"),
            params,
        )
        .expect("could not load model");
        // now we can run the model
        let state = ctx.create_state().expect("failed to create state");

        // create a params object
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(4);
        params.set_no_timestamps(true);
        params.set_suppress_non_speech_tokens(false);

        Whisper { state, params }
    }
}

fn decode_first(whisper: &mut Whisper, samples: &[i16]) {
    let mut float_samples = [0f32; SAMPLE_RATE * 30];
    whisper_rs::convert_integer_to_float_audio(samples, &mut float_samples[..samples.len()])
        .expect("should be able to de-quantize data");

    eprintln!("running transcription with {}", samples.len());
    whisper
        .state
        .full(whisper.params.clone(), &float_samples)
        .expect("failed to run model");

    // fetch the results
    let num_segments = whisper
        .state
        .full_n_segments()
        .expect("failed to get number of segments");

    for i in 0..num_segments {
        let segment = whisper
            .state
            .full_get_segment_text(i)
            .expect("should have a segment");
        println!("{segment}");
    }
}

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use clap::{arg, Parser};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleRate;
use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};
use wav_io::writer::Writer;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// path to the whisper.cpp model to be used
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,

    /// path to the whisper.cpp binary
    #[arg(short, long, value_name = "FILE")]
    whisper_cpp: PathBuf,
}

fn main() {
    let args = Args::parse();
    whisper(args);
}

const SAMPLE_RATE: usize = 16000;
fn whisper(args: Args) {
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

    let mut speech = None;

    let mut accumulator = [0i16; SAMPLE_RATE * 30];
    const HEAD_ZERO: usize = 1600 * 7; // prepend 700ms of silence so single word statements get picked up
    let mut head = HEAD_ZERO;

    while let Ok(partial) = rx.recv() {
        let predict = vad
            .predict_16khz(&partial)
            .expect("should be able to predict audio");
        if !predict {
            if speech.is_some_and(|last: Instant| last.elapsed() < Duration::from_millis(95)) {
                // linger 3 segments to allow the word to finish
            } else if speech
                .is_some_and(|last: Instant| last.elapsed() > Duration::from_millis(750))
            {
                let header = wav_io::new_header(SAMPLE_RATE as u32, 16, false, true);
                // wav_io::writer::i16samples_to_file(
                //     &mut File::create("output.wav").unwrap(),
                //     &header,
                //     &accumulator[..head].to_vec(),
                // )
                // .expect("could not write tmpfile");
                let mut writer = Writer::new();
                writer
                    .from_scratch_i16(&header, &accumulator[..head].to_vec())
                    .expect("could not turn into wav file");
                let bytes = writer.to_bytes();

                let mut out = Command::new(args.whisper_cpp.clone())
                    .arg("--no-prints")
                    .arg("--no-timestamps")
                    .arg("-f")
                    .arg("-") // read from stdin
                    .arg("-m")
                    .arg(args.model.clone().into_os_string())
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
                // decode_first(&mut state, params.clone(), &float_samples[..head]);
                println!("\t@{:?}", speech.unwrap().elapsed());
                accumulator.fill(0);
                head = HEAD_ZERO;
                speech = None;
                continue; // skip the sample
            } else {
                continue; // skip the sample
            }
        }
        if speech.is_none() {
            eprintln!("started speech");
        }
        if predict {
            speech = Some(Instant::now());
        }
        if head < accumulator.len() {
            let overflow = (head + partial.len()).saturating_sub(accumulator.len());
            let max = partial.len() - overflow;
            accumulator[head..(head + max)].copy_from_slice(&partial[..max]);
            head += max;
            continue;
        }
        panic!("overrun case not handled");
    }
}

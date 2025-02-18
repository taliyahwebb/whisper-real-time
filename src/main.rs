use std::env::args;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleRate;
use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

fn main() {
    let model = args()
        .skip(1)
        .next()
        .unwrap_or("./ggml-tiny.en.bin".into())
        .into();
    whisper(model);
}

const SAMPLE_RATE: usize = 16000;
fn whisper(model: PathBuf) {
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

    let mut params = WhisperContextParameters::default();
    params.flash_attn(true);
    let ctx = WhisperContext::new_with_params(model.to_str().expect("path should be utf8"), params)
        .expect("could not load model");

    // create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(10);

    // now we can run the model
    let mut state = ctx.create_state().expect("failed to create state");
    eprintln!("model loaded");

    let mut vad = VoiceActivityDetector::new_with_model(
        VoiceActivityModel::ES_ALPHA,
        VoiceActivityProfile::VERY_AGGRESSIVE,
    );

    // assume we have a buffer of audio data
    // here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
    // let file = File::open("output.wav").expect("no test file");
    // let (_header, samples) = wav_io::read_from_file(file).expect("parsing
    // error"); let start = Instant::now();
    // decode_first(&mut state, params.clone(), &samples[..]);
    // println!("transcription took {:?}", start.elapsed());
    // return;

    let mut speech = None;
    let mut float_samples = [0f32; SAMPLE_RATE * 30];
    let mut accumulator = [0i16; SAMPLE_RATE * 30];
    let mut head = 0;
    while let Ok(partial) = rx.recv() {
        let predict = vad
            .predict_16khz(&partial)
            .expect("should be able to predict audio");
        if !predict {
            if speech.is_some_and(|last: Instant| last.elapsed() < Duration::from_millis(50)) {
            } else if speech
                .is_some_and(|last: Instant| last.elapsed() > Duration::from_millis(500))
            {
                whisper_rs::convert_integer_to_float_audio(&accumulator[..], &mut float_samples)
                    .expect("should be able to de-quantize data");
                // let header = wav_io::new_header(SAMPLE_RATE as u32, 32, true, true);
                // wav_io::write_to_file(
                //     &mut File::create("output.wav").unwrap(),
                //     &header,
                //     &float_samples[..head].to_vec(),
                // )
                // .unwrap();
                decode_first(&mut state, params.clone(), &float_samples[..head]);
                eprintln!("@{:?}", speech.unwrap().elapsed());
                accumulator.fill(0);
                head = 0;
                speech = None;
                continue;
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
        whisper_rs::convert_integer_to_float_audio(&accumulator[..], &mut float_samples)
            .expect("should be able to de-quantize data");
        decode_first(&mut state, params.clone(), &float_samples[..head]);
        eprintln!("\t@{:?}", speech.unwrap().elapsed());
        accumulator.fill(0);
        head = 0;
        speech = None;
    }
}

fn decode_first(state: &mut WhisperState, params: FullParams, samples: &[f32]) -> Option<String> {
    eprintln!("running transcription with {}", samples.len());
    state.full(params, &samples).expect("failed to run model");

    // fetch the results
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");

    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get segment start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get segment end timestamp");
        // println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        println!("{segment}");
    }
    return None;
}

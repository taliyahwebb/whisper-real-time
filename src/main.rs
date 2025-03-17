use core::panic;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use clap::{arg, Parser};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Stream};
use ringbuf::traits::{Consumer, Split};
use ringbuf::HeapRb;
use vad::{get_resampler, Vad, VadActivity};
use wav_io::header;
use wav_io::writer::Writer;
use whisper::{Whisper, WhisperOptions, MAX_WHISPER_FRAME, SAMPLE_RATE};

mod vad;
mod whisper;

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

    /// list available audio devices
    #[arg(short, long)]
    list: bool,

    /// audio device to listen to
    #[arg(short, long)]
    device: Option<String>,
}

fn main() {
    let args = Args::parse();
    whisper(args);
}

// this is a drop guard/container object
// no need to read the values
enum StreamHandle {
    #[allow(dead_code)]
    Thread(JoinHandle<()>),
    #[allow(dead_code)]
    Stream(Stream),
}

fn whisper(args: Args) {
    if args.list {
        let host = cpal::default_host();
        let devices = host.input_devices().unwrap();
        eprintln!("Available audio devices:");
        for dev in devices {
            println!("- {}", dev.name().expect("couldnt get device name"));
        }
        return;
    }

    let whisper_opts = WhisperOptions {
        translate_en: false,
        language: "en".to_string(),
    };
    let mut whisper =
        Whisper::with_options(&args.model, whisper_opts).expect("should be able to load whisper");
    let host = cpal::default_host(); // TODO add mic selection
    let mic = host.default_input_device().expect("no mic");
    let (mic, config) = vad::get_microphone_by_name(
        &args
            .device
            .unwrap_or_else(|| mic.name().expect("default device should have a name")),
    )
    .expect("should be able to get default mic");
    eprintln!("using audio: '{}'", mic.name().unwrap());
    let ring = HeapRb::<i16>::try_new(MAX_WHISPER_FRAME * 2).expect("cannot allocate audio ring");
    let (mut producer, mut consumer) = ring.split();
    let (mut activity_tx, activity_rx) = mpsc::channel::<VadActivity>();
    let mut vad = Vad::try_new(&config).expect("should be able to build vad");
    let _handle = if let Some(file) = args.file {
        let (header, waveform) =
            wav_io::read_from_file(File::open(file).expect("file doesnt exist"))
                .expect("invalid wav file");
        let buf_size = (header.sample_rate / 30) * header.channels as u32;
        let handle = thread::spawn(move || {
            let resample_with = get_resampler(header.sample_rate);
            if header.channels == 2 {
                eprintln!("converting stereo to mono audio");
            }
            for chunk in waveform.chunks(buf_size as usize) {
                let now = Instant::now();
                let timeout =
                    Duration::from_millis((chunk.len() as u64 * 1000) / header.sample_rate as u64);
                vad::audio_loop(
                    chunk,
                    header.channels,
                    &resample_with,
                    &mut producer,
                    &mut vad,
                    &mut activity_tx,
                );
                let delta = Instant::now() - now;
                thread::sleep(timeout - delta);
            }
        });
        StreamHandle::Thread(handle)
    } else {
        let handle = thread::spawn(move || {
            let resample_with = get_resampler(config.sample_rate.0);
            if config.channels == 2 {
                eprintln!("converting stereo to mono audio");
            }
            let (audio_tx, audio_rx) = mpsc::sync_channel(10);
            let stream = mic
                .build_input_stream(
                    &config,
                    move |data: &[f32], _info| {
                        if audio_tx.try_send(data.to_vec()).is_err() {
                            eprintln!("audio is being dropped");
                        }
                    },
                    move |err| {
                        eprintln!("error: {err}");
                    },
                    None,
                )
                .expect("config should be able to work");
            stream.play().expect("could not listen to microphone");
            while let Ok(data) = audio_rx.recv() {
                vad::audio_loop(
                    &data,
                    config.channels,
                    &resample_with,
                    &mut producer,
                    &mut vad,
                    &mut activity_tx,
                );
            }
        });
        StreamHandle::Thread(handle)
    };

    while let Ok(event) = activity_rx.recv() {
        match event {
            VadActivity::SpeechStart => eprintln!("speech started"),
            VadActivity::SpeechEnd(samples) => {
                let now = Instant::now();
                match args.whisper_cpp.clone() {
                    Some(bin) => {
                        let mut buf = vec![0; samples];
                        if consumer.pop_slice(&mut buf) != samples {
                            panic!("logic error: not enough samples could be fetched");
                        }
                        decode_bin(args.model.clone(), bin, &buf)
                    }
                    None => {
                        if consumer.pop_slice(whisper.audio_buf(samples)) != samples {
                            panic!("logic error: not enough samples could be fetched");
                        }
                        if let Some(final_text) = whisper.transcribe() {
                            println!("{final_text}");
                        }
                    }
                }
                println!("\t@{:?}", now.elapsed());
            }
        }
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

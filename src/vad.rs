use core::panic;
use std::mem::{self, MaybeUninit};
use std::sync::mpsc::Sender;

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{BufferSize, Device, SampleRate, StreamConfig};
use earshot::{VoiceActivityDetector, VoiceActivityModel, VoiceActivityProfile};
use ringbuf::storage::Heap;
use ringbuf::traits::{Consumer, Observer, Producer};
use ringbuf::LocalRb;
use samplerate::Samplerate;
use wav_io::utils::stereo_to_mono;

use crate::whisper::SAMPLE_RATE;

/// selectable alsa buffer sizes follow a weird pattern 32 seems to work as a
/// quantum over a wide range of buffer sizes
const ALSA_BUFFER_QAUANTUM: u32 = 32;
const ALSA_BUFFER_MIN: u32 = 32;

/// ~30ms of audio
pub const VAD_FRAME: usize = 480; // sample count

pub type NSamples = usize;
pub enum VadStatus {
    Silence,
    SpeechStart,
    Speech,
    SpeechEnd(NSamples),
}

#[derive(Debug)]
pub enum AudioError {
    #[allow(dead_code)] // this is implicitly read during except via debug
    InputDeviceUnavailable(String),
}

pub enum VadActivity {
    SpeechStart,
    SpeechEnd(NSamples),
}

pub struct Vad {
    vad: VoiceActivityDetector,
    ring: LocalRb<Heap<i16>>,
    // TODO: build control structure
    /// reading this while `last_speech_frame = None` is undefined behavior
    current_frame: usize,
    last_speech_frame: Option<usize>,
    /// reading this while `last_speech_frame = None` is undefined behavior
    current_speech_samples: NSamples,
}

impl Vad {
    pub fn try_new(config: &StreamConfig) -> Result<Vad, &'static str> {
        let BufferSize::Fixed(buffer_size) = config.buffer_size else {
            return Err("config doesnt allow safe vad setup");
        };
        let ring = LocalRb::new((buffer_size * 2).max(VAD_FRAME as u32 * 2) as usize);
        Ok(Vad {
            vad: VoiceActivityDetector::new_with_model(
                VoiceActivityModel::ES_ALPHA,
                VoiceActivityProfile::VERY_AGGRESSIVE,
            ),
            ring,
            current_frame: 0,
            last_speech_frame: None,
            current_speech_samples: 0,
        })
    }

    pub fn input(&mut self, samples: &[i16]) {
        if self.ring.push_slice(samples) != samples.len() {
            eprintln!("warning: internal buffer full, some audio was dropped");
        }
    }

    pub fn output_to(&mut self, final_ring: &mut impl Producer<Item = i16>) -> VadStatus {
        while self.ring.occupied_len() >= VAD_FRAME {
            let mut frame: [MaybeUninit<i16>; VAD_FRAME] =
                [const { MaybeUninit::uninit() }; VAD_FRAME];
            if VAD_FRAME != self.ring.pop_slice_uninit(&mut frame) {
                panic!("vad ring should have enough data for at least one frame");
            }
            // SAFETY: this is safe because the panic above makes sure that all i16 were
            // initialized
            let frame =
                unsafe { mem::transmute::<[MaybeUninit<i16>; VAD_FRAME], [i16; VAD_FRAME]>(frame) };

            let is_speech = self
                .vad
                .predict_16khz(&frame)
                .expect("frame should have valid length");

            let Some(last_speech_frame) = self.last_speech_frame.as_mut() else {
                // we are inside a silence window
                if !is_speech {
                    continue;
                }
                // speech just started
                let n = final_ring.push_slice(&frame);
                if n != frame.len() {
                    eprintln!("transcription audio ring was full, dropped some audio");
                }

                self.last_speech_frame = Some(0);
                self.current_speech_samples = n;
                self.current_frame = 0;
                return VadStatus::SpeechStart; // it's ok to return here since
                                               // the upper level will poll
                                               // again until `Speech`
            };
            // we are inside a speech window
            self.current_frame += 1;
            let silence_frames = self.current_frame - *last_speech_frame;
            if !is_speech && silence_frames >= 8 {
                // if silence for 240ms
                self.last_speech_frame = None;
                return VadStatus::SpeechEnd(self.current_speech_samples);
            }

            if is_speech {
                *last_speech_frame = self.current_frame;
            }
            if is_speech || silence_frames <= 3 {
                // if speech or silence <= 90ms record audio
                let n = final_ring.push_slice(&frame);
                if n != frame.len() {
                    eprintln!("transcription audio ring was full, dropped some audio");
                }

                self.current_speech_samples += n;
            }
        }
        match self.last_speech_frame {
            Some(_) => VadStatus::Speech,
            None => VadStatus::Silence,
        }
    }
}

pub fn audio_loop(
    data: &[f32],
    channels: u16,
    resample_from: &Option<Samplerate>,
    ring_buffer: &mut impl Producer<Item = i16>,
    vad: &mut Vad,
    activity: &mut Sender<VadActivity>,
) {
    let data = match channels {
        1 => data,
        2 => &stereo_to_mono(data.to_vec()),
        n => panic!("configs with {n} channels are not supported"),
    };

    let data = match resample_from {
        None => data,
        Some(resampler) => &resampler.process(data).expect("should be able to resample"),
    };
    let data = wav_io::convert_samples_f32_to_i16(&data.to_vec());

    vad.input(&data);
    loop {
        let status = vad.output_to(ring_buffer);
        match status {
            VadStatus::Silence => (),
            VadStatus::Speech => (),
            VadStatus::SpeechEnd(samples) => {
                // can safely drop the error case here as it only happens when the receiver has
                // hung up (which means the stream is bound to stop soon too)
                let _ = activity.send(VadActivity::SpeechEnd(samples));
                continue; // make sure we run this input to completion
            }
            VadStatus::SpeechStart => {
                // can safely drop the error case here as it only happens when the receiver has
                // hung up (which means the stream is bound to stop soon too)
                let _ = activity.send(VadActivity::SpeechStart);
                continue; // make sure we run this input to completion
            }
        }
        break;
    }
}

pub fn get_microphone_by_name(name: &str) -> Result<(Device, StreamConfig), AudioError> {
    let host = cpal::default_host();
    let mut devices = host.input_devices().unwrap();
    if let Some(device) = devices.find(|device| device.name().unwrap() == name) {
        let config = device
            .supported_input_configs()
            .map_err(|err| AudioError::InputDeviceUnavailable(format!("{name}: '{err}'")))?
            .next()
            .ok_or_else(|| {
                AudioError::InputDeviceUnavailable(format!(
                    "{name}: 'does not have any valid input configurations'"
                ))
            })?;
        let config = config
            .try_with_sample_rate(SampleRate(SAMPLE_RATE as u32))
            .unwrap_or_else(|| {
                let dev_rate = if config.min_sample_rate().0 > SAMPLE_RATE as u32 {
                    config.min_sample_rate()
                } else {
                    config.max_sample_rate()
                };
                config.with_sample_rate(dev_rate)
            });
        let buffer_size = BufferSize::Fixed(match config.buffer_size() {
            cpal::SupportedBufferSize::Range { min, max } => ((config.sample_rate().0 / 30)
                .next_multiple_of(ALSA_BUFFER_QAUANTUM)
                .max(ALSA_BUFFER_MIN))
            .max(*min)
            .min(*max),
            cpal::SupportedBufferSize::Unknown => (config.sample_rate().0 / 30)
                .next_multiple_of(ALSA_BUFFER_QAUANTUM)
                .max(ALSA_BUFFER_MIN),
        });
        let sample_rate = config.sample_rate();
        let channels = config.channels();
        if channels > 2 {
            return Err(AudioError::InputDeviceUnavailable(format!(
                "{} has more then two channels. Only Mono and Stereo audio is supported",
                name
            )));
        }
        let config = StreamConfig {
            channels,
            sample_rate,
            buffer_size,
        };
        Ok((device, config))
    } else {
        Err(AudioError::InputDeviceUnavailable(name.into()))
    }
}

pub fn get_resampler(src_rate: u32) -> Option<Samplerate> {
    if src_rate != SAMPLE_RATE as u32 {
        eprintln!(
            "running with resampling src{:?}->dest{SAMPLE_RATE}",
            src_rate
        );
        let resampler = Samplerate::new(
            samplerate::ConverterType::SincFastest,
            src_rate,
            SAMPLE_RATE as u32,
            1,
        )
        .expect("should be able to build resampler");
        Some(resampler)
    } else {
        None
    }
}

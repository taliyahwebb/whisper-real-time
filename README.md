# Whisper Real Time
Provides an interface to openai-whisper for continuous transcription of the default input device

- [System Requirements](#requirements)
- [Usage](#usage)
  - [Where to find models](#where-to-find-models)
- [License](#license)

## Requirements
### NixOS
This repository provides a [Nix flake](https://nixos.wiki/wiki/flakes) which provides:
- [Development Environment](https://nixos.wiki/wiki/Development_environment_with_nix-shell) via `nix develop`
- Nix Package as the default flake package output
  - can be built with `nix build` (binary will be available as `./result/bin/whisper-real-time`)

The Development Environment provides all needed libraries to build the project.
Note: [Runtime Dependencies](#runtime-dependencies)


### Other Linux
Requires the following for building:
- [rust](https://www.rust-lang.org/)
- [cmake](https://cmake.org/)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) (`pkgconf` on arch-linux)
- [shaderc](https://github.com/google/shaderc)
- [clang](https://clang.llvm.org/)
These are required during linking:
- [alsa-lib](https://github.com/alsa-project/alsa-lib)
- [vulkan-headers](https://github.com/KhronosGroup/Vulkan-Headers)
- [vulkan-loader](https://wiki.archlinux.org/title/Vulkan) (`vulkan-icd-loader` on arch-linux)

List of archlinux packages: `rust cmake pkgconf shaderc alsa-lib vulkan-headers vulkan-icd-loader`

Then build using `cargo build --release` (binary will be produced at `./target/release/whisper-real-time`)

Note: [Runtime Dependencies](#runtime-dependencies)

### Runtime Dependencies
The default mode of operation assumes a Vulkan ready graphics driver is installed and will crash at runtime if none is present.
- NixOS: if you have a graphical user environment enabled and your hardware supports vulkan it will likely just work.
- ArchLinux: [Vulkan](https://wiki.archlinux.org/title/Vulkan)

Here is a list of [Vulkan ready devices](https://vulkan.gpuinfo.org/). Most modern Graphics drivers should support Vulkan.

If your device does not support vulkan, you could refer to [Using special hardware](#using-special-hardware) and run the application without using the builtin Vulkan functionality.

## Usage
To run with the system default input device
```bash
whisper-real-time --model "./path-to-ggml-model.bin"
```

### Where to find models
This project requires models in [`ggml` format](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#ggml-format).
You can download them [here](https://huggingface.co/ggerganov/whisper.cpp/tree/main) or follow these [instructions](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#ggml-format).

Personally i've had the best success with the `base.en-q5_1` model. [ggml-base.en-q5_1.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin?download=true)


### Suppressing Unwanted output
There is a lot of "debug" output on stderr to suppress it
```bash
whisper-real-time --model "./path-to-ggml-model.bin" 2>/dev/null
```

### Running the pipeline on an audio file

- WIP: currently the VAD pipeline is not run when using file mode

If you want to run the [VAD](https://en.wikipedia.org/wiki/Voice_activity_detection) pipeline used on a file simply add ``-f "audiofile.wav"``
```bash
whisper-real-time --model "./path-to-ggml-model.bin" --file "audiofile.wav" 2>/dev/null
```
Currently only WAV files are supported. see [ffmpeg](https://ffmpeg.org/) for file conversion


### Using special hardware
If the library included doesn't support your hardware you can build [whisper.cpp](https://github.com/ggerganov/whisper.cpp) yourself and supply the binary supporting your hardware like so
```bash
whisper-real-time --model "./path-to-ggml-model.bin" --whisper-cpp "./path-to-whisper.cpp-cli-binary"
```

## License
[MIT License](LICENSE)

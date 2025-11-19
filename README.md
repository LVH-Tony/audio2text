# Audio to Text Converter

A simple Python application that converts audio files (M4A, MP3, WAV, etc.) to text using OpenAI's Whisper model running locally on your CPU.

## Features

- 🎤 Converts audio files to text (M4A, MP3, WAV, MP4, WebM, OGG, FLAC, AAC)
- 🖥️ Runs entirely on local CPU (no cloud services required)
- 🌍 Supports multiple languages (auto-detection available)
- 📝 Multiple model sizes for accuracy vs speed tradeoff
- 💾 Saves transcription to text file

## Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed (required for audio processing)

### Installing FFmpeg

- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Note: The first time you run the app, Whisper will download the model (this is a one-time download).

## Usage

### Basic Usage

Convert an audio file to text:

```bash
python app.py input.m4a
# or
python app.py input.mp3
# or any supported format (WAV, MP4, WebM, OGG, FLAC, AAC)
```

The transcription will be printed to the console and saved to `input.txt` by default.

### Specify Output File

```bash
python app.py input.m4a -o output.txt
```

### Choose Model Size

Whisper offers different model sizes with different accuracy/speed tradeoffs:

- `tiny`: Fastest, least accurate (~39M parameters)
- `base`: Good balance (default, ~74M parameters)
- `small`: Better accuracy (~244M parameters)
- `medium`: High accuracy (~769M parameters)
- `large`: Best accuracy, slowest (~1550M parameters)

```bash
python app.py input.m4a -m small
```

### Specify Language

For better accuracy, you can specify the language:

```bash
python app.py input.m4a -l en
```

Common language codes: `en` (English), `vi` (Vietnamese), `es` (Spanish), `fr` (French), `de` (German), `zh` (Chinese), `ja` (Japanese), etc.

### Keep Intermediate WAV File

By default, the temporary WAV file is deleted after processing. To keep it:

```bash
python app.py input.m4a --keep-wav
```

## Examples

```bash
# Basic conversion
python app.py recording.m4a

# High accuracy conversion
python app.py recording.m4a -m large -o transcript.txt

# Spanish audio
python app.py audio.m4a -l es -m small

# Vietnamese audio (MP3)
python app.py audio.mp3 -l vi -m base

# Any supported format
python app.py recording.wav -l en -m small
```

## Model Download

The first time you use a model size, Whisper will automatically download it. Models are stored in your user cache directory (typically `~/.cache/whisper/` on macOS/Linux).

## Performance Notes

- **CPU Performance**: Processing time depends on your CPU and audio length. Expect roughly:
  - `base` model: ~1-2x real-time (1 minute audio = 1-2 minutes processing)
  - `small` model: ~2-3x real-time
  - `large` model: ~5-10x real-time

- **Memory**: Larger models require more RAM:
  - `base`: ~1GB RAM
  - `small`: ~2GB RAM
  - `large`: ~10GB RAM

## Troubleshooting

### "ffmpeg not found"

Make sure FFmpeg is installed and available in your PATH. Test with: `ffmpeg -version`

### "No module named 'whisper'"

Run: `pip install openai-whisper`

### Slow processing

- Use a smaller model (`tiny` or `base`)
- Consider using GPU acceleration if available (requires PyTorch with CUDA)

## License

This project uses OpenAI's Whisper model, which is licensed under the MIT License.

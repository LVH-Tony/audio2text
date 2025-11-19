#!/usr/bin/env python3
"""
Audio to Text Converter
Converts audio files (M4A, MP3, WAV, etc.) to text using local CPU-based Whisper model
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

try:
    import whisper
except ImportError:
    print(
        "Error: whisper library not installed. Please run: pip install openai-whisper"
    )
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub library not installed. Please run: pip install pydub")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm library not installed. Please run: pip install tqdm")
    sys.exit(1)


def get_audio_duration(audio_path):
    """Get audio file duration in seconds"""
    try:
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except Exception:
        # Fallback: try using ffprobe if available
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return None


def convert_m4a_to_wav(m4a_path, wav_path=None):
    """Convert .m4a file to .wav format for processing"""
    if wav_path is None:
        wav_path = m4a_path.with_suffix(".wav")

    print(f"Converting {m4a_path.name} to WAV format...")
    try:
        with tqdm(
            total=100,
            desc="Converting",
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            audio = AudioSegment.from_file(str(m4a_path), format="m4a")
            pbar.update(50)
            audio.export(str(wav_path), format="wav")
            pbar.update(50)
        return wav_path
    except Exception as e:
        print(f"Error converting M4A to WAV: {e}")
        print("This might indicate the file is corrupted or incomplete.")
        raise


def transcribe_audio(audio_path, model_size="base", language=None, duration=None):
    """
    Transcribe audio file to text using Whisper

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., 'en') or None for auto-detection
        duration: Audio duration in seconds (for progress estimation)
    """
    # Model loading progress
    with tqdm(
        total=100,
        desc="Loading model",
        unit="%",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as pbar:
        model = whisper.load_model(model_size)
        pbar.update(100)

    # Estimate processing time based on model size and audio duration
    # These are rough estimates for CPU processing (real-time multipliers)
    speed_multipliers = {
        "tiny": 0.5,  # ~0.5x real-time (faster than real-time)
        "base": 1.5,  # ~1.5x real-time
        "small": 2.5,  # ~2.5x real-time
        "medium": 5.0,  # ~5x real-time
        "large": 10.0,  # ~10x real-time
    }

    multiplier = speed_multipliers.get(model_size, 2.0)
    estimated_time = duration * multiplier if duration else None

    # Transcription progress
    if estimated_time:
        desc = f"Transcribing {Path(audio_path).name}"
        with tqdm(
            total=100,
            desc=desc,
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            # Start transcription in a thread to allow progress updates
            result_container = [None]
            error_container = [None]

            def transcribe_thread():
                try:
                    result = model.transcribe(
                        str(audio_path),
                        language=language,
                        fp16=False,  # Use CPU (fp16 requires GPU)
                        verbose=False,  # Suppress Whisper's own progress
                    )
                    result_container[0] = result
                except Exception as e:
                    error_container[0] = e

            thread = threading.Thread(target=transcribe_thread)
            thread.start()

            # Simulate progress (since Whisper doesn't expose real progress)
            # Update progress bar based on elapsed time vs estimated time
            start_time = time.time()
            last_update = 0

            while thread.is_alive():
                elapsed = time.time() - start_time
                if estimated_time:
                    progress = min(int((elapsed / estimated_time) * 100), 99)
                    if progress > last_update:
                        pbar.update(progress - last_update)
                        last_update = progress
                else:
                    # If no duration estimate, just show elapsed time
                    pbar.update(1)
                time.sleep(0.5)

            thread.join()

            # Complete the progress bar
            pbar.update(100 - last_update)

            if error_container[0]:
                raise error_container[0]

            return result_container[0]
    else:
        # Fallback if we can't estimate time
        desc = f"Transcribing {Path(audio_path).name}"
        with tqdm(
            total=100,
            desc=desc,
            unit="%",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            result = model.transcribe(
                str(audio_path), language=language, fp16=False, verbose=False
            )
            pbar.update(100)
            return result


def process_audio_file(
    input_path, output_path=None, model_size="base", language=None, keep_wav=False
):
    """
    Process an audio file and convert it to text

    Args:
        input_path: Path to input audio file (M4A, MP3, WAV, etc.)
        output_path: Path to output text file (optional)
        model_size: Whisper model size
        language: Language code or None for auto-detection
        keep_wav: Whether to keep the intermediate WAV file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    # Supported formats that Whisper can handle directly
    supported_formats = {
        ".m4a",
        ".mp3",
        ".wav",
        ".mp4",
        ".webm",
        ".ogg",
        ".flac",
        ".aac",
    }
    file_ext = input_path.suffix.lower()

    if file_ext not in supported_formats:
        print(f"Note: File format '{file_ext}' may not be directly supported.")
        print("Attempting to process anyway (Whisper may still handle it)...")

    # Try to use Whisper directly first (it supports many formats)
    # Only convert to WAV if direct processing fails for certain formats
    audio_path = input_path
    wav_path = None
    needs_conversion = False

    # Formats that might need conversion as fallback
    formats_may_need_conversion = {".m4a", ".mp3"}

    if file_ext in formats_may_need_conversion:
        print(f"Processing {input_path.name}...")
        print(
            f"(Whisper supports {file_ext.upper()} format, so conversion may not be needed)"
        )

        # Try direct processing first, but have conversion as fallback
        # We'll attempt conversion only if direct processing fails
        needs_conversion = True  # Flag to attempt conversion if needed
        wav_path = input_path.with_suffix(".wav")
    else:
        print(f"Processing {input_path.name}...")

    # Get audio duration for progress estimation
    print("Analyzing audio file...")
    duration = get_audio_duration(audio_path)
    if duration:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Audio duration: {minutes}m {seconds}s")
    else:
        print(
            "Could not determine audio duration (progress estimation may be less accurate)"
        )

    try:
        # Try transcribing directly first (Whisper supports many formats)
        try:
            result = transcribe_audio(audio_path, model_size, language, duration)
        except Exception as direct_error:
            # If direct processing fails, try converting to WAV first (for M4A/MP3)
            if needs_conversion and file_ext in {".m4a", ".mp3"}:
                print(f"\nDirect processing failed: {direct_error}")
                print("Attempting to convert to WAV format and retry...")
                try:
                    # Use pydub to convert - it handles both M4A and MP3
                    audio = AudioSegment.from_file(
                        str(input_path), format=file_ext[1:]
                    )  # Remove the dot
                    audio.export(str(wav_path), format="wav")
                    audio_path = wav_path
                    # Re-get duration for converted file
                    duration = get_audio_duration(audio_path)
                    result = transcribe_audio(
                        audio_path, model_size, language, duration
                    )
                except Exception as convert_error:
                    print(f"\nError: Could not process the audio file.")
                    print(f"Direct processing error: {direct_error}")
                    print(f"Conversion error: {convert_error}")
                    print("\nPossible issues:")
                    print("  - File may be corrupted or incomplete")
                    print("  - File format may not be supported")
                    print(f"  - Try checking the file with: ffmpeg -i {input_path}")
                    raise convert_error
            else:
                raise direct_error

        # Get the transcribed text
        text = result["text"].strip()

        # Print to console
        # print("\n" + "="*60)
        # print("TRANSCRIPTION RESULT:")
        # print("="*60)
        # print(text)
        # print("="*60 + "\n")

        # Save to file if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Transcription saved to: {output_path}")

        # Clean up temporary WAV file if created
        if needs_conversion and not keep_wav and wav_path and wav_path.exists():
            wav_path.unlink()
            print(f"Cleaned up temporary file: {wav_path}")

        return True

    except Exception as e:
        print(f"Error during transcription: {e}")
        return False
    finally:
        # Clean up WAV file on error too
        if needs_conversion and not keep_wav:
            temp_wav_path = input_path.with_suffix(".wav")
            if temp_wav_path.exists():
                temp_wav_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files (M4A, MP3, WAV, etc.) to text using local CPU-based Whisper model"
    )
    parser.add_argument(
        "input", help="Path to input audio file (M4A, MP3, WAV, MP4, etc.)"
    )
    parser.add_argument("-o", "--output", help="Path to output text file (optional)")
    parser.add_argument(
        "-m",
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base). Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the intermediate WAV file after processing",
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = input_path.with_suffix(".txt")

    success = process_audio_file(
        args.input, args.output, args.model, args.language, args.keep_wav
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

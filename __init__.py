import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch
import folder_paths
from scipy.io.wavfile import write

class SaveAudioAsMP3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bitrate_kbps": ([64, 128, 192, 256, 320], {"default": 192}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "save_as_mp3"
    OUTPUT_NODE = True
    CATEGORY = "LikeSpiderAI"

    def save_as_mp3(self, audio, bitrate_kbps):
        if not shutil.which("ffmpeg"):
            raise Exception("❌ ffmpeg not found. Please install and add to PATH.")

        # Extract waveform
        samples = audio["waveform"]
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples[0]

        max_val = np.max(np.abs(samples))
        if max_val > 1.0:
            samples = samples / max_val
        samples = np.clip(samples, -1.0, 1.0)
        audio_int16 = (samples * 32767).astype(np.int16)

        if audio_int16.ndim == 2 and audio_int16.shape[0] in [1, 2]:
            wav_data = audio_int16.T
        elif audio_int16.ndim == 1:
            wav_data = audio_int16
        else:
            raise Exception(f"❌ Unsupported audio shape: {audio_int16.shape}")

        # Save as .mp3 in output/audio
        output_dir = Path(folder_paths.get_output_directory()) / "audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob("audio_*.mp3"))
        index = len(existing)
        mp3_path = output_dir / f"audio_{index:05d}.mp3"

        # Save .wav temporarily for ffmpeg conversion
        temp_wav = output_dir / "temp_input.wav"
        write(temp_wav, 44100, wav_data)

        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_wav),
            "-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
            str(mp3_path)
        ], check=True)
        os.remove(temp_wav)

        print(f"[SaveAudioAsMP3] ✅ MP3 saved: {mp3_path}")

        # Return original audio to enable UI preview (if possible)
        return (audio,)

NODE_CLASS_MAPPINGS = {
    "SaveAudioAsMP3": SaveAudioAsMP3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioAsMP3": "Save Audio As MP3"
}

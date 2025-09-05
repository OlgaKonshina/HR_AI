import os
import sounddevice as sd
import soundfile as sf


def load_audio(duration: int = 25, folder: str = "audio/answers") -> str:
    os.makedirs(folder, exist_ok=True)

    existing = [f for f in os.listdir(folder) if f.startswith("answer_") and f.endswith(".ogg")]
    next_index = len(existing) + 1

    filename = os.path.join(folder, f"answer_{next_index}.ogg")

    print(f"время для ответа : {duration} секунд...")

    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write(filename, audio, 44100)

    return filename


def load_audio_(question_id: int, duration: int = 30, folder: str = "audio/answers") -> str:
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"answer_{question_id}.ogg")

    print(f"время ответа: {duration} секунд...")

    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()
    sf.write(filename, audio, 44100)

    return filename

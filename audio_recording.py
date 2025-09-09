import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import tempfile


def load_audio(duration: int = 25, folder: str = "audio/answers") -> str:
    os.makedirs(folder, exist_ok=True)

    existing = [f for f in os.listdir(folder) if f.startswith("answer_") and f.endswith(".wav")]
    next_index = len(existing) + 1

    filename = os.path.join(folder, f"answer_{next_index}.wav")

    print(f"🎙️ Запись звука: {duration} секунд...")

    try:
        # Записываем аудио
        sample_rate = 44100
        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()  # Ждем завершения записи

        # Сохраняем в WAV формате (более надежно)
        sf.write(filename, audio, sample_rate)
        print(f"✅ Аудио сохранено: {filename}")

        return filename

    except Exception as e:
        print(f"❌ Ошибка записи аудио: {e}")

        # Создаем заглушку для тестирования
        dummy_audio = np.zeros(44100 * 2)  # 2 секунды тишины
        sf.write(filename, dummy_audio, 44100)
        print(f"⚠️ Создан тестовый файл: {filename}")

        return filename


def load_audio(duration: int = 25, folder: str = "audio/answers") -> str:
    print(f"🎙️ Запись {duration} секунд...")

    try:
        # Создаем временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_filename = tmp.name

        # Записываем аудио
        sample_rate = 44100
        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1)
        sd.wait()

        # Сохраняем
        sf.write(temp_filename, audio, sample_rate)

        # Копируем в конечную папку
        os.makedirs(folder, exist_ok=True)
        final_filename = os.path.join(folder, f"answer_{len(os.listdir(folder)) + 1}.wav")
        import shutil
        shutil.copy2(temp_filename, final_filename)

        # Удаляем временный файл
        os.unlink(temp_filename)

        print(f"✅ Запись успешна: {final_filename}")
        return final_filename

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        # Возвращаем заглушку
        return "audio/answers/fallback.wav"


# Функция для проверки аудио устройств
def list_audio_devices():
    """Показать доступные аудио устройства"""
    try:
        devices = sd.query_devices()
        print("🎧 Доступные аудио устройства:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (входов: {device['max_input_channels']})")
        return devices
    except Exception as e:
        print(f"❌ Ошибка получения устройств: {e}")
        return []


# Функция для тестирования записи
def test_recording():
    """Тестовая функция для проверки записи"""
    print("🔊 Тестирование записи...")
    try:
        # Показываем устройства
        list_audio_devices()

        # Пробуем записать 3 секунды
        test_file = "test_recording.wav"
        print("🎤 Говорите сейчас...")

        sample_rate = 44100
        audio = sd.rec(int(3 * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
        sd.wait()

        sf.write(test_file, audio, sample_rate)
        print(f"✅ Тестовая запись сохранена: {test_file}")

        return True
    except Exception as e:
        print(f"❌ Тест failed: {e}")
        return False


if __name__ == "__main__":
    # Запустить тест при прямом выполнении файла
    test_recording()

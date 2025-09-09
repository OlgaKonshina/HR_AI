from config import YANDEX_API_KEY, YANDEX_FOLDER_ID
import os
import requests
import whisper
import pygame
import time
import subprocess
import sys

# Данные для Яндекс SpeechKit
API_KEY = YANDEX_API_KEY
FOLDER_ID = YANDEX_FOLDER_ID

# Глобальная переменная для модели Whisper
whisper_model = None


def load_whisper_model():
    """Загружает модель Whisper один раз при старте"""
    global whisper_model
    if whisper_model is None:
        try:
            print("🔄 Загрузка модели Whisper...")
            whisper_model = whisper.load_model('base')
            print("✅ Модель Whisper загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки Whisper: {e}")
            whisper_model = None
    return whisper_model


def recognize_audio_whisper(audio_file):
    """Распознавание речи через Whisper (оффлайн)"""
    try:
        # Проверяем файл
        if not os.path.exists(audio_file):
            return "Аудиофайл не найден"

        if os.path.getsize(audio_file) < 1000:
            return "Запись слишком короткая"

        # Загружаем модель
        model = load_whisper_model()
        if model is None:
            return "Модель распознавания не доступна"

        # Распознаем речь
        result = model.transcribe(audio_file, fp16=False, language='ru')
        print(f"✅ Распознано: {result['text']}")
        return result['text']

    except Exception as e:
        print(f"❌ Ошибка распознавания Whisper: {e}")
        return "Ошибка распознавания речи"

def text_to_ogg(text: str, folder: str = "audio/questions") -> str:
    os.makedirs(folder, exist_ok=True)

    existing = [f for f in os.listdir(folder) if f.startswith("question_") and f.endswith(".ogg")]
    next_index = len(existing) + 1

    filename = os.path.join(folder, f"question_{next_index}.ogg")
    url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
    headers = {'Authorization': f'Api-Key {API_KEY}'}

    data = {
        'folderId': FOLDER_ID,
        'text': text,
        'lang': 'ru-RU',
        'voice': 'zahar',
        'speed': '1.5',
        'format': 'oggopus',
        'sampleRateHertz': 48000,
    }

    # Отправляем запрос и сохраняем аудио
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Аудио сохранено в OGG: {filename}")

        # Воспроизводим аудио с помощью pygame (кроссплатформенное решение)
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"⚠️ Ошибка воспроизведения через pygame: {e}")
            # Альтернативный способ воспроизведения
            try:
                if sys.platform == "win32":
                    os.startfile(filename)
                elif sys.platform == "darwin":  # macOS
                    subprocess.call(("open", filename))
                else:  # Linux
                    subprocess.call(("aplay", filename))
                return True
            except Exception as e2:
                print(f"⚠️ Ошибка альтернативного воспроизведения: {e2}")
                return False
    else:
        print(f"❌ Ошибка синтеза речи: {response.status_code} - {response.text}")
        return False


def recognize_audio(audio_file, language='ru-RU'):
    API_KEY = YANDEX_API_KEY
    FOLDER_ID = YANDEX_FOLDER_ID

    url = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize'
    headers = {'Authorization': f'Api-Key {API_KEY}'}

    # Проверяем существование файла
    if not os.path.exists(audio_file):
        print(f"❌ Файл {audio_file} не найден!")
        return None

    try:
        # Читаем аудио файл
        with open(audio_file, 'rb') as f:
            audio_data = f.read()

        # Отправляем запрос
        response = requests.post(
            url,
            headers=headers,
            data=audio_data,
            params={
                'folderId': FOLDER_ID,
                'lang': language,
            }
        )

        # Обрабатываем ответ
        if response.status_code == 200:
            result = response.json()
            return result.get('result', '')
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Детали: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Ошибка при обработке файла: {e}")
        return None


def recognize_audio_whisper(audio_file):
    try:
        # Проверяем существование файла
        if not os.path.exists(audio_file):
            print(f"❌ Аудио файл не найден: {audio_file}")
            return "Файл не найден"

        # Проверяем размер файла
        file_size = os.path.getsize(audio_file)
        if file_size == 0:
            print(f"⚠️ Пустой аудио файл: {audio_file}")
            return "Пустая запись"

        model = whisper.load_model('base')
        result = model.transcribe(audio_file, fp16=False)
        print(f"✅ Распознано: {result['text']}")
        return result['text']
    except Exception as e:
        print(f"❌ Ошибка распознавания Whisper: {e}")
        return "Не удалось распознать речь"
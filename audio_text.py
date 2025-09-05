from config import YANDEX_API_KEY, YANDEX_FOLDER_ID
import os
import requests
import whisper
import pygame
import time
from playsound import playsound
# Данные для Яндекс SpeechKit
API_KEY = YANDEX_API_KEY
FOLDER_ID = YANDEX_FOLDER_ID


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
        'format': 'oggopus',  # Формат OGG Opus
        'sampleRateHertz': 48000,
    }

    # Отправляем запрос и сохраняем аудио
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
            playsound(filename)
        print(f"Аудио сохранено в OGG: {filename}")
        return True
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")
        return False


def recognize_audio(audio_file, language='ru-RU'):

    API_KEY = YANDEX_API_KEY
    FOLDER_ID = YANDEX_FOLDER_ID

    url = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize'
    headers = {'Authorization': f'Api-Key {API_KEY}'}

    # Проверяем существование файла
    if not os.path.exists(audio_file):
        print(f"Файл {audio_file} не найден!")
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
            print(f"Ошибка API: {response.status_code}")
            print(f"Детали: {response.text}")
            return None

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return None


def recognize_audio_whisper(audio):
    model = whisper.load_model('turbo')
    result = model.transcribe(audio, fp16=False)  # добавляем аудио для обработки
    print(result['text'])
    return result['text']

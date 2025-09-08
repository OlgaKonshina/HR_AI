FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директории для аудио
RUN mkdir -p audio/questions audio/answers

# Открываем порт
EXPOSE 8501

# Запускаем Streamlit
CMD ["streamlit", "run", "app_5.py", "--server.port=8501", "--server.address=0.0.0.0"]
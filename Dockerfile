FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости включая PostgreSQL клиент
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    libpq-dev \  # Добавляем для работы с PostgreSQL
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директории для аудио
RUN mkdir -p audio/questions audio/answers

EXPOSE 8501

# Запускаем Streamlit
CMD ["streamlit", "run", "web_app_1.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем компилятор и системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ЯВНО ОБНОВЛЯЕМ STREAMLIT ПЕРВЫМ ДЕЛОМ
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit==1.28.0

# Затем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаем директории для аудио
RUN mkdir -p audio/questions audio/answers

EXPOSE 8501

# Запускаем Streamlit
CMD ["streamlit", "run", "app_5.py", "--server.port=8501", "--server.address=0.0.0.0"]
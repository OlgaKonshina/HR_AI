from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from config import DATABASE_URL

def init_db():
    """Упрощенная инициализация базы данных"""
    try:
        if not DATABASE_URL:
            print("DATABASE_URL не настроен")
            return None

        print("Пытаемся подключиться к БД...")

        # Упрощенное подключение
        engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=0)  # Ограничиваем пул соединений

        # Быстрая проверка подключения
        with engine.connect() as conn:
            conn.execute("SELECT 1")  # Простой запрос для проверки

        print("✅ Успешное подключение к БД")
        return engine

    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
        return None
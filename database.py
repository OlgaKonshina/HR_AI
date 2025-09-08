import os
import psycopg2
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from config import DATABASE_URL
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class Interview(Base):
    __tablename__ = 'interviews'

    id = Column(String, primary_key=True)
    candidate_data = Column(JSON)
    job_description = Column(Text)
    hr_email = Column(String)
    interview_link = Column(String)
    status = Column(String, default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    report = Column(Text, nullable=True)


def init_db():
    """Инициализация базы данных с обработкой ошибок"""
    try:
        if not DATABASE_URL:
            logger.error("DATABASE_URL не настроен")
            return None

        logger.info(f"Подключаемся к БД: {DATABASE_URL[:30]}...")

        engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=0)

        # Проверяем соединение
        with engine.connect() as conn:
            conn.execute("SELECT 1")

        Base.metadata.create_all(engine)
        logger.info("✅ Таблицы БД созданы/проверены")
        return engine

    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        return None


def get_session(engine):
    """Создает сессию для работы с БД - ДОБАВЬТЕ ЭТУ ФУНКЦИЮ"""
    if not engine:
        logger.error("Движок БД не предоставлен")
        return None
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info("✅ Сессия БД создана")
        return session
    except Exception as e:
        logger.error(f"❌ Ошибка создания сессии: {e}")
        return None


def create_interview_in_db(session, interview_id, candidate_data, job_description, hr_email, interview_link):
    """Создает запись интервью в базе данных"""
    if not session:
        logger.error("Сессия БД не доступна")
        return False

    try:
        expires_at = datetime.utcnow() + timedelta(days=7)

        interview = Interview(
            id=interview_id,
            candidate_data=candidate_data,
            job_description=job_description,
            hr_email=hr_email,
            interview_link=interview_link,
            expires_at=expires_at
        )

        session.add(interview)
        session.commit()
        logger.info(f"✅ Интервью {interview_id} сохранено в БД")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Ошибка сохранения интервью: {e}")
        return False


def get_interview_from_db(session, interview_id):
    """Получает интервью из базы данных по ID"""
    if not session:
        logger.error("Сессия БД не доступна")
        return None

    try:
        interview = session.query(Interview).filter_by(id=interview_id).first()

        if interview and interview.expires_at < datetime.utcnow():
            interview.status = 'expired'
            session.commit()
            logger.warning(f"⚠️ Интервью {interview_id} просрочено")
            return None

        return interview
    except Exception as e:
        logger.error(f"❌ Ошибка получения интервью: {e}")
        return None


def update_interview_report(session, interview_id, report):
    """Обновляет отчет интервью в базе данных"""
    if not session:
        logger.error("Сессия БД не доступна")
        return False

    try:
        interview = session.query(Interview).filter_by(id=interview_id).first()
        if interview:
            interview.report = report
            interview.status = 'completed'
            session.commit()
            logger.info(f"✅ Отчет для интервью {interview_id} обновлен")
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Ошибка обновления отчета: {e}")
        return False


def cleanup_expired_interviews(session):
    """Очищает просроченные интервью"""
    if not session:
        return 0
    try:
        expired_count = session.query(Interview).filter(
            Interview.expires_at < datetime.utcnow(),
            Interview.status != 'expired'
        ).update({'status': 'expired'})

        session.commit()
        return expired_count
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Ошибка очистки просроченных интервью: {e}")
        return 0
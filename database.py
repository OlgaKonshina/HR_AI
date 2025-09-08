import os
import psycopg2
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from config import DATABASE_URL

Base = declarative_base()


class Interview(Base):
    __tablename__ = 'interviews'

    id = Column(String, primary_key=True)
    candidate_data = Column(JSON)  # Храним все данные кандидата
    job_description = Column(Text)
    hr_email = Column(String)
    interview_link = Column(String)
    status = Column(String, default='pending')  # pending, completed, expired
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    report = Column(Text, nullable=True)


def init_db():
    """Инициализация базы данных"""
    try:
        engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(engine)
        return engine
    except Exception as e:
        print(f"Ошибка инициализации БД: {e}")
        return None


def get_session(engine):
    """Создает сессию для работы с БД"""
    Session = sessionmaker(bind=engine)
    return Session()


def create_interview_in_db(session, interview_id, candidate_data, job_description, hr_email, interview_link):
    """Создает запись интервью в базе данных"""
    try:
        # Срок действия - 7 дней
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
        return True
    except Exception as e:
        session.rollback()
        print(f"Ошибка создания интервью в БД: {e}")
        return False


def get_interview_from_db(session, interview_id):
    """Получает интервью из базы данных по ID"""
    try:
        interview = session.query(Interview).filter_by(id=interview_id).first()

        # Проверяем не истекла ли ссылка
        if interview and interview.expires_at < datetime.utcnow():
            interview.status = 'expired'
            session.commit()
            return None

        return interview
    except Exception as e:
        print(f"Ошибка получения интервью из БД: {e}")
        return None


def update_interview_report(session, interview_id, report):
    """Обновляет отчет интервью в базе данных"""
    try:
        interview = session.query(Interview).filter_by(id=interview_id).first()
        if interview:
            interview.report = report
            interview.status = 'completed'
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Ошибка обновления отчета в БД: {e}")
        return False


def cleanup_expired_interviews(session):
    """Очищает просроченные интервью (можно запускать периодически)"""
    try:
        expired_count = session.query(Interview).filter(
            Interview.expires_at < datetime.utcnow(),
            Interview.status != 'expired'
        ).update({'status': 'expired'})

        session.commit()
        return expired_count
    except Exception as e:
        session.rollback()
        print(f"Ошибка очистки просроченных интервью: {e}")
        return 0
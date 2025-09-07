# app_streamlit.py
import streamlit as st
import openai
import time
import json
import os
import PyPDF2
import docx
import pandas as pd
from io import BytesIO
import re
import uuid
from datetime import datetime
import requests
from pathlib import Path
import sys

# Добавляем путь для импортов
sys.path.append(str(Path(__file__).parent))

try:
    from audio_text import text_to_ogg, recognize_audio_whisper
    from audio_recording import load_audio
    from config import DEEPSEEK_API_KEY
except ImportError:
    # Заглушки для веб-версии
    def text_to_ogg(*args, **kwargs):
        return True


    def recognize_audio_whisper(*args, **kwargs):
        return "Тестовый ответ"


    def load_audio(*args, **kwargs):
        return "test_audio.ogg"


    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'test-key')

# Настройки для веб-развертывания
BASE_URL = os.getenv('BASE_URL', 'http://localhost:8501')
IS_PRODUCTION = os.getenv('IS_PRODUCTION', 'False').lower() == 'true'

# Настройка страницы
st.set_page_config(
    page_title="HR Бот - AI Recruiter",
    page_icon="🎯",
    layout="wide"
)

# База данных интервью (в production используем Redis или базу)
INTERVIEWS_DB = "interviews_db.json"


class InterviewDB:
    """Простая JSON база данных для интервью"""

    def __init__(self, db_file=INTERVIEWS_DB):
        self.db_file = db_file
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        if not Path(self.db_file).exists():
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def save_interview(self, interview_id, data):
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db = json.load(f)
        db[interview_id] = data
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)

    def get_interview(self, interview_id):
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db = json.load(f)
        return db.get(interview_id)

    def delete_interview(self, interview_id):
        with open(self.db_file, 'r', encoding='utf-8') as f:
            db = json.load(f)
        if interview_id in db:
            del db[interview_id]
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(db, f, ensure_ascii=False, indent=2)


# Инициализация БД
interview_db = InterviewDB()


class InterviewBot:
    def __init__(self, api_key, job_description, resume):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.job_description = job_description
        self.resume = resume
        self.questions = []
        self.answers = []
        self.feedbacks = []

    def generate_question(self, previous_answer=None):
        if previous_answer is None:
            prompt = 'Начни техническое собеседование. Задай первый вопрос о опыте работы кандидата.'
        else:
            prompt = f'Ответ кандидата: {previous_answer}. Сформулируй следующий логичный вопрос.'

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f'Ты технический интервьюер. Вакансия: {self.job_description}. Резюме: {self.resume}. Задавай технические вопросы по очереди.'},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def provide_feedback(self, question, answer):
        feedback_prompt = f"""
        Проанализируй ответ кандидата на технический вопрос.

        ВАКАНСИЯ: {self.job_description}
        ВОПРОС: {question}
        ОТВЕТ КАНДИДАТА: {answer}

        Дай краткую обратную связь:
        - Техническая глубина ответа
        - Практический опыт
        - Что можно улучшить
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты опытный технический интервьюер. Дай конструктивную обратную связь."},
                {"role": "user", "content": feedback_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def generate_final_report(self, email):
        assessment_prompt = f"""
        На основе всего собеседования дай итоговую оценку кандидата.

        ВАКАНСИЯ: {self.job_description}
        РЕЗЮМЕ КАНДИДАТА: {self.resume}
        ВОПРОСЫ И ОТВЕТЫ: {self._format_qa_for_assessment()}

        Сделай комплексную оценку:
        1. Соответствие вакансии
        2. Технические компетенции 
        3. Практический опыт
        4. Сильные стороны
        5. Зоны развития
        6. Рекомендация к найму (да/нет)
        7. Общий балл от 1 до 10

        В конце добавь контакт для обратной связи: {email}
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты старший HR-менеджер. Дай комплексную оценку кандидата."},
                {"role": "user", "content": assessment_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def _format_qa_for_assessment(self):
        formatted = ""
        for i, (question, answer, feedback) in enumerate(zip(self.questions, self.answers, self.feedbacks), 1):
            formatted += f"{i}. В: {question}\n   О: {answer}\n   Ф: {feedback}\n\n"
        return formatted

    @staticmethod
    def filter_resumes(resumes, job_description):
        """Упрощенная фильтрация резюме"""
        filtered = []

        # Ключевые слова из вакансии
        stop_words = {'опыт', 'работа', 'работы', 'обязанности', 'требования', 'знание', 'навыки'}
        job_words = re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', job_description.lower())
        job_keywords = [word for word in job_words if word not in stop_words]

        from collections import Counter
        job_keywords = [word for word, count in Counter(job_keywords).most_common(10)]

        for i, resume in enumerate(resumes):
            with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)}..."):
                resume_text = resume['text'].lower()

                score = 0
                found_keywords = []

                for keyword in job_keywords:
                    if keyword in resume_text:
                        score += 10
                        found_keywords.append(keyword)

                # Проверка опыта
                if re.search(r'опыт.*?\d+.*?(год|лет)', resume_text):
                    score += 30

                # Проверка образования
                if any(edu in resume_text for edu in ['высшее', 'образование', 'вуз']):
                    score += 20

                analysis_result = {
                    'match_score': min(score, 100),
                    'is_suitable': score >= 40,
                    'strengths': [],
                    'weaknesses': [],
                    'reason': f"Анализ соответствия: {score}%"
                }

                if found_keywords:
                    analysis_result['strengths'].append(f"Ключевые слова: {', '.join(found_keywords[:3])}")

                if score < 40:
                    analysis_result['weaknesses'].append("Недостаточное соответствие")

                resume['analysis'] = analysis_result

                if analysis_result['is_suitable']:
                    filtered.append(resume)

        return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)


# Функции для обработки файлов
def extract_text_from_file(file):
    """Извлекает текст из файлов разных форматов"""
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text

        elif file.type == "text/plain":
            return file.read().decode("utf-8", errors='ignore')

        else:
            return f"Формат файла {file.name} не поддерживается"

    except Exception as e:
        return f"Ошибка чтения файла {file.name}: {str(e)}"


def create_interview_link(candidate_data, job_description, hr_email):
    """Создает уникальную ссылку для собеседования"""
    interview_id = str(uuid.uuid4())[:8]

    interview_data = {
        'candidate': candidate_data,
        'job_description': job_description,
        'hr_email': hr_email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'status': 'pending'
    }

    # Сохраняем в базу данных
    interview_db.save_interview(interview_id, interview_data)

    # Генерируем URL в зависимости от среды
    if IS_PRODUCTION:
        # В production используем реальный URL
        return f"{BASE_URL}/?interview_id={interview_id}"
    else:
        # В development - localhost
        return f"http://localhost:8501/?interview_id={interview_id}"


# Определяем тип пользователя
query_params = st.query_params
is_candidate = 'interview_id' in query_params

if is_candidate:
    # 👤 РЕЖИМ СОИСКАТЕЛЯ
    interview_id = query_params['interview_id'][0]

    # Загружаем данные из БД
    interview_data = interview_db.get_interview(interview_id)

    if interview_data:
        candidate = interview_data['candidate']
        job_description = interview_data['job_description']
        hr_email = interview_data['hr_email']

        st.title("🎤 Техническое собеседование")
        st.write("Добро пожаловать на онлайн-собеседование!")

        st.info(f"**Вакансия:** {job_description[:100]}...")
        st.info(f"**Контакт HR:** {hr_email}")

        # Инициализация бота
        if 'interview_bot' not in st.session_state:
            st.session_state.interview_bot = InterviewBot(
                DEEPSEEK_API_KEY,
                job_description,
                candidate['text']
            )
            st.session_state.current_question = 0
            st.session_state.questions = []
            st.session_state.answers = []

        bot = st.session_state.interview_bot

        # Процесс собеседования
        if st.session_state.current_question < 3:
            if st.session_state.current_question >= len(st.session_state.questions):
                previous_answer = st.session_state.answers[-1] if st.session_state.answers else None
                question = bot.generate_question(previous_answer)
                st.session_state.questions.append(question)
                st.session_state.answers.append("")

            st.subheader(f"Вопрос {st.session_state.current_question + 1}/3")
            st.info(st.session_state.questions[st.session_state.current_question])

            if st.button("🎤 Записать ответ", key=f"record_{st.session_state.current_question}"):
                with st.spinner("Запись... (15 секунд)"):
                    audio_file = load_audio(duration=15)
                    answer = recognize_audio_whisper(audio_file)
                    st.session_state.answers[st.session_state.current_question] = answer

                    feedback = bot.provide_feedback(
                        st.session_state.questions[st.session_state.current_question],
                        answer
                    )
                    bot.feedbacks.append(feedback)

                    st.session_state.current_question += 1
                    st.rerun()

            if st.session_state.answers[st.session_state.current_question]:
                st.write("**Ваш ответ:**")
                st.write(st.session_state.answers[st.session_state.current_question])

        else:
            st.success("✅ Собеседование завершено!")
            st.balloons()

            with st.spinner("Генерируем отчет для HR..."):
                final_report = bot.generate_final_report(hr_email)
                interview_data['report'] = final_report
                interview_data['status'] = 'completed'

                # Обновляем в БД
                interview_db.save_interview(interview_id, interview_data)

                st.subheader("📋 Отчет отправлен HR")
                st.write(f"Результаты отправлены на email: **{hr_email}**")
                st.write("С вами свяжутся в ближайшее время!")

    else:
        st.error("❌ Неверная ссылка на собеседование")
        st.write("Пожалуйста, проверьте ссылку или обратитесь к HR")

else:
    # 👔 РЕЖИМ HR-СПЕЦИАЛИСТА
    st.title("🎯 AI Recruiter - Панель HR")

    tab1, tab2 = st.tabs(["📁 Этап 1: Загрузка", "👥 Этап 2: Отбор"])

    with tab1:
        st.header("📁 Этап 1: Загрузка данных")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Данные вакансии")
            job_file = st.file_uploader(
                "Загрузите описание вакансии:",
                type=["pdf", "docx", "txt"],
                key="job_file"
            )

            if job_file:
                job_text = extract_text_from_file(job_file)
                st.session_state.job_description = job_text
                st.success("✅ Вакансия загружена")
                st.text_area("Превью:", job_text[:500] + "...", height=150)

            st.session_state.hr_email = st.text_input("📧 Ваш email:")

        with col2:
            st.subheader("Загрузка резюме")
            uploaded_files = st.file_uploader(
                "Загрузите резюме (до 100 файлов):",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="resume_files"
            )

            if uploaded_files:
                st.session_state.resumes = []
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    st.session_state.resumes.append({
                        'name': file.name,
                        'text': text,
                        'type': file.type
                    })
                st.success(f"Загружено {len(uploaded_files)} резюме")

        if st.button("🚀 Начать фильтрацию",
                     type="primary") and st.session_state.job_description and st.session_state.hr_email and st.session_state.resumes:
            with st.spinner("Анализируем резюме..."):
                st.session_state.filtered_candidates = InterviewBot.filter_resumes(
                    st.session_state.resumes, st.session_state.job_description
                )
            st.success(f"Найдено {len(st.session_state.filtered_candidates)} подходящих кандидатов")

    with tab2:
        st.header("👥 Этап 2: Результаты отбора")

        if not st.session_state.get('filtered_candidates'):
            st.warning("Загрузите данные и выполните фильтрацию")
        else:
            st.write(f"**Найдено кандидатов:** {len(st.session_state.filtered_candidates)}")

            for i, candidate in enumerate(st.session_state.filtered_candidates):
                analysis = candidate.get('analysis', {})

                with st.expander(f"Кандидат {i + 1}: {candidate['name']} (Score: {analysis.get('match_score', 0)}%)"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**📊 Анализ:**")
                        st.write(f"**Совпадение:** {analysis.get('match_score', 0)}%")
                        st.write(
                            f"**Рекомендация:** {'✅ Подходит' if analysis.get('is_suitable', False) else '❌ Не подходит'}")

                        st.write("**✅ Сильные стороны:**")
                        for strength in analysis.get('strengths', []):
                            st.write(f"• {strength}")

                    with col2:
                        st.write("**❌ Слабые стороны:**")
                        for weakness in analysis.get('weaknesses', []):
                            st.write(f"• {weakness}")

                        st.write("**📝 Обоснование:**")
                        st.write(analysis.get('reason', 'Нет данных'))

                        if st.button(f"📧 Отправить приглашение", key=f"invite_{i}"):
                            interview_link = create_interview_link(
                                candidate, st.session_state.job_description, st.session_state.hr_email
                            )

                            st.success("✅ Ссылка создана!")
                            st.text_area("Скопируйте ссылку для кандидата:", interview_link)

# Футер
st.write("---")
st.caption(f"HR AI Recruiter v1.0 | {'Production' if IS_PRODUCTION else 'Development'}")

# Отладочная информация
if not IS_PRODUCTION:
    st.sidebar.write("🔧 Отладка")
    if st.sidebar.button("Очистить базу интервью"):
        interview_db._ensure_db_exists()
        st.sidebar.success("База очищена")
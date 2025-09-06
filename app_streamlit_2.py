# app_streamlit.py
import streamlit as st
import openai
import time
import json
import os
import PyPDF2
import docx
from io import BytesIO
import re
import pandas as pd
import uuid
from datetime import datetime
from audio_text import text_to_ogg, recognize_audio_whisper
from audio_recording import load_audio
from config import DEEPSEEK_API_KEY

# Настройка страницы
st.set_page_config(
    page_title="HR Бот - AI Recruiter",
    page_icon="🎯",
    layout="wide"
)

# Глобальное хранилище (в реальном приложении - база данных)
if 'interviews' not in st.session_state:
    st.session_state.interviews = {}
if 'candidates' not in st.session_state:
    st.session_state.candidates = {}


class InterviewBot:
    def __init__(self, api_key, job_description, resume):
        self.client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
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
            return file.read().decode("utf-8")

        else:
            return f"Файл {file.name} не поддерживается"

    except Exception as e:
        return f"Ошибка чтения файла: {str(e)}"


def filter_resumes(resumes, job_description, keywords):
    """Фильтрация резюме по ключевым словам"""
    filtered = []

    for resume in resumes:
        score = 0
        found_keywords = []

        # Проверяем ключевые слова
        for keyword in keywords:
            if re.search(rf'\b{re.escape(keyword.lower())}\b', resume['text'].lower()):
                score += 1
                found_keywords.append(keyword)

        # Проверяем соответствие вакансии
        job_keywords = re.findall(r'\b\w{4,}\b', job_description.lower())
        job_match = sum(1 for word in job_keywords if word in resume['text'].lower())

        resume['score'] = score
        resume['found_keywords'] = found_keywords
        resume['job_match'] = job_match

        if score >= 2:  # Минимум 2 ключевых слова
            filtered.append(resume)

    return sorted(filtered, key=lambda x: x['score'], reverse=True)


def create_interview_link(candidate_data, job_description, hr_email):
    """Создает уникальную ссылку для собеседования"""
    interview_id = str(uuid.uuid4())[:8]

    st.session_state.interviews[interview_id] = {
        'candidate': candidate_data,
        'job_description': job_description,
        'hr_email': hr_email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'status': 'pending'
    }

    # В реальном приложении здесь будет генерация настоящей URL
    return f"https://your-domain.com/interview/{interview_id}"


# Определяем тип пользователя по query parameters
query_params = st.experimental_get_query_params()
is_candidate = 'interview_id' in query_params

if is_candidate:
    # 👤 РЕЖИМ СОИСКАТЕЛЯ - ЭТАП 3
    interview_id = query_params['interview_id'][0]

    if interview_id in st.session_state.interviews:
        interview_data = st.session_state.interviews[interview_id]
        candidate = interview_data['candidate']
        job_description = interview_data['job_description']
        hr_email = interview_data['hr_email']

        st.title("🎤 Техническое собеседование")
        st.write("Добро пожаловать на онлайн-собеседование!")

        st.info(f"**Вакансия:** {job_description[:100]}...")
        st.info(f"**Контакт HR:** {hr_email}")

        # Инициализация бота для собеседования
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
        if st.session_state.current_question < 3:  # 3 вопроса
            if st.session_state.current_question >= len(st.session_state.questions):
                # Генерируем новый вопрос
                previous_answer = st.session_state.answers[-1] if st.session_state.answers else None
                question = bot.generate_question(previous_answer)
                st.session_state.questions.append(question)
                st.session_state.answers.append("")

            st.subheader(f"Вопрос {st.session_state.current_question + 1}/3")
            st.info(st.session_state.questions[st.session_state.current_question])

            # Запись ответа
            if st.button("🎤 Записать ответ", key=f"record_{st.session_state.current_question}"):
                with st.spinner("Запись... (15 секунд)"):
                    audio_file = load_audio(duration=15)
                    answer = recognize_audio_whisper(audio_file)
                    st.session_state.answers[st.session_state.current_question] = answer

                    # Генерируем обратную связь
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
            # Завершение собеседования
            st.success("✅ Собеседование завершено!")
            st.balloons()

            # Генерация отчета
            with st.spinner("Генерируем отчет для HR..."):
                final_report = bot.generate_final_report(hr_email)
                interview_data['report'] = final_report
                interview_data['status'] = 'completed'

                st.subheader("📋 Отчет отправлен HR")
                st.write(f"Результаты собеседования отправлены на email: **{hr_email}**")
                st.write("С вами свяжутся в ближайшее время!")

    else:
        st.error("❌ Неверная ссылка на собеседование")
        st.write("Пожалуйста, проверьте ссылку или обратитесь к HR")

else:
    # 👔 РЕЖИМ HR-СПЕЦИАЛИСТА - ЭТАПЫ 1 и 2
    st.title("🎯 AI Recruiter - Панель HR")

    tab1, tab2 = st.tabs(["📁 Этап 1: Загрузка", "👥 Этап 2: Отбор"])

    with tab1:
        st.header("📁 Этап 1: Загрузка вакансии и резюме")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Данные вакансии")
            job_description = st.text_area(
                "Описание вакансии:",
                height=150,
                placeholder="Python разработчик с опытом работы от 2 лет...",
                key="job_desc"
            )

            hr_email = st.text_input("📧 Ваш email для обратной связи:")

            st.subheader("Ключевые слова для отбора")
            keywords = st.text_area(
                "Ключевые слова (через запятую):",
                height=100,
                placeholder="python, django, flask, sql, git",
                key="keywords"
            )

        with col2:
            st.subheader("Загрузка резюме")
            uploaded_files = st.file_uploader(
                "Загрузите резюме кандидатов:",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True
            )

            if uploaded_files:
                st.session_state.resumes = []
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    st.session_state.resumes.append({
                        'name': file.name,
                        'text': text,
                        'size': file.size
                    })

                st.success(f"Загружено {len(uploaded_files)} резюме")

        if st.button("🚀 Начать фильтрацию", type="primary") and job_description and keywords and hr_email:
            keyword_list = [k.strip() for k in keywords.split(',')]
            st.session_state.filtered_candidates = filter_resumes(
                st.session_state.resumes, job_description, keyword_list
            )
            st.success(f"Найдено {len(st.session_state.filtered_candidates)} подходящих кандидатов")

    with tab2:
        st.header("👥 Этап 2: Отбор кандидатов")

        if not st.session_state.get('filtered_candidates'):
            st.warning("Загрузите резюме и выполните фильтрацию в Этапе 1")
        else:
            for i, candidate in enumerate(st.session_state.filtered_candidates):
                with st.expander(f"Кандидат {i + 1}: {candidate['name']} (score: {candidate['score']})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Найденные ключевые слова:**")
                        st.write(", ".join(candidate['found_keywords']))

                        st.write("**Совпадение с вакансией:**")
                        st.write(f"{candidate['job_match']} ключевых слов")

                    with col2:
                        if st.button(f"Отправить приглашение", key=f"invite_{i}"):
                            interview_link = create_interview_link(
                                candidate, job_description, hr_email
                            )

                            st.success("✅ Ссылка на собеседование создана!")
                            st.text_area("Скопируйте ссылку для кандидата:", interview_link)

                            st.info("Отправьте эту ссылку кандидату по email")

                    st.write("**Превью резюме:**")
                    st.text(candidate['text'][:500] + "..." if len(candidate['text']) > 500 else candidate['text'])

# Футер
st.write("---")
st.caption("AI Recruiter System v2.0 | Двухрежимный интерфейс для HR и соискателей")
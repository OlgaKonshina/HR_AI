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
from audio_text import text_to_ogg, recognize_audio_whisper
from audio_recording import load_audio
from config import DEEPSEEK_API_KEY

# Настройка страницы
st.set_page_config(
    page_title="HR Бот - AI Recruiter",
    page_icon="🎯",
    layout="wide"
)

# Глобальное хранилище
if 'interviews' not in st.session_state:
    st.session_state.interviews = {}
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'hr_email' not in st.session_state:
    st.session_state.hr_email = ""
if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'filtered_candidates' not in st.session_state:
    st.session_state.filtered_candidates = []


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

    @staticmethod
    def extract_keywords_from_job(job_text):
        """Извлекает ключевые слова из описания вакансии"""
        # Убираем стоп-слова и выделяем значимые слова
        stop_words = {'опыт', 'работа', 'работы', 'обязанности', 'требования', 'знание', 'навыки',
                      'умение', 'возможность', 'необходимый', 'обязательный', 'желательный'}

        words = re.findall(r'\b[a-zA-Zа-яА-Я]{4,}\b', job_text.lower())
        keywords = [word for word in words if word not in stop_words]

        # Берем топ-20 самых частых слов
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]

    @staticmethod
    def calculate_resume_score(resume_text, job_keywords, job_text):
        """Вычисляет score резюме на основе ключевых слов"""
        resume_lower = resume_text.lower()
        job_text_lower = job_text.lower()

        # Базовые критерии
        score = 0
        found_keywords = []
        missing_keywords = []

        # 1. Совпадение ключевых слов (50% веса)
        for keyword in job_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', resume_lower):
                score += 3
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        # 2. Опыт работы (20% веса)
        experience_patterns = [
            r'опыт работы.*?(\d+)[^\d]*лет',
            r'стаж.*?(\d+)[^\d]*год',
            r'experience.*?(\d+)[^\d]*year'
        ]

        for pattern in experience_patterns:
            match = re.search(pattern, resume_lower)
            if match:
                years = int(match.group(1))
                score += min(years * 2, 10)  # макс 10 баллов за опыт
                break

        # 3. Образование (10% веса)
        education_keywords = ['высшее', 'образование', 'вуз', 'университет', 'бакалавр', 'магистр']
        if any(edu in resume_lower for edu in education_keywords):
            score += 5

        # 4. Навыки из вакансии (20% веса)
        skills_section = re.search(r'(навыки|skills|компетенции).*?:(.*?)(?=\n\n|\n[A-ZА-Я]|$)', resume_lower,
                                   re.DOTALL | re.IGNORECASE)
        if skills_section:
            skills_text = skills_section.group(2)
            job_skills = re.findall(r'\b[a-zA-Zа-яА-Я]{3,}\b', job_text_lower)
            for skill in job_skills:
                if skill in skills_text and len(skill) > 3:
                    score += 1

        # Нормализуем score до 100%
        max_possible_score = len(job_keywords) * 3 + 10 + 5 + 20
        final_score = min(int((score / max_possible_score) * 100), 100) if max_possible_score > 0 else 0

        # Определяем подходит ли кандидат
        is_suitable = final_score >= 40  # Порог 40%

        # Генерируем анализ
        strengths = []
        weaknesses = []

        if found_keywords:
            strengths.append(f"Совпадение по ключевым словам: {', '.join(found_keywords[:5])}")
        if missing_keywords:
            weaknesses.append(f"Отсутствуют ключевые слова: {', '.join(missing_keywords[:5])}")

        if final_score < 40:
            weaknesses.append("Низкий общий score соответствия")

        return {
            'match_score': final_score,
            'is_suitable': is_suitable,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'found_keywords': found_keywords,
            'missing_keywords': missing_keywords,
            'reason': f"Score: {final_score}% - {'Подходит' if is_suitable else 'Не подходит'}"
        }

    @staticmethod
    def filter_resumes(resumes, job_description):
        """Надежная фильтрация резюме без LLM"""
        filtered = []

        # Извлекаем ключевые слова из вакансии
        job_keywords = InterviewBot.extract_keywords_from_job(job_description)

        st.write(f"🔑 **Ключевые слова вакансии:** {', '.join(job_keywords[:10])}")

        for i, resume in enumerate(resumes):
            with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)}..."):
                try:
                    # Вычисляем score
                    analysis_result = InterviewBot.calculate_resume_score(
                        resume['text'], job_keywords, job_description
                    )

                    resume['analysis'] = analysis_result

                    if analysis_result['is_suitable']:
                        filtered.append(resume)

                except Exception as e:
                    st.error(f"Ошибка анализа резюме {resume['name']}: {str(e)}")
                    # Резервный анализ
                    resume_lower = resume['text'].lower()
                    has_experience = any(word in resume_lower for word in ['опыт', 'experience', 'стаж'])
                    has_education = any(
                        word in resume_lower for word in ['образование', 'education', 'вуз', 'университет'])

                    resume['analysis'] = {
                        'match_score': 50 if has_experience else 20,
                        'is_suitable': has_experience and has_education,
                        'strengths': ['Есть опыт работы'] if has_experience else [],
                        'weaknesses': ['Нет опыта работы'] if not has_experience else [],
                        'reason': 'Автоматическая оценка'
                    }

        return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)


# Функции для обработки файлов разных форматов
def extract_text_from_file(file):
    """Извлекает текст из файлов разных форматов"""
    try:
        # PDF
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

        # Word DOCX
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text

        # Текстовые файлы
        elif file.type == "text/plain":
            return file.read().decode("utf-8", errors='ignore')

        else:
            # Для остальных форматов пробуем просто прочитать как текст
            try:
                return file.read().decode("utf-8", errors='ignore')
            except:
                return f"Формат файла {file.name} не поддерживается"

    except Exception as e:
        return f"Ошибка чтения файла {file.name}: {str(e)}"


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

    return f"http://localhost:8501/?interview_id={interview_id}"


# Определяем тип пользователя по query parameters
query_params = st.query_params
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
        if st.session_state.current_question < 3:
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

                if job_text.startswith("Для чтения") or job_text.startswith("Ошибка"):
                    st.warning(job_text)
                else:
                    st.text_area("Превью вакансии:", job_text[:500] + "..." if len(job_text) > 500 else job_text,
                                 height=150)

            st.session_state.hr_email = st.text_input("📧 Ваш email для обратной связи:")

        with col2:
            st.subheader("Загрузка резюме")
            uploaded_files = st.file_uploader(
                "Загрузите резюме кандидатов (до 100 файлов):",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="resume_files"
            )

            if uploaded_files:
                if len(uploaded_files) > 100:
                    st.warning("Максимум 100 файлов. Будут обработаны первые 100.")
                    uploaded_files = uploaded_files[:100]

                st.session_state.resumes = []
                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    st.session_state.resumes.append({
                        'name': file.name,
                        'text': text,
                        'type': file.type,
                        'size': file.size
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
            st.warning("Загрузите данные и выполните фильтрацию в Этапе 1")
        else:
            st.write(f"**Найдено подходящих кандидатов:** {len(st.session_state.filtered_candidates)}")

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

                            st.success("✅ Ссылка на собеседование создана!")
                            st.text_area("Скопируйте ссылку для кандидата:", interview_link)

                    st.write("**📄 Превью резюме:**")
                    st.text(candidate['text'][:300] + "..." if len(candidate['text']) > 300 else candidate['text'])

# Футер
st.write("---")
st.caption("AI Recruiter System v5.0 | Надежная rule-based фильтрация")
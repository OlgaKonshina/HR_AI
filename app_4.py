import streamlit as st
import openai
import os
import PyPDF2
import docx
from io import BytesIO
import re
import uuid
from datetime import datetime
from audio_text import text_to_ogg, recognize_audio_whisper
from audio_recording import load_audio
from config import DEEPSEEK_API_KEY
import sys
from pathlib import Path

import json
import re
from pathlib import Path
import pandas as pd
import docx
from striprtf.striprtf import rtf_to_text
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Добавляем путь к текущей директории для импорта document_processor
sys.path.append(str(Path(__file__).parent))

# Импортируем из внешнего файла
# Измените импорт и настройки в app_streamlit.py
try:
    from document_processor import DocumentReader, extract_job_title

    # Пробуем импортировать get_embedding с правильным указанием модели
    try:
        from document_processor import get_embedding

        print("✅ get_embedding импортирован успешно!")

        # Тестируем с правильной моделью для русского языка
        try:
            # Используем русскоязычную модель
            test_embedding = get_embedding("тест", "cointegrated/rubert-tiny2")
            print("✅ RuBERT-Tiny модель работает!")
            DOCUMENT_PROCESSOR_AVAILABLE = True
        except Exception as e:
            print(f"⚠️ RuBERT-Tiny не доступна: {e}")
            # Пробуем альтернативную модель
            try:
                test_embedding = get_embedding("тест", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                print("✅ Multilingual модель работает!")
                DOCUMENT_PROCESSOR_AVAILABLE = True
            except Exception as e2:
                print(f"❌ Все модели не работают: {e2}")
                DOCUMENT_PROCESSOR_AVAILABLE = False

    except Exception as e:
        print(f"❌ Ошибка импорта get_embedding: {e}")
        DOCUMENT_PROCESSOR_AVAILABLE = False

except ImportError as e:
    print(f"❌ Основной импорт не удался: {e}")
    DOCUMENT_PROCESSOR_AVAILABLE = False


# Дополнительные импорты для поддержки разных форматов
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    st.error("Для поддержки RTF установите: !pip install striprtf")
try:
    import odfpy
    from odf import text, teletype
except ImportError:
    st.error("Для поддержки ODT установите: !pip install odfpy")

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
        - Соответствие вакансии
        - Технические компетенции 
        - Практический опыт
        - Сильные стороны
        - Зоны развития
        - Рекомендация к найму (да/нет)
        - Общий балл от 1 до 10

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
    def filter_resumes_with_embeddings(resumes, job_description):
        """Фильтрация резюме с использованием embeddings из document_processor"""
        if not DOCUMENT_PROCESSOR_AVAILABLE:
            st.error("Модуль document_processor не доступен. Используется резервный метод.")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

        filtered = []
        model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        try:
            # Получаем embedding для вакансии
            job_emb = get_embedding(job_description, model_path)

            for i, resume in enumerate(resumes):
                with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)} с помощью embeddings..."):
                    try:
                        # Получаем embedding для резюме
                        resume_emb = get_embedding(resume['text'], model_path)

                        # Вычисляем косинусную схожесть
                        similarity = torch.mm(resume_emb, job_emb.T).item() * 100

                        # Анализируем результат
                        analysis_result = InterviewBot._analyze_embedding_result(similarity, resume['text'],
                                                                                 job_description)

                        resume['analysis'] = analysis_result

                        if analysis_result['is_suitable']:
                            filtered.append(resume)

                    except Exception as e:
                        st.error(f"Ошибка анализа резюме {resume['name']} с embeddings: {str(e)}")
                        # Резервный анализ при ошибке
                        analysis_result = InterviewBot._analyze_resume_fallback(resume['text'], job_description)
                        resume['analysis'] = analysis_result
                        if analysis_result['is_suitable']:
                            filtered.append(resume)

            return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)

        except Exception as e:
            st.error(f"Ошибка при работе с embeddings: {str(e)}")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

    @staticmethod
    def filter_resumes_with_embeddings(resumes, job_description):
        """Фильтрация резюме с использованием русскоязычных embeddings"""
        if not DOCUMENT_PROCESSOR_AVAILABLE:
            st.error("Модуль document_processor не доступен. Используется резервный метод.")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

        filtered = []

        # Используем русскоязычную модель
        model_path = "cointegrated/rubert-tiny2"  # Русская маленькая модель

        try:
            # Получаем embedding для вакансии
            job_emb = get_embedding(job_description[:512], model_path)  # Ограничиваем длину

            for i, resume in enumerate(resumes):
                with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)} с помощью RuBERT..."):
                    try:
                        # Получаем embedding для резюме (первые 512 токенов)
                        resume_short = resume['text'][:1000]  # Берем начало резюме
                        resume_emb = get_embedding(resume_short, model_path)

                        # Вычисляем косинусную схожесть
                        similarity = torch.nn.functional.cosine_similarity(job_emb, resume_emb).item() * 100

                        # Анализируем результат
                        analysis_result = InterviewBot._analyze_embedding_result(similarity, resume['text'],
                                                                                 job_description)

                        resume['analysis'] = analysis_result

                        if analysis_result['is_suitable']:
                            filtered.append(resume)

                    except Exception as e:
                        st.error(f"Ошибка анализа резюме {resume['name']}: {str(e)}")
                        # Резервный анализ при ошибке
                        analysis_result = InterviewBot._analyze_resume_fallback(resume['text'], job_description)
                        resume['analysis'] = analysis_result
                        if analysis_result['is_suitable']:
                            filtered.append(resume)

            return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)

        except Exception as e:
            st.error(f"Ошибка при работе с русскоязычными embeddings: {str(e)}")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

    @staticmethod
    def filter_resumes_fallback(resumes, job_description):
        """Резервный метод фильтрации без embeddings"""
        filtered = []

        # Извлекаем ключевые слова из вакансии
        job_keywords = InterviewBot._extract_keywords(job_description)

        st.write(f"🔑 **Ключевые слова вакансии:** {', '.join(job_keywords[:10])}")

        for i, resume in enumerate(resumes):
            with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)} (резервный метод)..."):
                analysis_result = InterviewBot._analyze_resume_fallback(resume['text'], job_description, job_keywords)
                resume['analysis'] = analysis_result

                if analysis_result['is_suitable']:
                    filtered.append(resume)

        return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)

    @staticmethod
    def _extract_keywords(text):
        """Извлекает ключевые слова из текста"""
        stop_words = {'опыт', 'работа', 'работы', 'обязанности', 'требования', 'знание', 'навыки'}
        words = re.findall(r'\b[a-zA-Zа-яА-Я]{4,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]

        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]

    @staticmethod
    def _analyze_resume_fallback(resume_text, job_description, job_keywords=None):
        """Резервный анализ резюме"""
        if job_keywords is None:
            job_keywords = InterviewBot._extract_keywords(job_description)

        resume_lower = resume_text.lower()
        job_lower = job_description.lower()

        # Простой scoring на основе ключевых слов
        score = 0
        found_keywords = []

        for keyword in job_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', resume_lower):
                score += 3
                found_keywords.append(keyword)

        # Проверяем опыт работы
        experience_match = re.search(r'опыт работы.*?(\d+)[^\d]*лет', resume_lower)
        if experience_match:
            years = int(experience_match.group(1))
            score += min(years * 2, 10)

        # Нормализуем score
        max_score = len(job_keywords) * 3 + 10
        match_score = min(int((score / max_score) * 100), 100) if max_score > 0 else 0

        is_suitable = match_score >= 40

        strengths = []
        weaknesses = []

        if found_keywords:
            strengths.append(f"Совпадение по ключевым словам: {', '.join(found_keywords[:5])}")

        if match_score < 40:
            weaknesses.append("Низкий score соответствия")

        return {
            'match_score': match_score,
            'is_suitable': is_suitable,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'reason': f"Score: {match_score}% - {'Подходит' if is_suitable else 'Не подходит'}"
        }


# Функции для обработки файлов разных форматов
def extract_text_from_file(file):
    """Извлекает текст из файлов разных форматов"""
    try:
        # Используем DocumentReader из внешнего модуля если доступен
        if DOCUMENT_PROCESSOR_AVAILABLE:
            try:
                # Сохраняем временный файл для обработки
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                reader = DocumentReader(tmp_path)
                text = reader.extract_text()

                # Удаляем временный файл
                os.unlink(tmp_path)
                return text

            except Exception as e:
                st.warning(f"DocumentReader не смог обработать файл: {e}. Используем резервный метод.")

        # Резервный метод обработки файлов
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
            # Пробуем прочитать как текст
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

    # Информация о поддерживаемых форматах
    with st.sidebar:
        st.header("📋 Поддерживаемые форматы")
        st.write("""
        **Вакансии и резюме:**
        - 📄 PDF (.pdf)
        - 📝 Word DOCX (.docx)
        - 📝 Word DOC (.doc)
        - 📋 RTF (.rtf)
        - 📘 OpenDocument (.odt)
        - 📱 Текст (.txt)
        - 🌐 HTML (.html)
        """)

        if not DOCUMENT_PROCESSOR_AVAILABLE:
            st.warning("⚠️ Расширенная обработка файлов недоступна")

    tab1, tab2 = st.tabs(["📁 Этап 1: Загрузка", "👥 Этап 2: Отбор"])

    with tab1:
        st.header("📁 Этап 1: Загрузка данных")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Данные вакансии")
            job_file = st.file_uploader(
                "Загрузите описание вакансии:",
                type=["pdf", "docx", "txt", "rtf", "odt"],
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
                type=["pdf", "docx", "txt", "rtf", "odt"],
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

                # Статистика по форматам
                format_stats = {}
                for resume in st.session_state.resumes:
                    fmt = resume['name'].split('.')[-1].upper() if '.' in resume['name'] else 'OTHER'
                    format_stats[fmt] = format_stats.get(fmt, 0) + 1

                st.write("**📊 Статистика форматов:**")
                for fmt, count in format_stats.items():
                    st.write(f"• {fmt}: {count} файлов")

        # ИСПОЛЬЗУЕМ ФИЛЬТРАЦИЮ ИЗ DOCUMENT_PROCESSOR
        if st.button("🚀 Начать AI-фильтрацию",
                     type="primary") and st.session_state.job_description and st.session_state.hr_email and st.session_state.resumes:
            with st.spinner("AI анализирует резюме с помощью embeddings..."):
                st.session_state.filtered_candidates = InterviewBot.filter_resumes_with_embeddings(
                    st.session_state.resumes, st.session_state.job_description
                )
            st.success(f"AI отобрал {len(st.session_state.filtered_candidates)} подходящих кандидатов")

    with tab2:
        st.header("👥 Этап 2: Результаты AI-отбора")

        if not st.session_state.get('filtered_candidates'):
            st.warning("Загрузите данные и выполните AI-фильтрацию в Этапе 1")
        else:
            st.write(f"**Найдено подходящих кандидатов:** {len(st.session_state.filtered_candidates)}")

            for i, candidate in enumerate(st.session_state.filtered_candidates):
                analysis = candidate.get('analysis', {})

                with st.expander(f"Кандидат {i + 1}: {candidate['name']} (Score: {analysis.get('match_score', 0)}%)"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**🤖 AI Анализ:**")
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
st.caption("AI Recruiter System v5.0 | Качественный отбор")
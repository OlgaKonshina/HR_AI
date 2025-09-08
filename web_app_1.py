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
from config import DEEPSEEK_API_KEY, SITE_URL, SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, HR_EMAIL_SIGNATURE
import sys
from pathlib import Path
import json
import re
from pathlib import Path
import pandas as pd
import docx
from striprtf.striprtf import rtf_to_text
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import init_db, get_session, create_interview_in_db, get_interview_from_db, update_interview_report

# Добавляем путь к текущей директории для импорта document_processor
sys.path.append(str(Path(__file__).parent))

# Импортируем из внешнего файла
try:
    from document_processor import DocumentReader, extract_job_title

    try:
        from document_processor import get_embedding

        print("✅ get_embedding импортирован успешно!")
        try:
            test_embedding = get_embedding("тест", "cointegrated/rubert-tiny2")
            print("✅ RuBERT-Tiny модель работает!")
            DOCUMENT_PROCESSOR_AVAILABLE = True
        except Exception as e:
            print(f"⚠️ RuBERT-Tiny не доступна: {e}")
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
# Инициализация базы данных
if 'db_initialized' not in st.session_state:
    st.session_state.db_engine = init_db()
    st.session_state.db_initialized = True
    if st.session_state.db_engine:
        st.session_state.db_session = get_session(st.session_state.db_engine)
    else:
        st.error("❌ Не удалось подключиться к базе данных. Некоторые функции могут быть недоступны.")
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'hr_email' not in st.session_state:
    st.session_state.hr_email = ""
if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'filtered_candidates' not in st.session_state:
    st.session_state.filtered_candidates = []


# Функция для извлечения email из текста
def extract_email_from_text(text):
    """Пытается найти и извлечь email адрес из текста резюме."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    if match:
        return match.group(0)
    return None


# Функция для отправки email через Яндекс
def send_interview_invitation(candidate_email, candidate_name, interview_link, hr_email):
    """Отправляет письмо с приглашением на собеседование кандидату через Яндекс.Почту."""

    subject = f"Приглашение на собеседование в AI Recruiter System"

    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
          .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
          .header {{ background-color: #ffcc00; padding: 20px; text-align: center; color: #000; }}
          .content {{ padding: 20px; background-color: #f9f9f9; }}
          .button {{ display: inline-block; padding: 12px 24px; background-color: #ffcc00; 
                    color: #000; text-decoration: none; border-radius: 5px; font-weight: bold; }}
          .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #666; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h2>🎯 Приглашение на собеседование</h2>
          </div>
          <div class="content">
            <p>Здравствуйте, <strong>{candidate_name}</strong>!</p>
            <p>Благодарим Вас за проявленный интерес к нашей компании и отправленное резюме.</p>
            <p>Мы были впечатлены Вашим опытом и приглашаем Вас на следующий этап отбора — <strong>интервью с нашим AI-ассистентом Львом</strong>.</p>

            <p><strong>📋 Информация о собеседовании:</strong></p>
            <ul>
              <li>🎯 <strong>Формат:</strong> Онлайн-собеседование с AI-ассистентом</li>
              <li>📅 <strong>Срок:</strong> Ссылка действительна 7 дней</li>
              <li>⏰ <strong>Время:</strong> В любое удобное для Вас время</li>
              <li>💻 <strong>Требования:</strong> Компьютер с микрофоном и стабильный интернет</li>
              <li>⏱️ <strong>Длительность:</strong>约30-40 минут</li>
            </ul>

            <p style="text-align: center; margin: 30px 0;">
              <a href="{interview_link}" class="button">🎤 Начать собеседование</a>
            </p>

            <p>Или скопируйте ссылку вручную:<br>
            <code>{interview_link}</code></p>

            <p>Это автоматизированное интервью поможет нам лучше узнать Ваши навыки и опыт.</p>

            <p>Если возникнут технические трудности, пожалуйста, свяжитесь с нами:<br>
            <strong>Email:</strong> <a href="mailto:{hr_email}">{hr_email}</a></p>
          </div>
          <div class="footer">
            <p>Это письмо отправлено автоматически. Пожалуйста, не отвечайте на него.</p>
            <p>{HR_EMAIL_SIGNATURE}</p>
          </div>
        </div>
      </body>
    </html>
    """

    text_body = f"""
    Приглашение на собеседование

    Здравствуйте, {candidate_name}!

    Благодарим Вас за проявленный интерес и отправленное резюме.
    Мы приглашаем Вас на онлайн-собеседование с нашим AI-ассистентом Львом.

    Ссылка для прохождения: {interview_link}
    Ссылка действительна в течение 7 дней.

    Требования: компьютер с микрофоном.

    По вопросам обращайтесь: {hr_email}

    {HR_EMAIL_SIGNATURE}
    """

    msg = MIMEMultipart("alternative")
    msg['Subject'] = subject
    msg['From'] = SMTP_USERNAME
    msg['To'] = candidate_email

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, candidate_email, msg.as_string())
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("Ошибка авторизации: проверьте логин и пароль приложения Яндекс")
        return False
    except Exception as e:
        st.error(f"Ошибка отправки письма: {e}")
        return False


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
                 "content": f'Ты HR-интервьюер. Тебя зовут Лев. Вакансия: {self.job_description}. Резюме: {self.resume}. Задавай вопросы по очереди.Задавай наводящие и уточняющие вопросы'},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def provide_feedback(self, question, answer):
        feedback_prompt = f"""
        Проанализируй ответ кандидата на вопрос собеседования.

        ВАКАНСИЯ: {self.job_description}
        ВОПРОС: {question}
        ОТВЕТ КАНДИДАТА: {answer}

        Дай краткую обратную связь (3-4 предложения):
        - Сильные стороны ответа
        - Что можно улучшить
        - Рекомендации для будущих собеседований
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Ты опытный HR-специалист. Дай конструктивную обратную связь по ответам на собеседовании."},
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
        if not DOCUMENT_PROCESSOR_AVAILABLE:
            st.error("Модуль document_processor не доступен. Используется резервный метод.")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

        filtered = []
        model_path = "cointegrated/rubert-tiny2"

        try:
            job_emb = get_embedding(job_description[:512], model_path)

            for i, resume in enumerate(resumes):
                with st.spinner(f"Анализируем резюме {i + 1}/{len(resumes)} с помощью embeddings..."):
                    try:
                        resume_short = resume['text'][:1000]
                        resume_emb = get_embedding(resume_short, model_path)
                        similarity_tensor = torch.nn.functional.cosine_similarity(resume_emb, job_emb)
                        similarity = similarity_tensor.item() * 100

                        analysis_result = {
                            'match_score': round(similarity, 1),
                            'is_suitable': similarity >= 40,
                            'strengths': [f"Семантическое соответствие: {similarity:.1f}%"],
                            'weaknesses': [],
                            'reason': f"Семантический анализ: {similarity:.1f}%"
                        }

                        if similarity < 40:
                            analysis_result['weaknesses'].append("Низкое семантическое соответствие")

                        resume['analysis'] = analysis_result

                        if analysis_result['is_suitable']:
                            filtered.append(resume)

                    except Exception as e:
                        st.error(f"Ошибка анализа резюме {resume['name']}: {str(e)}")
                        analysis_result = {
                            'match_score': 50,
                            'is_suitable': True,
                            'strengths': ['Автоматическая оценка'],
                            'weaknesses': [],
                            'reason': 'Автоматическая оценка'
                        }
                        resume['analysis'] = analysis_result
                        filtered.append(resume)

            return sorted(filtered, key=lambda x: x['analysis']['match_score'], reverse=True)

        except Exception as e:
            st.error(f"Ошибка при работе с embeddings: {str(e)}")
            return InterviewBot.filter_resumes_fallback(resumes, job_description)

    @staticmethod
    def filter_resumes_fallback(resumes, job_description):
        filtered = []
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
        stop_words = {'опыт', 'работа', 'работы', 'обязанности', 'требования', 'знание', 'навыки'}
        words = re.findall(r'\b[a-zA-Zа-яА-Я]{4,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]

    @staticmethod
    def _analyze_resume_fallback(resume_text, job_description, job_keywords=None):
        if job_keywords is None:
            job_keywords = InterviewBot._extract_keywords(job_description)

        resume_lower = resume_text.lower()
        job_lower = job_description.lower()

        score = 0
        found_keywords = []

        for keyword in job_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', resume_lower):
                score += 3
                found_keywords.append(keyword)

        experience_match = re.search(r'опыт работы.*?(\d+)[^\d]*лет', resume_lower)
        if experience_match:
            years = int(experience_match.group(1))
            score += min(years * 2, 10)

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


# Функции для обработки файлов
def extract_text_from_file(file):
    try:
        if DOCUMENT_PROCESSOR_AVAILABLE:
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                reader = DocumentReader(tmp_path)
                text = reader.extract_text()
                os.unlink(tmp_path)
                return text
            except Exception as e:
                st.warning(f"DocumentReader не смог обработать файл: {e}. Используем резервный метод.")

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
            try:
                return file.read().decode("utf-8", errors='ignore')
            except:
                return f"Формат файла {file.name} не поддерживается"
    except Exception as e:
        return f"Ошибка чтения файла {file.name}: {str(e)}"


def create_interview_link(candidate_data, job_description, hr_email):
    interview_id = str(uuid.uuid4())[:8]
    interview_link = f"{SITE_URL}/?interview_id={interview_id}"

    # Сохраняем в базу данных вместо session_state
    if st.session_state.get('db_session'):
        success = create_interview_in_db(
            st.session_state.db_session,
            interview_id,
            candidate_data,
            job_description,
            hr_email,
            interview_link
        )
        if not success:
            st.error("❌ Не удалось сохранить интервью в базе данных")
            return None

    return interview_link


# Определяем тип пользователя
query_params = st.query_params
is_candidate = 'interview_id' in query_params

if is_candidate:
    interview_id = query_params['interview_id'][0]

    # Получаем данные из базы данных вместо session_state
    interview_data = None
    candidate = None
    job_description = None
    hr_email = None

    if st.session_state.get('db_session'):
        interview_db = get_interview_from_db(st.session_state.db_session, interview_id)
        if interview_db:
            interview_data = {
                'candidate': interview_db.candidate_data,
                'job_description': interview_db.job_description,
                'hr_email': interview_db.hr_email,
                'status': interview_db.status,
                'interview_link': interview_db.interview_link
            }
            candidate = interview_db.candidate_data
            job_description = interview_db.job_description
            hr_email = interview_db.hr_email

    if interview_data and interview_data['status'] != 'expired':
        st.title("🎤 Техническое собеседование")
        st.write("Добро пожаловать на онлайн-собеседование!")
        st.info(f"**Вакансия:** {job_description[:100]}...")
        st.info(f"**Контакт HR:** {hr_email}")

        # Используем ID интервью для уникальности сессии бота
        bot_session_key = f'interview_bot_{interview_id}'
        questions_key = f'questions_{interview_id}'
        answers_key = f'answers_{interview_id}'
        current_question_key = f'current_question_{interview_id}'

        if bot_session_key not in st.session_state:
            st.session_state[bot_session_key] = InterviewBot(
                DEEPSEEK_API_KEY,
                job_description,
                candidate['text']
            )
            st.session_state[current_question_key] = 0
            st.session_state[questions_key] = []
            st.session_state[answers_key] = []

        bot = st.session_state[bot_session_key]
        current_question = st.session_state[current_question_key]
        questions = st.session_state[questions_key]
        answers = st.session_state[answers_key]

        if current_question < 3:
            if current_question >= len(questions):
                previous_answer = answers[-1] if answers else None
                question = bot.generate_question(previous_answer)
                questions.append(question)
                answers.append("")

            st.subheader(f"Вопрос {current_question + 1}/3")
            st.info(questions[current_question])

            if st.button("🎤 Записать ответ", key=f"record_{interview_id}_{current_question}"):
                with st.spinner("Запись... (15 секунд)"):
                    audio_file = load_audio(duration=15)
                    answer = recognize_audio_whisper(audio_file)
                    answers[current_question] = answer
                    feedback = bot.provide_feedback(
                        questions[current_question],
                        answer
                    )
                    bot.feedbacks.append(feedback)
                    st.session_state[current_question_key] += 1
                    st.rerun()

            if answers[current_question]:
                st.write("**Ваш ответ:**")
                st.write(answers[current_question])
        else:
            st.success("✅ Собеседование завершено!")
            st.balloons()
            with st.spinner("Генерируем отчет для HR..."):
                final_report = bot.generate_final_report(hr_email)

                # Сохраняем отчет в базу данных
                if st.session_state.get('db_session'):
                    update_interview_report(st.session_state.db_session, interview_id, final_report)

                st.subheader("📋 Отчет отправлен HR")
                st.write(f"Результаты собеседования отправлены на email: **{hr_email}**")
                st.write("С вами свяжутся в ближайшее время!")

                # Очищаем сессию собеседования
                del st.session_state[bot_session_key]
                del st.session_state[questions_key]
                del st.session_state[answers_key]
                del st.session_state[current_question_key]

    elif interview_data and interview_data['status'] == 'expired':
        st.error("❌ Ссылка на собеседование истекла (действительна 7 дней)")
        st.write("Пожалуйста, свяжитесь с HR для получения новой ссылки")
        st.info(f"Контакт HR: {hr_email}")

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

        # Информация о базе данных
        if st.session_state.get('db_engine'):
            st.success("✅ База данных подключена")
        else:
            st.error("❌ База данных не подключена")

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

        # Кнопка запуска AI-фильтрации
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

                    # Извлечение email и отправка приглашения
                    candidate_text = candidate['text']
                    extracted_email = extract_email_from_text(candidate_text)
                    candidate_name = candidate['name'].split('.')[0]  # Простое извлечение имени

                    if extracted_email:
                        email_to_send = st.text_input("Email кандидата:", value=extracted_email, key=f"email_{i}")
                    else:
                        email_to_send = st.text_input("Email кандидата (не найден в резюме):", key=f"email_{i}")

                    if st.button(f"📧 Отправить приглашение", key=f"invite_{i}"):
                        if not email_to_send:
                            st.error("Поле email не может быть пустым.")
                        else:
                            interview_link = create_interview_link(
                                candidate, st.session_state.job_description, st.session_state.hr_email
                            )

                            if interview_link:  # Проверяем что ссылка создалась успешно
                                with st.spinner("Отправляем приглашение на email..."):
                                    email_sent = send_interview_invitation(
                                        email_to_send, candidate_name, interview_link, st.session_state.hr_email
                                    )

                                if email_sent:
                                    st.success("✅ Приглашение отправлено на email кандидата!")
                                    st.balloons()
                                else:
                                    st.error(
                                        "❌ Не удалось отправить email. Проверьте настройки SMTP или попробуйте позже.")
                                    st.info("**Ссылка для собеседования (скопируйте и отправьте вручную):**")
                                    st.code(interview_link, language=None)
                            else:
                                st.error("❌ Не удалось создать ссылку для собеседования")

                    st.write("**📄 Превью резюме:**")
                    st.text(candidate['text'][:300] + "..." if len(candidate['text']) > 300 else candidate['text'])
st.write("---")
st.caption("AI Recruiter System v5.0 | Mindshift")
import streamlit as st
import torch
import time
import os
from config import DEEPSEEK_API_KEY
from audio_recording import load_audio
from audio_text import recognize_audio_whisper, text_to_ogg
from document_processor import DocumentReader, extract_job_title, get_embedding, _generate_recommendation
from app_new_2 import InterviewBot  # Обновлённый класс

# === Настройки страницы ===
st.set_page_config(page_title="Interview Bot", page_icon="🤖", layout="wide")
st.title("🤖 HR - бот Лев")

# === Загрузка документов ===
st.header("📂 Загрузка документов")
job_file = st.file_uploader("Описание вакансии", type=["pdf", "docx", "rtf", "txt", "csv", "json"])
resume_file = st.file_uploader("Резюме кандидата", type=["pdf", "docx", "rtf", "txt", "csv", "json"])

job_text, resume_text, similarity = None, None, None

if job_file and resume_file:
    job_path = f"uploaded_{job_file.name}"
    resume_path = f"uploaded_{resume_file.name}"
    with open(job_path, "wb") as f:
        f.write(job_file.read())
    with open(resume_path, "wb") as f:
        f.write(resume_file.read())

    job_text = DocumentReader(job_path).extract_text()
    resume_text = DocumentReader(resume_path).extract_text()

    st.subheader("📊 Анализ документов")
    job_title = extract_job_title(job_text)
    st.write(f"**Вакансия:** {job_title}")

    try:
        model_path = "model" if os.path.exists("model/config.json") else "cointegrated/rubert-tiny2"
        job_emb = get_embedding(job_text, model_path)
        resume_emb = get_embedding(resume_text, model_path)
        if job_emb is not None and resume_emb is not None:
            similarity = torch.mm(resume_emb, job_emb.T).item() * 100
            st.write(f"🔗 Схожесть резюме и вакансии: **{similarity:.2f}%**")
            st.info(_generate_recommendation(similarity))
    except Exception as e:
        st.error(f"❌ Ошибка вычисления эмбеддингов: {e}")

# === Автоматический диалог ===
if similarity and similarity >= 85.5:
    st.success("✅ Кандидат подходит! Можно начать собеседование.")
    num_questions = st.slider("Количество вопросов", 3, 30, 5)

    if st.button("🚀 Старт собеседования"):
        bot = InterviewBot(
            api_key=DEEPSEEK_API_KEY,
            job_description=job_text,
            resume=resume_text,
            num_questions=num_questions  # передаём количество вопросов
        )
        st.session_state["bot"] = bot
        st.session_state["num_questions"] = num_questions
        st.session_state["dialog_active"] = True
        st.session_state["current_question"] = 0
        st.session_state["chat_log"] = []
        st.session_state["interview_terminated"] = False
        st.rerun()

if st.session_state.get("dialog_active"):
    bot = st.session_state["bot"]
    current_q = st.session_state["current_question"]

    # === Досрочное завершение ===
    # При нажатии кнопки досрочного завершения
    if st.button("🛑 Закончить собеседование"):
        bot.terminated = True
        last_answer_note = "⚠️ Кандидат досрочно завершил интервью. Он сам закончил собеседование."

        # Генерация итогов
        bot.overall_feedback = bot.generate_overall_feedback(last_answer_note=last_answer_note)
        bot.final_assessment = bot.generate_final_assessment(last_answer_note=last_answer_note)
        bot.save_interview()

        # Вывод на экран
        st.subheader("📊 Итоговая оценка для HR")
        st.write(bot.final_assessment)
        st.sidebar.subheader("📝 Фидбек для кандидата")
        st.sidebar.write(bot.overall_feedback)

        # ⛔ Останавливаем интервью сразу
        st.session_state["dialog_active"] = False
        st.stop()

    # === Генерация вопросов ===
    if current_q < st.session_state["num_questions"]:
        prev_answer = bot.answers[-1] if bot.answers else None
        question = bot.generate_question(prev_answer)
        if question is None:
            st.session_state["current_question"] = st.session_state["num_questions"]
            st.rerun()

        bot.questions.append(question)

        st.subheader(f"Вопрос :")
        st.write(question)

        try:
            text_to_ogg(question)
        except:
            pass

        st.write("🎙️ Говорите (25 секунд)...")
        try:
            audio_file = load_audio(duration=25)
            answer = recognize_audio_whisper(audio_file)
            bot.answers.append(answer)
        except:
            answer = "Не удалось распознать ответ"
            bot.answers.append(answer)

        # Заглушка фидбека по каждому вопросу
        feedback = "Обратная связь будет дана после завершения интервью"
        st.session_state["chat_log"].append({"question": question, "answer": answer, "feedback": feedback})
        st.session_state["current_question"] += 1
        time.sleep(1)
        st.rerun()

    else:  # автоматическое завершение
        bot.terminated = False
        candidate_feedback = bot.generate_overall_feedback()
        final_assessment = bot.generate_final_assessment()
        bot.overall_feedback = candidate_feedback
        bot.final_assessment = final_assessment
        bot.save_interview()

        st.subheader("📊 Итоговая оценка для HR")
        st.write(final_assessment)

        with st.sidebar.expander("📝 Фидбек для кандидата", expanded=True):
            st.write(candidate_feedback)

        st.session_state["dialog_active"] = False

# === Информация о системе ===
st.sidebar.info("""
**ℹ️ Статус системы:**
- Модель: RuBERT-Tiny2
- Аудио: Whisper + Yandex SpeechKit
- AI: DeepSeek API
- Поддержка форматов: PDF, DOCX, RTF, TXT, CSV, JSON
""")

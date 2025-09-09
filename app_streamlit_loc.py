import streamlit as st
import torch
import time
from app import InterviewBot
from config import DEEPSEEK_API_KEY
from audio_recording import load_audio
from audio_text import recognize_audio_whisper, text_to_ogg
from document_processor import DocumentReader, extract_job_title, get_embedding, _generate_recommendation
from app import InterviewBot, print_interview_summary
# === Настройки страницы ===
st.set_page_config(page_title="Interview Bot", page_icon="🤖", layout="wide")
st.title("🤖 Виртуальное собеседование")

# === Загрузка документов ===
st.header("📂 Загрузка документов")
job_file = st.file_uploader("Описание вакансии", type=["pdf", "docx", "rtf", "txt", "csv", "json"])
resume_file = st.file_uploader("Резюме кандидата", type=["pdf", "docx", "rtf", "txt", "csv", "json"])

job_text, resume_text, similarity = None, None, None

if job_file and resume_file:
    job_path = f"uploaded_{job_file.name}"
    resume_path = f"uploaded_{resume_file.name}"
    with open(job_path, "wb") as f: f.write(job_file.read())
    with open(resume_path, "wb") as f: f.write(resume_file.read())

    job_text = DocumentReader(job_path).extract_text()
    resume_text = DocumentReader(resume_path).extract_text()

    st.subheader("📊 Анализ документов")
    job_title = extract_job_title(job_text)
    st.write(f"**Вакансия:** {job_title}")

    try:
        model_path = "model"  # путь к модели эмбеддингов
        job_emb = get_embedding(job_text, model_path)
        resume_emb = get_embedding(resume_text, model_path)
        similarity = torch.mm(resume_emb, job_emb.T).item() * 100
        st.write(f"🔗 Схожесть резюме и вакансии: **{similarity:.2f}%**")
        st.info(_generate_recommendation(similarity))
    except Exception as e:
        st.error(f"Не удалось вычислить эмбеддинги: {e}")

# === Автоматический диалог ===
if similarity and similarity >= 85.5:
    st.success("✅ Кандидат подходит! Можно начать собеседование.")
    num_questions = st.slider("Количество вопросов", 3, 15, 30)

    if st.button("🚀 Старт собеседования"):
        bot = InterviewBot(
            api_key=DEEPSEEK_API_KEY,
            job_description=job_text,
            resume=resume_text
        )
        st.session_state["bot"] = bot
        st.session_state["num_questions"] = num_questions
        st.session_state["dialog_active"] = True
        st.session_state["current_question"] = 0
        st.session_state["chat_log"] = []
        st.rerun()

if st.session_state.get("dialog_active"):
    bot = st.session_state["bot"]
    current_q = st.session_state["current_question"]

    # 🔴 Кнопка завершения собеседования
    if st.button("🛑 Закончить собеседование"):
        st.success("✅ Собеседование завершено кандидатом.")
        final_assessment = bot.generate_final_assessment()
        st.subheader("Итоговая оценка")
        st.write(final_assessment)
        bot.save_interview()

        # 📊 Краткая выжимка
        with st.sidebar.expander("📊 Краткая выжимка собеседования"):
            print_interview_summary(bot)

        st.session_state["dialog_active"] = False
        st.stop()

    if current_q < st.session_state["num_questions"]:
        prev_answer = bot.answers[-1] if bot.answers else None
        question = bot.generate_question(prev_answer)
        bot.questions.append(question)

        st.subheader(f"Вопрос {current_q + 1}:")
        st.write(question)
        text_to_ogg(question)  # озвучиваем вопрос

        # запись ответа сразу
        st.write("🎙️ Говорите (25 секунд)...")
        audio_file = load_audio(duration=25)
        answer = recognize_audio_whisper(audio_file)
        bot.answers.append(answer)

        feedback = bot.provide_feedback(question, answer)
        bot.feedbacks.append(feedback)

        st.session_state["chat_log"].append(
            {"question": question, "answer": answer, "feedback": feedback}
        )
        st.session_state["current_question"] += 1

        # Переход к следующему вопросу
        time.sleep(2)
        st.rerun()
    else:
        st.success("🎯 Собеседование завершено автоматически!")
        final_assessment = bot.generate_final_assessment()
        st.subheader("Итоговая оценка")
        st.write(final_assessment)
        bot.save_interview()

        # 📊 Краткая выжимка
        with st.sidebar.expander("📊 Краткая выжимка собеседования"):
            print_interview_summary(bot)

        st.session_state["dialog_active"] = False

# === История диалога ===
if "chat_log" in st.session_state and st.session_state["chat_log"]:
    st.sidebar.header("История интервью")
    for i, entry in enumerate(st.session_state["chat_log"], 1):
        st.sidebar.write(f"**Вопрос {i}:** {entry['question']}")
        st.sidebar.write(f"💬 {entry['answer']}")
        st.sidebar.write(f"📝 {entry['feedback']}")
        st.sidebar.markdown("---")

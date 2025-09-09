import streamlit as st
import torch
import time
import os
from app import InterviewBot
from config import DEEPSEEK_API_KEY
from audio_recording import load_audio
from audio_text import recognize_audio_whisper, text_to_ogg
from document_processor import DocumentReader, extract_job_title, get_embedding, _generate_recommendation
from app import InterviewBot, print_interview_summary

# === Настройки страницы ===
st.set_page_config(page_title="Interview Bot", page_icon="🤖", layout="wide")
st.title("🤖 HR - бот Лев")

# Предзагрузка модели Whisper при старте
try:
    from audio_text import load_whisper_model
    load_whisper_model()
except Exception as e:
    st.sidebar.warning(f"⚠️ Модель Whisper не загружена: {e}")

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
        # Проверяем, есть ли локальная модель, если нет - используем онлайн
        if os.path.exists("model") and os.path.exists("model/config.json"):
            model_path = "model"
            st.info("🔄 Используется локальная модель RuBERT-Tiny2")
        else:
            model_path = "cointegrated/rubert-tiny2"
            st.info("🌐 Используется онлайн модель RuBERT-Tiny2")

        job_emb = get_embedding(job_text, model_path)
        resume_emb = get_embedding(resume_text, model_path)

        # Проверяем, что эмбеддинги успешно получены
        if job_emb is not None and resume_emb is not None:
            similarity = torch.mm(resume_emb, job_emb.T).item() * 100
            st.write(f"🔗 Схожесть резюме и вакансии: **{similarity:.2f}%**")
            st.info(_generate_recommendation(similarity))
        else:
            st.error("❌ Не удалось получить эмбеддинги документов")

    except Exception as e:
        st.error(f"❌ Не удалось вычислить эмбеддинги: {e}")
        st.info("ℹ️ Попробуйте использовать упрощенный режим или проверьте подключение к интернету")

# === Упрощенный режим если модель не работает ===
if job_file and resume_file and (similarity is None or similarity == 0):
    st.warning("⚠️ Используется упрощенный расчет схожести")


    # Простая текстовая схожесть без эмбеддингов
    def simple_similarity(text1, text2):
        import re
        from collections import Counter

        # Извлекаем слова из текстов
        words1 = re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', text1.lower())
        words2 = re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', text2.lower())

        if not words1 or not words2:
            return 50.0  # Значение по умолчанию

        # Считаем совпадения ключевых слов
        common_words = set(words1) & set(words2)
        similarity_score = len(common_words) / len(set(words1)) *100

        return min(similarity_score, 100)


    similarity = simple_similarity(job_text, resume_text)
    st.write(f"🔗 Схожесть резюме и вакансии: **{similarity:.2f}%**")
    st.info(_generate_recommendation(similarity))

# === Автоматический диалог ===
if similarity and similarity >= 50:
    st.success("✅ Кандидат подходит! Можно начать собеседование.")
    num_questions = st.slider("Количество вопросов", 3, 15, 5)  # Уменьшено для тестирования

    if st.button("🚀 Старт собеседования"):
        try:
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
        except Exception as e:
            st.error(f"❌ Ошибка создания бота: {e}")

if st.session_state.get("dialog_active"):
    bot = st.session_state["bot"]
    current_q = st.session_state["current_question"]

    # 🔴 Кнопка завершения собеседования
    if st.button("🛑 Закончить собеседование"):
        st.success("✅ Собеседование завершено кандидатом.")
        try:
            final_assessment = bot.generate_final_assessment()
            st.subheader("Итоговая оценка")
            st.write(final_assessment)
            bot.save_interview()

            # 📊 Краткая выжимка
            with st.sidebar.expander("📊 Краткая выжимка собеседования"):
                print_interview_summary(bot)
        except Exception as e:
            st.error(f"❌ Ошибка генерации отчета: {e}")

        st.session_state["dialog_active"] = False
        st.stop()

    if current_q < st.session_state["num_questions"]:
        try:
            prev_answer = bot.answers[-1] if bot.answers else None
            question = bot.generate_question(prev_answer)
            bot.questions.append(question)

            st.subheader(f"Вопрос {current_q + 1}:")
            st.write(question)

            # Озвучиваем вопрос с обработкой ошибок
            try:
                text_to_ogg(question)
            except Exception as e:
                st.warning(f"⚠️ Не удалось озвучить вопрос: {e}")

            # Запись ответа
            st.write("🎙️ Говорите (25 секунд)...")
            try:
                audio_file = load_audio(duration=25)
                answer = recognize_audio_whisper(audio_file)
                bot.answers.append(answer)
            except Exception as e:
                st.error(f"❌ Ошибка записи аудио: {e}")
                answer = "Не удалось распознать ответ"
                bot.answers.append(answer)

            # Обратная связь
            try:
                feedback = bot.provide_feedback(question, answer)
                bot.feedbacks.append(feedback)
            except Exception as e:
                st.error(f"❌ Ошибка генерации обратной связи: {e}")
                feedback = "Не удалось сгенерировать обратную связь"
                bot.feedbacks.append(feedback)

            st.session_state["chat_log"].append(
                {"question": question, "answer": answer, "feedback": feedback}
            )
            st.session_state["current_question"] += 1

            # Переход к следующему вопросу
            time.sleep(2)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Ошибка во время собеседования: {e}")
            st.session_state["dialog_active"] = False

    else:
        st.success("🎯 Собеседование завершено автоматически!")
        try:
            final_assessment = bot.generate_final_assessment()
            st.subheader("Итоговая оценка")
            st.write(final_assessment)
            bot.save_interview()

            # 📊 Краткая выжимка
            with st.sidebar.expander("📊 Краткая выжимка собеседования"):
                print_interview_summary(bot)
        except Exception as e:
            st.error(f"❌ Ошибка генерации отчета: {e}")

        st.session_state["dialog_active"] = False

# === История диалога ===
if "chat_log" in st.session_state and st.session_state["chat_log"]:
    st.sidebar.header("История интервью")
    for i, entry in enumerate(st.session_state["chat_log"], 1):
        st.sidebar.write(f"**Вопрос {i}:** {entry['question']}")
        st.sidebar.write(f"💬 Ответ: {entry['answer']}")
        st.sidebar.write(f"📝 Обратная связь: {entry['feedback']}")
        st.sidebar.markdown("---")

# === Информация о статусе ===
st.sidebar.info("""
**ℹ️ Статус системы:**
- Модель: RuBERT-Tiny2
- Аудио: Whisper + Yandex SpeechKit
- AI: DeepSeek API
- Поддержка форматов: PDF, DOCX, RTF, TXT, CSV, JSON
""")

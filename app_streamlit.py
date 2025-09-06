import streamlit as st
import openai
import time
import json
import os
from audio_text import text_to_ogg, recognize_audio_whisper
from audio_recording import load_audio
from config import DEEPSEEK_API_KEY
import tempfile

# Настройка страницы
st.set_page_config(
    page_title="HR Бот - Собеседование",
    page_icon="🎤",
    layout="wide"
)

class InterviewBot:
    def __init__(self, api_key, job_description, resume):
        self.client = openai.OpenAI(api_key=DEEPSEEK_API_KEY,
                                    base_url="https://api.deepseek.com/v1")
        self.job_description = job_description
        self.resume = resume
        self.questions = []
        self.answers = []
        self.feedbacks = []
        self.current_question_number = 0

    def generate_question(self, previous_answer=None):
        """Генерирует следующий вопрос на основе предыдущего ответа"""
        if previous_answer is None:
            prompt = 'Начни собеседование. Задай первый релевантный вопрос кандидату.'
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
        """Дает обратную связь по ответу на вопрос"""
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

    def generate_final_assessment(self):
        """Генерирует итоговую оценку кандидата"""
        assessment_prompt = f"""
        На основе всего собеседования дай итоговую оценку кандидата.

        ВАКАНСИЯ: {self.job_description}
        РЕЗЮМЕ КАНДИДАТА: {self.resume}
        ВОПРОСЫ И ОТВЕТЫ:
        {self._format_qa_for_assessment()}

        Сделай комплексную оценку по следующим критериям:
        1. Соответствие вакансии
        2. Профессиональные компетенции 
        3. Коммуникативные навыки
        4. Сильные стороны
        5. Зоны развития
        6. Рекомендация к найму (да/нет)
        7. Общий балл от 1 до 10
        """

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Ты старший HR-менеджер. Дай комплексную оценку кандидата после собеседования."},
                {"role": "user", "content": assessment_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def _format_qa_for_assessment(self):
        """Форматирует вопросы и ответы для итоговой оценки"""
        formatted = ""
        for i, (question, answer, feedback) in enumerate(zip(self.questions, self.answers, self.feedbacks), 1):
            formatted += f"{i}. В: {question}\n   О: {answer}\n   Ф: {feedback}\n\n"
        return formatted

    def conduct_interview_step(self, num_questions=3):
        """Проводит один шаг собеседования"""
        if len(self.questions) < num_questions:
            # Генерируем вопрос
            previous_answer = self.answers[-1] if self.answers else None
            question = self.generate_question(previous_answer)
            self.questions.append(question)
            
            # Озвучиваем вопрос
            text_to_ogg(question)
            
            return question, False  # False - собеседование не завершено
        
        else:
            # Итоговая оценка
            final_assessment = self.generate_final_assessment()
            self.save_interview()
            return final_assessment, True  # True - собеседование завершено

    def save_interview(self):
        """Сохраняет полные результаты собеседования"""
        results = {
            "job_description": self.job_description,
            "resume": self.resume,
            "questions": self.questions,
            "answers": self.answers,
            "feedbacks": self.feedbacks,
            "final_assessment": self.generate_final_assessment()
        }

        # Сохранение в JSON
        with open("interview_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    st.title("🎤 HR Бот - Проведение собеседований")
    st.write("Интерактивная система для проведения автоматических собеседований")

    # Создаем временные директории если их нет
    os.makedirs("audio/questions", exist_ok=True)
    os.makedirs("audio/answers", exist_ok=True)

    # Загрузка данных
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Описание вакансии")
        job_description = st.text_area(
            "Введите описание вакансии:",
            height=200,
            placeholder="Например: Разработчик Python с опытом работы от 2 лет..."
        )
    
    with col2:
        st.subheader("Резюме кандидата")
        resume = st.text_area(
            "Введите резюме кандидата:",
            height=200,
            placeholder="Опыт работы, навыки, образование..."
        )

    # Настройки собеседования
    st.subheader("Настройки собеседования")
    num_questions = st.slider("Количество вопросов:", 1, 10, 3)
    recording_duration = st.slider("Длительность ответа (секунды):", 5, 60, 25)

    # Инициализация состояния сессии
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_completed' not in st.session_state:
        st.session_state.interview_completed = False

    # Кнопка начала собеседования
    if not st.session_state.interview_started and job_description and resume:
        if st.button("🚀 Начать собеседование", type="primary"):
            st.session_state.bot = InterviewBot(DEEPSEEK_API_KEY, job_description, resume)
            st.session_state.interview_started = True
            st.rerun()

    # Процесс собеседования
    if st.session_state.interview_started and not st.session_state.interview_completed:
        bot = st.session_state.bot
        
        if not st.session_state.current_question:
            # Генерируем первый вопрос
            with st.spinner("Генерируем первый вопрос..."):
                question, completed = bot.conduct_interview_step(num_questions)
                st.session_state.current_question = question
                st.rerun()
        
        else:
            # Показываем текущий вопрос
            st.subheader(f"Вопрос {len(bot.questions)}/{num_questions}")
            st.info(st.session_state.current_question)
            
            # Запись ответа
            st.subheader("Ваш ответ")
            st.write(f"Говорите после нажатия кнопки. Длительность: {recording_duration} секунд")
            
            if st.button("🎤 Начать запись ответа", type="secondary"):
                with st.spinner(f"Записываем ответ... ({recording_duration} сек)"):
                    audio_file = load_audio(duration=recording_duration)
                    
                with st.spinner("Обрабатываем аудио..."):
                    answer = recognize_audio_whisper(audio_file)
                    bot.answers.append(answer)
                    
                    # Показываем распознанный ответ
                    st.text_area("Распознанный ответ:", answer, height=100)
                    
                    # Генерируем обратную связь
                    with st.spinner("Анализируем ответ..."):
                        feedback = bot.provide_feedback(
                            st.session_state.current_question, 
                            answer
                        )
                        bot.feedbacks.append(feedback)
                    
                    st.success("✅ Ответ обработан!")
                    st.subheader("Обратная связь:")
                    st.write(feedback)
                    
                    # Переходим к следующему вопросу или завершаем
                    if len(bot.questions) < num_questions:
                        with st.spinner("Генерируем следующий вопрос..."):
                            next_question, completed = bot.conduct_interview_step(num_questions)
                            st.session_state.current_question = next_question
                    else:
                        st.session_state.interview_completed = True
                    
                    st.rerun()

    # Завершение собеседования
    if st.session_state.interview_completed:
        bot = st.session_state.bot
        
        st.success("🎉 Собеседование завершено!")
        
        # Итоговая оценка
        st.subheader("Итоговая оценка")
        final_assessment = bot.generate_final_assessment()
        st.write(final_assessment)
        
        # Скачивание результатов
        bot.save_interview()
        
        with open("interview_results.json", "rb") as f:
            st.download_button(
                label="📥 Скачать результаты (JSON)",
                data=f,
                file_name="interview_results.json",
                mime="application/json"
            )
        
        # Кнопка нового собеседования
        if st.button("🔄 Начать новое собеседование"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Боковая панель с инструкциями
    with st.sidebar:
        st.header("📋 Инструкция")
        st.write("""
        1. Заполните описание вакансии и резюме
        2. Настройте параметры собеседования
        3. Нажмите 'Начать собеседование'
        4. Слушайте вопрос и отвечайте после сигнала
        5. Получайте обратную связь после каждого ответа
        6. В конце получите итоговую оценку
        """)
        
        st.header("🎧 Требования")
        st.write("""
        - Микрофон для записи ответов
        - Колонки/наушники для прослушивания вопросов
        - Стабильное интернет-соединение
        """)
        
        if st.session_state.interview_started:
            st.header("📊 Прогресс")
            if st.session_state.bot:
                progress = len(st.session_state.bot.questions) / num_questions
                st.progress(progress)
                st.write(f"Вопрос {len(st.session_state.bot.questions)} из {num_questions}")

if __name__ == "__main__":
    main()

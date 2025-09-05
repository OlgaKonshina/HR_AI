import openai
import time
import json
from audio_text import text_to_ogg
from audio_text import recognize_audio_whisper
from config import DEEPSEEK_API_KEY
from audio_recording import load_audio


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
                 "content": f'Ты HR-интервьюер. Тебя зовут Борис. Вакансия: {self.job_description}. Резюме: {self.resume}. Задавай вопросы по очереди.'},
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

    def conduct_interview(self, num_questions=3):
        """Проводит собеседование с обратной связью"""
        print("=== НАЧАЛО СОБЕСЕДОВАНИЯ ===\n")

        for i in range(num_questions):
            self.current_question_number = i + 1

            # Генерируем вопрос
            previous_answer = self.answers[-1] if self.answers else None
            question = self.generate_question(previous_answer)
            self.questions.append(question)

            # Выводим вопрос
            print(f"🔹 Вопрос {self.current_question_number}/{num_questions}:")
            print(f"{question}\n")
            # озвучиваем вопрос
            text_to_ogg(question)
            # Получаем ответ
            answer = recognize_audio_whisper(load_audio())
            self.answers.append(answer)

            # Даем обратную связь
            print("\n⏳ Анализируем ответ...")
            feedback = self.provide_feedback(question, answer)
            self.feedbacks.append(feedback)

            print(f"📝 Обратная связь: {feedback}\n")
            print("-" * 60 + "\n")
            time.sleep(2)  # Пауза между вопросами

        # Итоговая оценка
        print("🎯 Завершаем собеседование...")
        final_assessment = self.generate_final_assessment()

        print("=== ИТОГОВАЯ ОЦЕНКА ===")
        print(final_assessment)

        self.save_interview()

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
        with open("reports/interview_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Сохранение в читаемом текстовом формате
        with open("reports/interview_results.txt", "w", encoding="utf-8") as f:
            f.write("=== ПОЛНЫЕ РЕЗУЛЬТАТЫ СОБЕСЕДОВАНИЯ ===\n\n")
            f.write(f"ВАКАНСИЯ: {self.job_description}\n")
            f.write(f"КАНДИДАТ: {self.resume[:200]}...\n\n")

            f.write("=== ВОПРОСЫ И ОТВЕТЫ ===\n")
            for i, (question, answer, feedback) in enumerate(zip(self.questions, self.answers, self.feedbacks), 1):
                f.write(f"\n🔹 ВОПРОС {i}:\n{question}\n")
                f.write(f"💬 ОТВЕТ:\n{answer}\n")
                f.write(f"📝 ОБРАТНАЯ СВЯЗЬ:\n{feedback}\n")
                f.write("-" * 50 + "\n")

            f.write("\n=== ИТОГОВАЯ ОЦЕНКА ===\n")
            f.write(results["final_assessment"])

        print("\n💾 Результаты сохранены в файлы:")
        print("   - interview_results.json (структурированные данные)")
        print("   - interview_results.txt (читаемый формат)")


# Дополнительные утилиты
def print_interview_summary(bot):
    """Печатает краткую сводку по собеседованию"""
    print("\n" + "=" * 60)
    print("📊 КРАТКАЯ СВОДКА СОБЕСЕДОВАНИЯ")
    print("=" * 60)

    for i, (question, feedback) in enumerate(zip(bot.questions, bot.feedbacks), 1):
        print(f"{i}. {question[:80]}...")
        print(f"   💡 {feedback[:100]}...\n")


# Использование
if __name__ == "__main__":
    # Загрузка данных
    try:
        with open('data/job_decription/job_description', "r", encoding="utf-8") as f:
            job_description = f.read()

        with open('data/resume/resume', "r", encoding="utf-8") as f:
            resume = f.read()
    except FileNotFoundError:
        print("❌ Файлы 'job_description' или 'resume' не найдены!")
        exit()

    # Создание и запуск бота
    bot = InterviewBot(
        api_key=DEEPSEEK_API_KEY,  # API-ключ DeepSeek
        job_description=job_description,
        resume=resume
    )

    try:
        bot.conduct_interview(num_questions=2)

        # Дополнительная сводка
        print_interview_summary(bot)

    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        print("Проверьте API-ключ и подключение к интернету")

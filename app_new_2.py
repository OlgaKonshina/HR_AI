import openai
import time
import json
import os
from audio_text import text_to_ogg
from audio_text import recognize_audio_whisper
from config import DEEPSEEK_API_KEY
from audio_recording import load_audio


class InterviewBot:
    def __init__(self, api_key, job_description, resume, num_questions):
        openai.api_key = api_key
        openai.api_base = "https://api.deepseek.com/v1"

        self.job_description = job_description
        self.resume = resume
        self.questions = []
        self.answers = []
        self.overall_feedback = ""  # фидбек кандидату
        self.final_assessment = ""  # отчёт для HR
        self.current_question_number = 0
        self.num_questions = num_questions

    def generate_question(self, previous_answer=None):
        """Генерирует следующий вопрос на основе предыдущего ответа"""
        if previous_answer is None:
            prompt = 'Начни собеседование. Задай первый релевантный вопрос кандидату.'
        else:
            prompt = f'Ответ кандидата: {previous_answer}. Сформулируй следующий логичный вопрос.'

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": f'Ты HR-интервьюер. Тебя зовут Лев. Вакансия: {self.job_description}. '
                            f'Резюме: {self.resume}. Задавай вопросы по очереди. '
                            f'Всего вопросов: {self.num_questions}.'
                            f'Задавай наводящие и уточняющие вопросы'},
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message.content

    def generate_overall_feedback(self,last_answer_note=""):
        """Генерирует общий краткий фидбек кандидату по всем ответам"""
        feedback_prompt = f"""
        На основе ответов кандидата на собеседовании составь краткий общий фидбек (5-6 предложений). 
        Игнорируй орфографические ошибки. Ответы могут быть краткими из-за ограничения по времени

        ВАКАНСИЯ: {self.job_description}
        РЕЗЮМЕ КАНДИДАТА: {self.resume}
        ВОПРОСЫ И ОТВЕТЫ:
        {self._format_qa_for_assessment()}
        {last_answer_note}
 
        Структура ответа:
        1. Общая оценка выступления
        2. Сильные стороны кандидата
        3. Основные зоны для развития
        4. Краткая рекомендация для будущих собеседований
        """

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Ты HR-специалист. Составь общий краткий фидбек кандидату по результатам интервью."},
                {"role": "user", "content": feedback_prompt},
            ]
        )
        return response.choices[0].message.content

    def generate_final_assessment(self, last_answer_note=""):
        """Генерирует итоговую оценку кандидата для HR"""
        assessment_prompt = f"""
        На основе всего собеседования дай итоговую оценку кандидата. Игнорируй орфографические ошибки. 
        Ответы могут быть краткими из-за ограничения по времени

        ВАКАНСИЯ: {self.job_description}
        РЕЗЮМЕ КАНДИДАТА: {self.resume}
        ВОПРОСЫ И ОТВЕТЫ:
        {self._format_qa_for_assessment()}
        {last_answer_note}

        Сделай комплексную оценку по следующим критериям:
        1. Соответствие вакансии
        2. Профессиональные компетенции 
        3. Коммуникативные навыки
        4. Сильные стороны
        5. Зоны развития
        6. Рекомендация к найму (да/нет)
        7. Общий балл от 1 до 10
        """

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "Ты старший HR-менеджер. Дай комплексную оценку кандидата после собеседования."},
                {"role": "user", "content": assessment_prompt},
            ]
        )
        return response.choices[0].message.content

    def _format_qa_for_assessment(self, last_answer_note=""):
        """Форматирует вопросы и ответы для итоговой оценки"""
        formatted = ""
        for i, (question, answer) in enumerate(zip(self.questions, self.answers), 1):
            formatted += f"{i}. В: {question}\n   О: {answer}\n\n"
        return formatted

    def conduct_interview(self, num_questions=3):
        """Проводит собеседование и формирует 2 отчёта"""
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

            # Озвучиваем вопрос
            try:
                text_to_ogg(question)
            except Exception as e:
                print(f"⚠️ Ошибка озвучивания: {e}")

            # Получаем ответ
            try:
                audio_file = load_audio()
                answer = recognize_audio_whisper(audio_file)
                self.answers.append(answer)
            except Exception as e:
                print(f"⚠️ Ошибка записи аудио: {e}")
                answer = "Не удалось распознать ответ"
                self.answers.append(answer)

            print("-" * 60 + "\n")
            time.sleep(2)

        # Фидбек кандидату
        print("📝 Генерируем общий фидбек для кандидата...")
        try:
            self.overall_feedback = self.generate_overall_feedback()
            print("\n=== ФИДБЕК ДЛЯ КАНДИДАТА ===")
            print(self.overall_feedback)
        except Exception as e:
            print(f"⚠️ Ошибка генерации фидбека: {e}")
            self.overall_feedback = "Не удалось сгенерировать общий фидбек"

        # Оценка для HR
        print("\n🎯 Генерируем итоговую оценку для HR...")
        try:
            self.final_assessment = self.generate_final_assessment()
            print("=== ОЦЕНКА ДЛЯ HR ===")
            print(self.final_assessment)
        except Exception as e:
            print(f"⚠️ Ошибка генерации итоговой оценки: {e}")
            self.final_assessment = "Не удалось сгенерировать итоговую оценку"

        self.save_interview()

    def save_interview(self):
        """Сохраняет результаты собеседования в разные файлы"""
        os.makedirs("reports", exist_ok=True)

        results = {
            "job_description": self.job_description,
            "resume": self.resume,
            "questions": self.questions,
            "answers": self.answers,
            "overall_feedback": self.overall_feedback,
            "final_assessment": self.final_assessment
        }

        try:
            # JSON — общий архив
            with open("reports/interview_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # TXT для кандидата
            with open("reports/candidate_feedback.txt", "w", encoding="utf-8") as f:
                f.write("=== ФИДБЕК ДЛЯ КАНДИДАТА ===\n\n")
                f.write(self.overall_feedback)

            # TXT для HR
            with open("reports/hr_assessment.txt", "w", encoding="utf-8") as f:
                f.write("=== ОЦЕНКА ДЛЯ HR ===\n\n")
                f.write(f"ВАКАНСИЯ: {self.job_description}\n")
                f.write(f"КАНДИДАТ: {self.resume[:200]}...\n\n")
                f.write("=== ВОПРОСЫ И ОТВЕТЫ ===\n")
                for i, (question, answer) in enumerate(zip(self.questions, self.answers), 1):
                    f.write(f"\n🔹 ВОПРОС {i}:\n{question}\n")
                    f.write(f"💬 ОТВЕТ:\n{answer}\n")
                    f.write("-" * 50 + "\n")
                f.write("\n=== ИТОГОВАЯ ОЦЕНКА ===\n")
                f.write(self.final_assessment)

            print("\n💾 Результаты сохранены в файлы:")
            print("   - reports/candidate_feedback.txt (фидбек для кандидата)")
            print("   - reports/hr_assessment.txt (оценка для HR)")
            print("   - reports/interview_results.json (архив)")
        except Exception as e:
            print(f"❌ Ошибка сохранения результатов: {e}")

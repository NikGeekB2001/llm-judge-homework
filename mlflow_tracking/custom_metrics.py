import re
from typing import Dict, Any, Union
from mlflow.metrics import make_genai_metric
# Импортируем вашу функцию вызова LLM из клиента
# Убедитесь, что путь импорта верен относительно вашей структуры
from llm_client.openai_client_example import call_openai_compatible_api
import pandas as pd
import os

# Загружаем промпт для судьи из файла
# Убедитесь, что путь к judge_prompt.txt верен относительно места запуска скрипта (обычно корень проекта)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
JUDGE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "judge_prompt.txt")

with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
    JUDGE_PROMPT_TEMPLATE = f.read()

def evaluate_with_llm_judge(
    context: str,    # Это будет `inputs` из вашего датафрейма
    prediction: str, # Это будет `predictions` из вашего датафрейма
    parameters: Dict[str, Any] # Здесь можно передать имя модели LLM-судьи
) -> Dict[str, Union[float, str]]:
    """
    Функция, которая вызывает LLM-судью для оценки ответа и парсит ее вывод.
    """
    llm_judge_model = parameters.get("llm_judge_model", "facebook/opt-1.3b") # Модель для судьи
    
    # Формируем промпт для судьи
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=context, answer=prediction)
    
    # Вызываем вашу локальную LLM-судью
    judge_response = call_openai_compatible_api(judge_prompt, llm_judge_model)

    score = 0.0
    justification = "Failed to get justification from judge."

    if judge_response:
        # Парсинг ответа LLM-судьи
        # Ищем "Оценка: [число]"
        score_match = re.search(r"Оценка:\s*(\d+(\.\d+)?)", judge_response, re.IGNORECASE)
        # Ищем "Обоснование: [текст до конца или до следующей секции]"
        justification_match = re.search(r"Обоснование:\s*(.*)", judge_response, re.IGNORECASE | re.DOTALL)
        
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                print(f"Warning: Could not convert score '{score_match.group(1)}' to float.")
        else:
            print(f"Warning: Could not find score in judge response: {judge_response}")

        if justification_match:
            justification = justification_match.group(1).strip()
        else:
            print(f"Warning: Could not find justification in judge response: {judge_response}")
    else:
        justification = "No response from LLM judge."

    # Возвращаем словарь с метрикой и артефактами (обоснованием)
    return {"score": score, "justification": justification}


# Создаем кастомную метрику MLflow
# Укажите здесь ту же модель, которую вы запустили как LLM-судью через vLLM
# Она будет использоваться при вызове evaluate_with_llm_judge
llm_judge_metric = make_genai_metric(
    name="llm_judge_score", # Имя метрики, которое будет отображаться в MLflow UI
    metric_function=evaluate_with_llm_judge,
    higher_is_better=True, # Чем выше балл, тем лучше
    parameters={"llm_judge_model": "facebook/opt-1.3b"} # Передаем имя модели-судьи
)

import mlflow
import pandas as pd
import os
from dotenv import load_dotenv

# Загружаем переменные окружения (например, HUGGING_FACE_HUB_TOKEN)
load_dotenv()

# Импортируем нашу кастомную метрику
from mlflow_tracking.custom_metrics import llm_judge_metric

# Указываем MLflow Tracking URI, если MLflow Tracking Server запущен отдельно
# Если вы запускаете 'mlflow ui' локально и не меняли порт, то по умолчанию будет работать
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

if __name__ == "__main__":
    print("Starting MLflow LLM-as-a-Judge experiment...")

    # Путь к файлу с данными для оценки
    # Убедитесь, что этот путь верен относительно места запуска скрипта (обычно корень проекта)
    EVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_data.csv")

    # Загружаем данные для оценки
    eval_df = pd.read_csv(EVAL_DATA_PATH)

    # MLflow evaluate ожидает, что входные данные для метрики будут в колонках
    # 'inputs' (для context) и 'predictions' (для prediction).
    # Убедитесь, что ваш CSV файл имеет эти заголовки.

    # Запускаем эксперимент MLflow
    # Примечание: eval_df должен содержать колонки 'inputs' и 'predictions'.
    # `predictions` здесь - это ответы, которые мы хотим оценить.
    # `targets` не нужен для LLM-as-a-Judge, так как нет золотого стандарта.
    with mlflow.start_run(run_name="LLM_Judge_Evaluation_Run"):
        mlflow.evaluate(
            data=eval_df,
            targets=None, # Отсутствует, так как LLM-судья сама генерирует оценку
            predictions="predictions", # Колонка с ответами, которые будет оценивать LLM-судья
            input_cols=["inputs"], # Колонка с вопросами, передаваемыми в LLM-судью как context
            metrics=[llm_judge_metric], # Список ваших кастомных метрик
            model_type="llm/v1", # Указывает MLflow, что это LLM-оценка
            extra_tags={"experiment_type": "llm_judge_evaluation", "judge_model": llm_judge_metric.parameters.get("llm_judge_model")}
        )
    
    print("MLflow LLM-as-a-Judge experiment finished. Check MLflow UI at http://localhost:5000")

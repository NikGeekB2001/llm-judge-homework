from openai import OpenAI
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

def call_openai_compatible_api(prompt: str, model_name: str, base_url: str = "http://localhost:8000/v1") -> str | None:
    """
    Вызывает локальную модель vLLM через OpenAI-совместимый клиент.
    """
    # vLLM не требует настоящего API ключа, но клиент OpenAI его требует.
    # Поэтому используем "EMPTY" или то, что есть в переменной окружения.
    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("HUGGING_FACE_HUB_TOKEN", "EMPTY") # Можно использовать HF токен, если он нужен для модели, или просто "EMPTY"
    )
    try:
        # Для моделей с поддержкой chat-шаблонов (Meta-Llama-3-Instruct, Zephyr)
        # используйте формат messages.
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200 # Ограничиваем длину ответа
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI-compatible API with OpenAI client: {e}")
        return None

if __name__ == "__main__":
    # Укажите имя модели, которую вы запустили в run_vllm_server.sh
    model_to_use = "facebook/opt-1.3b" # Пример

    question = "What is the capital of Japan?"
    print(f"Question: {question}")
    answer = call_openai_compatible_api(question, model_to_use)
    if answer:
        print(f"Answer (OpenAI client): {answer}")
    else:
        print("Failed to get answer via OpenAI client.")

    print("\n--- Testing another question ---")
    question_2 = "Why is the sky blue?"
    print(f"Question: {question_2}")
    answer_2 = call_openai_compatible_api(question_2, model_to_use)
    if answer_2:
        print(f"Answer (OpenAI client): {answer_2}")
    else:
        print("Failed to get answer via OpenAI client.")

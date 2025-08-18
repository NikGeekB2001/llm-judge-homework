import requests
import json

def call_vllm_api(prompt: str, model_name: str, base_url: str = "http://localhost:8000") -> str | None:
    """
    Вызывает локальную модель vLLM через HTTP requests.
    Использует правильный эндпоинт /generate вместо /v1/chat/completions
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    }
    try:
        response = requests.post(f"{base_url}/generate", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()["text"][0]
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM API: {e}")
        return None

if __name__ == "__main__":
    model_to_use = "facebook/opt-1.3b"

    question = "What is the capital of Germany? Provide a concise answer."
    print(f"Question: {question}")
    answer = call_vllm_api(question, model_to_use)
    if answer:
        print(f"Answer: {answer}")
    else:
        print("Failed to get answer.")

    print("\n--- Testing another question ---")
    question_2 = """Instruction: Explain the concept of 'attention' in Transformers in one sentence.
Response:"""
    print(f"Question: {question_2}")
    answer_2 = call_vllm_api(question_2, model_to_use)
    if answer_2:
        print(f"Answer: {answer_2}")
    else:
        print("Failed to get answer.")

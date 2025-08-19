# Трекинг LLM-инференса с vLLM (LLM-as-a-Judge)

## Цель задания

В данном домашнем задании освоен полный цикл использования LLM в роли оценщика (LLM-as-a-Judge):
*   Развёрнута собственная модель через vLLM.
*   Настроена кастомная метрика оценки качества ответов с помощью MLflow.
*   Осуществлены вызовы модели через OpenAI-совместимый API.
*   Сформированы запросы на оценку и проанализировано поведение модели в роли "судьи".

## Структура проекта

##  Структура проекта

```
llm-judge-homework/
├── .vscode/                 # Конфигурация VS Code
├── .env                     # Переменные окружения (токен Hugging Face)
├── .gitignore              # Файлы для игнорирования Git
├── README.md               # Документация проекта
├── requirements.txt        # Зависимости Python
├── vllm_server/
│   └── run_vllm_server.sh  # Скрипт для запуска vLLM сервера
├── llm_client/
│   ├── __init__.py
│   ├── vllm_api_client.py  # Клиент для HTTP запросов
│   └── openai_client_example.py  # Клиент для OpenAI API
├── mlflow_tracking/
│   ├── __init__.py
│   ├── custom_metrics.py   # Кастомная метрика LLM-as-a-Judge
│   ├── run_experiment.py   # Скрипт для запуска MLflow эксперимента
│   ├── data/
│   │   └── evaluation_data.csv  # Данные для оценки
│   └── prompts/
│       └── judge_prompt.txt     # Промпт для LLM-судьи
└── screenshots/            # Папка для скриншотов
```

##  Пошаговая инструкция по запуску
## Запуск проекта

### 1. Предварительная настройка

1.  **Клонируйте репозиторий (если вы его загрузили с GitHub):**
    ```bash
    cd llm-judge-homework
    ```
2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python3 -m venv .venv
    # Для Linux/macOS:
    source .venv/bin/activate
    # Для Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1
    ```
3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Настройте Hugging Face токен (если используете модели, требующие авторизации, например Llama 3, Zephyr):**
    *   Создайте файл `.env` в корне проекта (если его нет) и добавьте ваш токен:
        ```
        HUGGING_FACE_HUB_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN_HERE
        ```
    *   Выполните вход через Hugging Face CLI: `huggingface-cli login`.

### 2. Запуск vLLM сервера

1.  **Перейдите в директорию `vllm_server`:**
    ```bash
    cd vllm_server
    ```
2.  **Отредактируйте `run_vllm_server.sh`**, указав имя выбранной вами модели (например, `facebook/opt-1.3b` или `meta-llama/Meta-Llama-3-8B-Instruct`).
3.  **Запустите сервер:**
    ```bash
    bash run_vllm_server.sh
    ```
    Убедитесь, что сервер успешно загрузил модель и начал прослушивать порт `8000`.
    **Скриншот:** Сделайте скриншот работающего терминала с сервером vLLM и сохраните его как `screenshots/vllm_running.png`.
4.  **Оставьте сервер запущенным** для следующих шагов.
bash run_vllm_server.sh

перезапуск если потребуется 
cd llm-judge-homework/vllm_server

Проверьте работу сервера:
curl http://localhost:8000/v1/models



cd ../mlflow_tracking

### 3. Взаимодействие с локальной моделью

Откройте новый терминал и перейдите обратно в корневую директорию проекта.

1.  **Используя библиотеку `requests` (или `httpx`):**
    *   Отредактируйте `llm_client/vllm_api_client.py`, указав имя вашей модели.
    *   Запустите скрипт:
        ```bash
        python llm_client/vllm_api_client.py
        ```
    *   **Проверка:** Убедитесь, что вы получаете корректный ответ от модели.
    *   **Скриншот:** Сделайте скриншот вывода скрипта и сохраните его как `screenshots/vllm_response_requests.png`.

2.  **Используя библиотеку `openai`:**
    *   Отредактируйте `llm_client/openai_client_example.py`, указав имя вашей модели.
    *   Запустите скрипт:
        ```bash
        python llm_client/openai_client_example.py
        ```
    *   **Проверка:** Убедитесь, что вы получаете корректный ответ. Обратите внимание на настройки `base_url` и `api_key` для клиента `openai`.
    *   **Скриншот:** Сделайте скриншот вывода скрипта и сохраните его как `screenshots/vllm_response_openai.png`.

### 4. MLflow и интеграция кастомной genai метрики (LLM-as-a-Judge)

1.  **Подготовьте данные для оценки:**
    *   Откройте `mlflow_tracking/data/evaluation_data.csv`.
    *   Заполните его тестовыми вопросами в колонке `inputs` и *ответами, которые ваша LLM-судья должна будет оценить*, в колонке `predictions`.
        ```csv
        inputs,predictions
        What is the capital of France?,Paris is the capital of France.
        Who is the current president of the USA?,Joe Biden is the current president of the United States.
        Tell me about quantum physics.,Quantum physics is a branch of science that describes the behavior of matter and energy at the atomic and subatomic level.
        What is the primary function of a CPU?,"A CPU is the central processing unit, responsible for executing instructions and performing calculations in a computer."
        ```
2.  **Разработайте промпт для LLM-судьи:**
    *   Откройте `mlflow_tracking/prompts/judge_prompt.txt`.
    *   Сформулируйте четкий и подробный промпт, который будет инструктировать LLM оценивать ответы. Включите критерии оценки (точность, полнота, релевантность, лаконичность) и **обязательно** укажите ожидаемый формат вывода (например, "Оценка: [число от 1 до 5]\nОбоснование: [текст]").
    *   **Рекомендация:** Добавьте несколько примеров (few-shot) в промпт, если ваша модель-судья не является очень мощной, чтобы улучшить качество оценки.
    Пример содержимого `judge_prompt.txt`:
    ```
    Вы являетесь экспертным оценщиком качества ответов. Ваша задача - оценить предоставленный ответ на вопрос по шкале от 1 до 5, где 1 - очень плохо, 5 - отлично. Обоснуйте свою оценку, ссылаясь на указанные критерии.

    **Критерии оценки:**
    - **Точность (Accuracy):** Ответ должен быть фактически верным.
    - **Полнота (Completeness):** Ответ должен быть достаточно полным, чтобы полностью ответить на вопрос, но без излишней информации.
    - **Релевантность (Relevance):** Ответ должен быть прямо относиться к заданному вопросу.
    - **Лаконичность (Conciseness):** Ответ должен быть кратким и по существу, без "воды".

    **Формат вывода:**
    Оценка: [ЧИСЛО ОТ 1 ДО 5]
    Обоснование: [ВАШЕ ОБОСНОВАНИЕ, ссылающееся на критерии]

    ---
    Вопрос: {question}
    Ответ для оценки: {answer}
    ---
    ```
3.  **Реализуйте кастомную метрику:**
    *   Отредактируйте `mlflow_tracking/custom_metrics.py`.
    *   Убедитесь, что функция `evaluate_with_llm_judge` правильно вызывает вашу локальную LLM-судью и парсит её ответ для получения числовой оценки и обоснования.
    *   Укажите имя вашей модели-судьи в `parameters` для `make_genai_metric`.
4.  **Запустите MLflow эксперимент:**
    *   Отредактируйте `mlflow_tracking/run_experiment.py`, убедившись, что он правильно загружает данные и вызывает `mlflow.evaluate()`.
    *   Запустите скрипт:
        ```bash
        python mlflow_tracking/run_experiment.py
        ```

### 5. Проверка результатов в MLflow UI

1.  **Запустите MLflow UI:**
    *   В корневой директории проекта в терминале выполните:
        ```bash
        mlflow ui
        ```
2.  **Откройте MLflow UI в браузере:** Перейдите по адресу `http://localhost:5000` (или любой другой порт, который указал `mlflow ui`).
3.  **Проверка:**
    *   Найдите свой эксперимент (по тегу `experiment_type: llm_judge_evaluation` или по имени).
    *   Убедитесь, что метрика `llm_judge_accuracy` отображается.
    *   Перейдите на страницу деталей Run'а (нажав на дату/время) и проверьте, что там отображаются числовая оценка и обоснование от LLM-судьи для каждого примера.
    *   **Скриншот:** Сделайте скриншот вкладки "Experiments" (`screenshots/mlflow_experiments_tab.png`).
    *   **Скриншот:** Сделайте скриншот страницы с деталями вашего успешного запуска (`screenshots/mlflow_run_details.png`), где видны метрики и параметры, а также логи LLM-судьи.

## 📊 Анализ и интерпретация результатов

Добавьте этот раздел в `README.md` после выполнения всех шагов.

1.  **Как работает метрика LLM-as-a-Judge:**
    *   Опишите, как ваша кастомная метрика вызывает LLM-судью, как формируется промпт, и как парсится ответ для получения оценки и обоснования.
2.  **Поведение модели-судьи:**
    *   Приведите примеры оценок и обоснований, которые дала ваша LLM-судья.
    *   Проанализируйте, в каких случаях оценки были адекватными, а в каких — возможно, ошибочными или необоснованными. Какие критерии модель учитывала лучше всего?
    *   Были ли неожиданные результаты? Почему, как вы думаете?
3.  **Преимущества и недостатки LLM-as-a-Judge:**
    *   **Преимущества:** Скорость, масштабируемость (можно оценивать тысячи ответов), автоматизация, гибкость (легко менять критерии в промпте).
    *   **Недостатки:** Зависимость от качества LLM-судьи (может быть предвзятой, "галлюцинировать" оценки), стоимость инференса LLM-судьи, потенциальные сложности с парсингом, отсутствие абсолютной надежности.
4.  **Ограничения вашей реализации:**
    *   Какие упрощения были сделаны? Что можно улучшить? (Например, более сложный парсинг ответа судьи, использование более мощной LLM-судьи, добавление more few-shot примеров, обработка ошибок).

---

### **`.vscode/settings.json`**
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": [
        "--line-length",
        "100"
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.terminal.activateEnvironment": true
}

--------------
✅ Текущий статус
- **vLLM-сервер** запущен и доступен по адресу `http://localhost:8000`  
- **Модель** `facebook/opt-1.3b` загружена и готова к работе  
- **API-эндпоинт** `/generate` функционирует корректно  

✅ Как пользоваться

1. Проверить работоспособность сервера  
   ```bash
   curl http://localhost:8000/health
   ```

2. Протестировать генерацию текста  
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Столица Германии?", "max_tokens": 50}'
   ```

3. Запустить клиентский скрипт  
   ```bash
   cd llm-judge-homework/llm_client
   python3 vllm_api_client.py
   ```

4. Запустить эксперимент в MLflow  
   ```bash
   cd llm-judge-homework/mlflow_tracking
   python3 run_experiment.py
   ```

Проект полностью готов к решению задач LLM-as-a-Judge!

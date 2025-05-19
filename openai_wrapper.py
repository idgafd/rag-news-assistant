import json
from datetime import date
from openai import OpenAI
from typing import Optional, Any, Union, List
from config import OPENAI_API_KEY


api_key = OPENAI_API_KEY
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in the .env file")
client = OpenAI(api_key=api_key)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


ADDITIONAL_CONTEXT = f"TODAY is {date.today().isoformat()}. Use this as the current date for interpreting relative time expressions.\n\n"
ROUTER_PROMPT = ADDITIONAL_CONTEXT+load_prompt("prompts/router_prompt.txt")
ANSWER_PROMPT = load_prompt("prompts/answer_prompt.txt")
FALLBACK_PROMPT = load_prompt("prompts/fallback_answer_prompt.txt")


def route_user_query(user_message: str, image_path: Optional[str] = None,
                     model: str = "gpt-4-turbo") -> Union[str, None, dict[Any, Any]]:
    full_user_input = user_message.strip()
    if image_path:
        full_user_input += f"\n[Uploaded Image: {image_path}]"

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": full_user_input}
            ]
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        return parsed
    except Exception as e:
        print(f"Routing Error: failed to parse OpenAI response: {e}")
        return {}


def answer_user_query(user_query: str, articles: List, model: str = "gpt-4-turbo") -> str:
    formatted_articles = []

    for i, article in enumerate(articles, start=1):
        payload = article.payload
        title = payload.get("title", "Untitled")
        date = payload.get("date", "Unknown date")
        caption = payload.get("image_caption", None)
        text = payload.get("text", "").strip()

        block = f"{i}. Title: {title}\nDate: {date}"
        if caption and caption.lower() != "none":
            block += f"\nImage Description: {caption}"
        block += f"\nText:\n{text}"
        formatted_articles.append(block)

    articles_block = "\n\n".join(formatted_articles)

    full_prompt = ANSWER_PROMPT.replace("<insert user query here>", user_query)
    prompt_input = f"{full_prompt}\n\nArticles:\n{articles_block}\n\nAnswer:"

    response = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_input}
        ]
    )

    return response.choices[0].message.content.strip()


def fallback_answer_user_query(user_query: str) -> str:
    prompt = FALLBACK_PROMPT.replace("<insert question here>", user_query)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


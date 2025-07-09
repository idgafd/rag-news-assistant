import json
from datetime import date
from typing import Optional, Union, Any, List

from openai import OpenAI
from clients.config import OPENAI_API_KEY


class OpenAIRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in the .env file")
        self.client = OpenAI(api_key=self.api_key)

        self.router_prompt = self._load_prompt("prompts/router_prompt.txt")
        self.answer_prompt = self._load_prompt("prompts/answer_prompt.txt")
        self.fallback_prompt = self._load_prompt("prompts/fallback_answer_prompt.txt")
        self.query_prompt = self._load_prompt("prompts/generate_queries.txt")
        self.additional_context = f"TODAY is {date.today().isoformat()}. Use this as the current date for interpreting relative time expressions.\n\n"

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def route_user_query(self, user_message: str, image_path: Optional[str] = None,
                         model: str = "gpt-4-turbo") -> Union[dict[Any, Any], None]:
        full_user_input = user_message.strip()
        if image_path:
            full_user_input += f"\n[Uploaded Image: {image_path}]"

        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": self.additional_context + self.router_prompt},
                    {"role": "user", "content": full_user_input}
                ]
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Routing Error: failed to parse OpenAI response: {e}")
            return {}

    def answer_user_query(self, user_query: str, articles: List[dict], model: str = "gpt-4-turbo") -> str:
        formatted_articles = []

        for i, article in enumerate(articles, start=1):
            payload = article.get("payload", {})
            title = payload.get("title", "Untitled")
            pub_date = payload.get("date", "Unknown date")
            caption = payload.get("image_caption", None)
            text = payload.get("text", "").strip()

            block = f"{i}. Title: {title}\nDate: {pub_date}"
            if caption and caption.lower() != "none":
                block += f"\nImage Description: {caption}"
            block += f"\nText:\n{text}"
            formatted_articles.append(block)

        articles_block = "\n\n".join(formatted_articles)
        full_prompt = self.answer_prompt.replace("<insert user query here>", user_query)
        prompt_input = f"{full_prompt}\n\nArticles:\n{articles_block}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=model,
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_input}
            ]
        )
        return response.choices[0].message.content.strip()

    def fallback_answer_user_query(self, user_query: str, model: str = "gpt-4-turbo") -> str:
        prompt = self.fallback_prompt.replace("<insert question here>", user_query)

        response = self.client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def generate_queries(self, text: str, model: str = "gpt-4o") -> str:
        prompt = self.query_prompt.replace("<insert article here>", text)
        response = self.client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

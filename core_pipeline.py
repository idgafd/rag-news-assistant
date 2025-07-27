from utils import search_similar_articles, get_articles_by_date_range, find_by_uploaded_image

from typing import List, Dict


def execute_routed_call(call: Dict) -> List[Dict]:
    if not call or not isinstance(call, dict):
        print("Empty or invalid routing result. Nothing to execute.")
        return []

    func = call.get("function")

    prompt = None if call.get("prompt") == "None" else call.get("prompt")
    image_path = None if call.get("image_path_or_url") == "None" else call.get("image_path_or_url")
    date_from = None if call.get("date_from") == "None" else call.get("date_from")
    date_to = None if call.get("date_to") == "None" else call.get("date_to")

    if func == "search_similar_articles":
        return search_similar_articles(prompt=prompt, date_from=date_from, date_to=date_to)

    elif func == "get_articles_by_date_range":
        return get_articles_by_date_range(date_from=date_from, date_to=date_to)

    elif func == "find_by_uploaded_image":
        return find_by_uploaded_image(image_path_or_url=image_path, prompt=prompt)

    else:
        print(f"Unknown function: {func}")
        return []


def format_resources_block(articles: List) -> str:
    if not articles:
        return ""

    lines = ["\n\n**Resources:**"]
    for i, article in enumerate(articles, start=1):
        payload = article.payload
        title = payload.get("title", "Untitled")
        url = payload.get("url", "No URL")
        image = payload.get("image_url", None)

        lines.append(f"{i}. **{title}**")
        lines.append(f"   [Read article]({url})")
        if image and image.lower() != "none":
            lines.append(f"   ![Image]({image})")

    return "\n".join(lines)


# Example of use
# user_query = "Did anything important happen in generative AI in March?"
# image = None
#
# call = route_user_query(user_query, image_path=image)
# articles = execute_routed_call(call)
# if articles:
#     answer = answer_user_query(user_query, articles)
#     resources = format_resources_block(articles)
#     final_message = f"{answer}\n{resources}"
# else:
#     final_message = fallback_answer_user_query(user_query)
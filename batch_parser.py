import uuid
import requests
import logging
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup

from clients.qdrant_client_wrapper import QdrantVectorStoreClient
from utils import generate_combined_embedding_with_metadata

logger = logging.getLogger(__name__)


def format_date(date_str):
    return datetime.strptime(date_str, "%b %d, %Y").strftime("%Y-%m-%d")


# Main page parsing
def parse_the_batch_pages(start_url="https://www.deeplearning.ai/the-batch/", max_pages=10,
                          date_from="2025-05-01", date_to="2025-05-15"):
    headers = {"User-Agent": "Mozilla/5.0"}
    all_articles = []

    date_from_dt = datetime.strptime(date_from, "%Y-%m-%d")
    date_to_dt = datetime.strptime(date_to, "%Y-%m-%d")

    for page in range(1, max_pages + 1):
        url = start_url if page == 1 else f"{start_url.rstrip('/')}/page/{page}/"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.find_all("article", class_="bg-white relative rounded-lg shadow-sm hover:shadow-md transition-shadow flex flex-col h-full")
        if not articles:
            break

        for article in articles:
            title_tag = article.find("h2")
            title = title_tag.get_text(strip=True) if title_tag else None

            summary_tag = article.find("div", class_="text-sm")
            summary = summary_tag.get_text(strip=True) if summary_tag else None

            date_tag = article.find("a")
            date_raw = date_tag.get_text(strip=True) if date_tag else None

            try:
                date_dt = datetime.strptime(date_raw, "%b %d, %Y")
            except:
                continue

            if not (date_from_dt <= date_dt <= date_to_dt):
                continue

            image_tag = article.find("img")
            image_url = None
            if image_tag:
                src = image_tag.get("src")
                if "url=" in src:
                    parsed_url = urlparse(src)
                    image_url = parse_qs(parsed_url.query).get("url", [None])[0]
                else:
                    image_url = src

            link_tags = article.find_all("a")
            url = None
            if len(link_tags) > 1:
                url = "https://www.deeplearning.ai" + link_tags[1]["href"]

            all_articles.append({
                "title": title,
                "summary": summary,
                "date": date_raw,
                "image": image_url,
                "url": url
            })

    return all_articles


# Article content extraction
def parse_article_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    intro_blocks = []
    content_blocks = []
    found_news = False
    stop_collecting = False

    skip_ad_block = False
    image_count_after_ad = 0

    for elem in soup.find_all(["h1", "h2", "p", "figure"]):
        if elem.name in ["h1", "h2"] and elem.get("id") == "news":
            found_news = True
            continue

        if not found_news:
            if elem.name == "p":
                text = elem.get_text(strip=True)
                if text:
                    intro_blocks.append({"type": "paragraph", "text": text})

            elif elem.name in ["h1", "h2"]:
                heading = elem.get_text(strip=True)
                if heading:
                    intro_blocks.append({"type": "heading", "text": heading})

            elif elem.name == "figure":
                img_tag = elem.find("img")
                if img_tag and img_tag.get("src"):
                    intro_blocks.append({"type": "image", "url": img_tag["src"]})

        else:
            if stop_collecting:
                break

            if skip_ad_block:
                if elem.name == "figure":
                    img_tag = elem.find("img")
                    if img_tag and img_tag.get("src"):
                        image_count_after_ad += 1
                        if image_count_after_ad >= 2:
                            skip_ad_block = False
                    continue
                else:
                    continue

            if elem.name in ["h1", "h2"]:
                heading = elem.get_text(strip=True)
                if heading.lower().strip() == "subscribe to the batch" or heading.lower().strip() == "data points":
                    stop_collecting = True
                    continue

                if "deeplearning.ai" in heading.lower():
                    skip_ad_block = True
                    image_count_after_ad = 0
                    continue

                content_blocks.append({"type": "heading", "text": heading})

            elif elem.name == "p":
                text = elem.get_text(strip=True)
                if text:
                    content_blocks.append({"type": "paragraph", "text": text})

            elif elem.name == "figure":
                img_tag = elem.find("img")
                if img_tag and img_tag.get("src"):
                    content_blocks.append({"type": "image", "url": img_tag["src"]})

    return {
        "intro": intro_blocks,
        "content": content_blocks
    }


# Content structuring
def extract_letter_from_intro(intro_blocks):
    start_idx = next((i for i, b in enumerate(intro_blocks)
                      if b["type"] == "paragraph" and "dear friends" in b["text"].lower()), None)

    end_idx = next((i for i, b in enumerate(intro_blocks)
                    if b["type"] == "paragraph" and "andrew" in b["text"].lower()), None)

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return None

    letter_blocks = intro_blocks[start_idx + 1:end_idx + 1]

    image_block = next((b for b in letter_blocks if b["type"] == "image"), None)
    image_url = image_block["url"] if image_block else None

    letter_text = "\n".join(b["text"].replace("\xa0", " ") for b in letter_blocks if b["type"] == "paragraph").strip()

    return {
        "text": letter_text,
        "image": image_url
    }


def split_into_subarticles(blocks):
    subarticles = []
    current = None
    pending_image = None

    for block in blocks:
        if block["type"] == "image":
            pending_image = block
            continue

        if block["type"] == "heading":
            if current:
                subarticles.append(current)
            current = {
                "title": block["text"],
                "blocks": []
            }
            if pending_image:
                current["blocks"].append(pending_image)
                pending_image = None

        elif current:
            current["blocks"].append(block)

    if current:
        subarticles.append(current)

    return subarticles


def process_article(article, article_content):
    subarticles = []
    blocks = split_into_subarticles(article_content)

    for sub in blocks:
        text = "\n".join(b['text'] for b in sub['blocks'] if b['type'] == 'paragraph')
        image_block = next((b for b in sub['blocks'] if b['type'] == 'image'), None)

        subarticles.append({
            "title": sub['title'],
            "type": "News",
            "text": text,
            "image": image_block['url'] if image_block else None,
            "date": format_date(article['date']),
            "url": article['url']
        })

    return subarticles


def convert_to_qdrant_points(all_subarticles):
    points = []

    for i, entry in enumerate(all_subarticles):
        if "embedding" not in entry or entry["embedding"] is None:
            continue

        try:
            date_int = int(entry["date"].replace("-", ""))
        except:
            date_int = None

        point = {
            "id": str(uuid.uuid4()),
            "vector": entry["embedding"],
            "payload": {
                "title": entry.get("title"),
                "type": entry.get("type"),
                "date": entry.get("date"),
                "date_int": date_int,
                "url": entry.get("url"),
                "text": entry.get("text"),
                "image_url": entry.get("image"),
                "image_caption": entry.get("image_caption"),
                "image_embedding": entry.get("image_embedding"),
                "combined_text": entry.get("combined_text"),
            }
        }
        points.append(point)

    return points


def run_pipeline_for_range(date_from: str, date_to: str, max_pages: int = 10):
    """
    Run full pipeline from scraping, processing, embedding and uploading articles to Qdrant
    for a given date range.
    """
    qdrant_client = QdrantVectorStoreClient()

    logger.info(f"Deleting points in Qdrant for range {date_from} to {date_to}")
    qdrant_client.delete_all_points(filter={"must": [{"key": "date_int",
                                        "range": {"gte": int(date_from.replace("-", "")),
                                                  "lte": int(date_to.replace("-", ""))}}]})

    articles = parse_the_batch_pages(max_pages=max_pages, date_from=date_from, date_to=date_to)
    logger.info(f"Found {len(articles)} articles for processing.")

    all_subarticles = []

    for article in articles:
        logger.debug(f"Parsing content from {article['url']}")
        content = parse_article_content(article['url'])
        if not content:
            continue

        letter = extract_letter_from_intro(content["intro"])
        if letter:
            all_subarticles.append({
                "title": article["title"],
                "type": "Letter",
                "text": letter["text"],
                "image": letter.get("image"),
                "date": format_date(article["date"]),
                "url": article["url"]
            })

        subarticles = process_article(article, content["content"])
        all_subarticles.extend(subarticles)

    logger.info(f"Total subarticles to embed: {len(all_subarticles)}")

    for entry in all_subarticles:
        enriched = generate_combined_embedding_with_metadata(entry["text"], entry.get("image"))
        entry.update(enriched)

    points = convert_to_qdrant_points(all_subarticles)
    logger.info(f"Uploading {len(points)} points to Qdrant")
    qdrant_client.upload_points(points)
    logger.info("Pipeline completed.")

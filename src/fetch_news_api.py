import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

script_dir = Path(__file__).parent
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

def fetch_real_news(sources="bbc-news,reuters,associated-press,cnn", page_size=100):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "sources": sources,
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("articles", [])


if __name__ == "__main__":
    articles = fetch_real_news()

    rows = []
    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or "" 
        
        rows.append({
            "title": title,
            "text": f"{title} {description}".strip(),
            "source": article.get("source", {}).get("name"),
            "publishedAt": article.get("publishedAt"),
            "label": 1 
        })

    df = pd.DataFrame(rows)
    output_path = script_dir.parent / "data" / "raw" / "news_api_real.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")

"""
Module for fetching and processing Wikipedia articles.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import requests
from bs4 import BeautifulSoup

from ..config import CACHE_DIR, CACHE_TTL_DAYS, FALLBACK_ARTICLES


def fetch_top_wikipedia_articles(
    n: int = 100,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Optional[str] = None,
    model_name: str = "default",
) -> List[str]:
    """
    Fetch the top N most viewed Wikipedia articles.

    Args:
        n: Number of top articles to fetch
        use_cache: Whether to use cached list if available
        force_refresh: Whether to force a refresh even if cache is valid
        cache_dir: Directory to store cached data
        model_name: Name of the model (used for cache file naming)

    Returns:
        List of article titles
    """
    print(
        f"[WIKI-LIST] Fetching top {n} Wikipedia articles (cache={use_cache}, force_refresh={force_refresh})"
    )
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Create a model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    os.makedirs(model_cache_dir, exist_ok=True)

    cache_file = os.path.join(model_cache_dir, "top_articles.json")
    print(f"[WIKI-LIST] Cache file: {cache_file}")
    print(f"[WIKI-LIST] Cache exists: {os.path.exists(cache_file)}")

    # Check if we should use cached data
    if use_cache and not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Check if cache is still valid
            cache_timestamp = datetime.fromisoformat(cached_data.get("timestamp", "2000-01-01"))
            cache_age = (datetime.now() - cache_timestamp).days
            print(f"[WIKI-LIST] Cache timestamp: {cache_timestamp}, age: {cache_age} days")

            if cache_age <= CACHE_TTL_DAYS["articles"]:
                articles = cached_data.get("articles", [])[:n]
                print(
                    f"[WIKI-LIST] HIT: Using cached list of {len(articles)} top articles from {cache_timestamp}"
                )
                if articles:
                    print(f"[WIKI-LIST] Sample articles: {articles[:3]}")
                return cast(List[str], articles)
            else:
                print(
                    f"[WIKI-LIST] EXPIRED: Cached article list is {cache_age} days old (> {CACHE_TTL_DAYS['articles']}), fetching new data"
                )
        except Exception as e:
            print(f"[WIKI-LIST] ERROR: Error reading cached article list: {e}")

    # Get the most viewed articles from the past month
    # Use current date instead of hardcoded 2023/10
    current_date = datetime.now()
    prev_month = current_date - timedelta(days=30)
    year = prev_month.strftime("%Y")
    month = prev_month.strftime("%m")

    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/all-days"
    print(f"[WIKI-LIST] API URL: {url}")

    try:
        print("[WIKI-LIST] MISS: Requesting data from Wikimedia API")
        response = requests.get(url)
        status_code = response.status_code
        print(f"[WIKI-LIST] API response status: {status_code}")

        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract article titles
        articles = []
        print("[WIKI-LIST] Processing API response to extract article titles")

        if "items" not in data or not data["items"]:
            print(f"[WIKI-LIST] ERROR: API response missing 'items' field: {data.keys()}")
            raise ValueError("Invalid API response - missing 'items' field")

        for item in data["items"][0]["articles"]:
            if "Main_Page" not in item["article"] and "Special:" not in item["article"]:
                articles.append(item["article"])
                if len(articles) >= n:
                    break

        print(f"[WIKI-LIST] Extracted {len(articles)} article titles")
        if articles:
            print(f"[WIKI-LIST] Sample articles: {articles[:3]}")

        # Cache the fetched articles
        cache_data = {"timestamp": datetime.now().isoformat(), "articles": articles}
        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            print(f"[WIKI-LIST] CACHED: Successfully cached list of {len(articles)} top articles")
        except Exception as e:
            print(f"[WIKI-LIST] ERROR: Error caching article list: {e}")

        return cast(List[str], articles)

    except requests.exceptions.HTTPError as http_err:
        print(f"[WIKI-LIST] ERROR: HTTP error occurred: {http_err}")
        print(f"[WIKI-LIST] ERROR: Status code: {response.status_code}")
        print(
            f"[WIKI-LIST] ERROR: Response text: {response.text[:500]}..."
        )  # Print first 500 chars of response
    except requests.exceptions.JSONDecodeError as json_err:
        print(f"[WIKI-LIST] ERROR: JSON decode error: {json_err}")
        print(
            f"[WIKI-LIST] ERROR: Response text: {response.text[:500]}..."
        )  # Print first 500 chars of response
    except Exception as err:
        print(f"[WIKI-LIST] ERROR: Other error occurred: {err}")

    # Try to load expired cache as a fallback if it exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            articles = cached_data.get("articles", [])[:n]
            if articles:
                print(
                    f"[WIKI-LIST] FALLBACK: Using expired cached list of {len(articles)} articles"
                )
                if articles:
                    print(f"[WIKI-LIST] FALLBACK sample articles: {articles[:3]}")
                return cast(List[str], articles)
        except Exception as e:
            print(f"[WIKI-LIST] ERROR: Error reading cached article list as fallback: {e}")

    # Fallback to a list of common Wikipedia articles if the API fails
    print("[WIKI-LIST] FALLBACK: Using hardcoded list of Wikipedia articles...")
    fallback_articles = FALLBACK_ARTICLES[:n]
    print(f"[WIKI-LIST] FALLBACK: Using {len(fallback_articles)} fallback articles")
    if fallback_articles:
        print(f"[WIKI-LIST] FALLBACK sample articles: {fallback_articles[:3]}")
    return cast(List[str], fallback_articles)


def fetch_article_content(
    article_title: str,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    model_name: str = "default",
) -> str:
    """
    Fetch the content of a Wikipedia article, using cache if available.

    Args:
        article_title: Title of the Wikipedia article
        use_cache: Whether to use cached content
        cache_dir: Directory to store cached data
        model_name: Name of the model (used for cache folder)

    Returns:
        Article content as a string
    """
    print(f"[WIKI] Processing article: '{article_title}'")

    # Create cache directory
    wiki_cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
    os.makedirs(wiki_cache_dir, exist_ok=True)

    cache_file = os.path.join(
        wiki_cache_dir, f"article_{article_title.replace(' ', '_').replace('/', '_')}.json"
    )

    print(f"[WIKI] Cache file: {cache_file}")
    print(f"[WIKI] Cache exists: {os.path.exists(cache_file)}")

    # Check cache first if enabled
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            cache_timestamp = cached_data.get("timestamp")
            if cache_timestamp:
                # Calculate age of cache in days
                cache_date = datetime.fromisoformat(cache_timestamp)
                cache_age = (datetime.now() - cache_date).days

                if cache_age <= CACHE_TTL_DAYS["content"]:
                    # Cache is still valid
                    content_sample = (
                        cached_data.get("content", "")[:100] + "..."
                        if cached_data.get("content")
                        else ""
                    )
                    print(
                        f"[WIKI] HIT: Using cached content for '{article_title}' ({cache_age} days old)"
                    )
                    print(f"[WIKI] Content sample: {content_sample}")
                    return cached_data.get("content", "")
                else:
                    print(
                        f"[WIKI] EXPIRED: Cached content for '{article_title}' is {cache_age} days old (> {CACHE_TTL_DAYS['content']})"
                    )
        except Exception as e:
            print(f"[WIKI] ERROR: Error reading cache for '{article_title}': {e}")

    # Cache miss or expired, fetch from Wikipedia
    print(f"[WIKI] MISS: Fetching content for '{article_title}' from Wikipedia")
    try:
        # Make the article title URL-safe
        url_title = article_title.replace(" ", "_")

        # Wikipedia API URL for content extraction
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles={url_title}"
        print(f"[WIKI] API URL: {url}")

        # Add user agent to avoid 403 errors
        headers = {
            "User-Agent": "DefaultModeNetworkExperiment/1.0 (Research project; contact@example.com)"
        }

        # Fetch with retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                print(f"[WIKI] API request attempt {attempt+1}/{max_retries}")
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                print(f"[WIKI] API request successful: status {response.status_code}")
                break
            except requests.exceptions.RequestException as e:
                print(f"[WIKI] API request failed: {e}")
                if attempt < max_retries - 1:
                    print(
                        f"[WIKI] Retry {attempt+1}/{max_retries} for '{article_title}' after error: {e}"
                    )
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise

        data = response.json()

        # Extract the page content
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            print(f"[WIKI] ERROR: No pages found for '{article_title}'")
            raise ValueError(f"No pages found for '{article_title}'")

        # Get the first page (there should only be one)
        page_id = next(iter(pages))
        print(f"[WIKI] Page ID: {page_id}")

        if page_id == "-1":
            print(f"[WIKI] ERROR: Article '{article_title}' not found")
            raise ValueError(f"Article '{article_title}' not found")

        content = pages[page_id].get("extract", "")
        content_length = len(content)
        content_sample = content[:100] + "..." if content else ""
        print(f"[WIKI] Content retrieved: {content_length} characters")
        print(f"[WIKI] Content sample: {content_sample}")

        # Cache the content if we got something
        if content and use_cache:
            try:
                cache_data = {
                    "title": article_title,
                    "timestamp": datetime.now().isoformat(),
                    "content": content,
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                print(
                    f"[WIKI] CACHED: Successfully saved content for '{article_title}' ({content_length} chars)"
                )
            except Exception as e:
                print(f"[WIKI] ERROR: Error caching content for '{article_title}': {e}")

        return content

    except Exception as e:
        print(f"[WIKI] ERROR: Error fetching content for '{article_title}': {e}")

        # Try to use expired cache as fallback
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                print(f"[WIKI] FALLBACK: Using expired cache for '{article_title}'")
                content = cached_data.get("content", "")
                content_sample = content[:100] + "..." if content else ""
                print(f"[WIKI] FALLBACK content sample: {content_sample}")
                return content
            except Exception as cache_e:
                print(f"[WIKI] ERROR: Error reading expired cache: {cache_e}")

        print(
            f"[WIKI] ERROR: Could not retrieve content for '{article_title}' - returning empty string"
        )
        return ""

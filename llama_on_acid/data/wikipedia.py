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
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Create a model-specific cache directory
    model_cache_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
    os.makedirs(model_cache_dir, exist_ok=True)

    cache_file = os.path.join(model_cache_dir, "top_articles.json")

    # Check if we should use cached data
    if use_cache and not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Check if cache is still valid
            cache_timestamp = datetime.fromisoformat(cached_data.get("timestamp", "2000-01-01"))
            cache_age = (datetime.now() - cache_timestamp).days

            if cache_age <= CACHE_TTL_DAYS["articles"]:
                articles = cached_data.get("articles", [])[:n]
                print(f"Using cached list of {len(articles)} top articles from {cache_timestamp}")
                return cast(List[str], articles)
            else:
                print(
                    f"Cached article list is {cache_age} days old (> {CACHE_TTL_DAYS['articles']}), fetching new data"
                )
        except Exception as e:
            print(f"Error reading cached article list: {e}")

    # Get the most viewed articles from the past month
    # Use current date instead of hardcoded 2023/10
    current_date = datetime.now()
    prev_month = current_date - timedelta(days=30)
    year = prev_month.strftime("%Y")
    month = prev_month.strftime("%m")

    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{year}/{month}/all-days"
    print(f"Requesting data from URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract article titles
        articles = []
        for item in data["items"][0]["articles"]:
            if "Main_Page" not in item["article"] and "Special:" not in item["article"]:
                articles.append(item["article"])
                if len(articles) >= n:
                    break

        # Cache the fetched articles
        cache_data = {"timestamp": datetime.now().isoformat(), "articles": articles}
        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            print(f"Cached list of {len(articles)} top articles")
        except Exception as e:
            print(f"Error caching article list: {e}")

        print(f"Fetched {len(articles)} articles")
        return cast(List[str], articles)

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
    except requests.exceptions.JSONDecodeError as json_err:
        print(f"JSON decode error: {json_err}")
        print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
    except Exception as err:
        print(f"Other error occurred: {err}")

    # Try to load expired cache as a fallback if it exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            articles = cached_data.get("articles", [])[:n]
            if articles:
                print(f"Using expired cached list of {len(articles)} articles as fallback")
                return cast(List[str], articles)
        except Exception as e:
            print(f"Error reading cached article list as fallback: {e}")

    # Fallback to a list of common Wikipedia articles if the API fails
    print("Using fallback list of Wikipedia articles...")
    fallback_articles = FALLBACK_ARTICLES[:n]
    print(f"Using {len(fallback_articles)} fallback articles")
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
    # Create cache directory
    wiki_cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
    os.makedirs(wiki_cache_dir, exist_ok=True)

    cache_file = os.path.join(
        wiki_cache_dir, f"article_{article_title.replace(' ', '_').replace('/', '_')}.json"
    )

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
                    print(f"Using cached content for '{article_title}' ({cache_age} days old)")
                    return cached_data.get("content", "")
                else:
                    print(
                        f"Cached content for '{article_title}' is {cache_age} days old (> {CACHE_TTL_DAYS['content']})"
                    )
        except Exception as e:
            print(f"Error reading cache for '{article_title}': {e}")

    # Cache miss or expired, fetch from Wikipedia
    try:
        # Make the article title URL-safe
        url_title = article_title.replace(" ", "_")

        # Wikipedia API URL for content extraction
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles={url_title}"

        # Add user agent to avoid 403 errors
        headers = {
            "User-Agent": "DefaultModeNetworkExperiment/1.0 (Research project; contact@example.com)"
        }

        # Fetch with retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt+1}/{max_retries} for '{article_title}' after error: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise

        data = response.json()

        # Extract the page content
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise ValueError(f"No pages found for '{article_title}'")

        # Get the first page (there should only be one)
        page_id = next(iter(pages))
        if page_id == "-1":
            raise ValueError(f"Article '{article_title}' not found")

        content = pages[page_id].get("extract", "")

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
            except Exception as e:
                print(f"Error caching content for '{article_title}': {e}")

        return content

    except Exception as e:
        print(f"Error fetching content for '{article_title}': {e}")

        # Try to use expired cache as fallback
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                print(f"Using expired cache as fallback for '{article_title}'")
                return cached_data.get("content", "")
            except Exception as cache_e:
                print(f"Error reading expired cache: {cache_e}")

        # If all else fails, return empty string
        return ""

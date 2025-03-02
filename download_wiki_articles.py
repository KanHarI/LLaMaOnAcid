#!/usr/bin/env python
"""
Script to download and cache the top 100 Wikipedia articles.
This pre-populates the cache so that future runs don't need to fetch articles online.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any
import json

# Ensure llama_on_acid package is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_on_acid.data.wikipedia import fetch_top_wikipedia_articles, fetch_article_content
from llama_on_acid.config import CACHE_DIR, DEFAULT_MODEL_NAME


def download_and_cache_wiki_articles(
    n_articles: int = 100,
    model_name: str = DEFAULT_MODEL_NAME,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Download and cache the top N Wikipedia articles and their content.
    
    Args:
        n_articles: Number of articles to download
        model_name: Model name for cache directory organization
        force_refresh: Whether to force refresh existing cached data
        
    Returns:
        Dictionary with statistics about the download/cache process
    """
    start_time = time.time()
    print(f"====== Downloading and caching top {n_articles} Wikipedia articles ======")
    print(f"Model name: {model_name}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Force refresh: {force_refresh}")
    
    # Create the cache directory for this model
    model_cache_dir = os.path.join(CACHE_DIR, model_name.replace("/", "_"))
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Stats for reporting
    stats = {
        "start_time": datetime.now().isoformat(),
        "articles_requested": n_articles,
        "articles_fetched": 0,
        "articles_with_content": 0,
        "total_content_size": 0,
        "success_rate": 0.0,
        "elapsed_time": 0,
        "articles": []
    }
    
    # Step 1: Get the list of top articles
    print("\n1. Fetching list of top Wikipedia articles...")
    article_titles = fetch_top_wikipedia_articles(
        n=n_articles,
        use_cache=not force_refresh,
        force_refresh=force_refresh,
        model_name=model_name
    )
    
    stats["articles_fetched"] = len(article_titles)
    print(f"Retrieved {len(article_titles)} article titles")
    
    # Step 2: Download and cache content for each article
    print("\n2. Downloading and caching article content...")
    
    for i, article_title in enumerate(article_titles):
        try:
            print(f"\n[{i+1}/{len(article_titles)}] Processing: {article_title}")
            content = fetch_article_content(
                article_title=article_title,
                use_cache=not force_refresh,
                cache_dir=CACHE_DIR,
                model_name=model_name
            )
            
            content_size = len(content)
            if content_size > 0:
                stats["articles_with_content"] += 1
                stats["total_content_size"] += content_size
                stats["articles"].append({
                    "title": article_title,
                    "size": content_size,
                    "status": "success"
                })
                print(f"✅ Successfully cached: {article_title} ({content_size} chars)")
            else:
                stats["articles"].append({
                    "title": article_title,
                    "size": 0,
                    "status": "empty_content"
                })
                print(f"⚠️ Warning: No content retrieved for {article_title}")
                
        except Exception as e:
            print(f"❌ Error processing {article_title}: {e}")
            stats["articles"].append({
                "title": article_title,
                "size": 0,
                "status": f"error: {str(e)}"
            })
    
    # Calculate stats
    elapsed_time = time.time() - start_time
    stats["elapsed_time"] = elapsed_time
    stats["success_rate"] = stats["articles_with_content"] / max(1, stats["articles_fetched"]) * 100
    
    # Save stats to file
    stats_file = os.path.join(model_cache_dir, "wiki_download_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n====== Download Summary ======")
    print(f"Articles requested: {stats['articles_requested']}")
    print(f"Articles fetched: {stats['articles_fetched']}")
    print(f"Articles with content: {stats['articles_with_content']}")
    print(f"Total content size: {stats['total_content_size'] / 1024:.2f} KB")
    print(f"Success rate: {stats['success_rate']:.2f}%")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Stats saved to: {stats_file}")
    print("\nCache is now ready for use in experiments!")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and cache Wikipedia articles")
    parser.add_argument(
        "--count", 
        type=int, 
        default=100, 
        help="Number of articles to download (default: 100)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_NAME, 
        help=f"Model name for cache organization (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force refresh existing cached data"
    )
    
    args = parser.parse_args()
    
    download_and_cache_wiki_articles(
        n_articles=args.count,
        model_name=args.model,
        force_refresh=args.force
    ) 
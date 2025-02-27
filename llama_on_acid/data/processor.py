"""
Module for processing and preparing text chunks for analysis.
"""

import os
import pickle
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..config import CACHE_DIR, MIN_CHUNK_TOKENS
from .wikipedia import fetch_article_content


def prepare_text_chunks(
    articles: List[str],
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 512,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    model_name: str = "default",
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
) -> List[str]:
    """
    Prepare chunks of Wikipedia articles for processing.

    Args:
        articles: List of Wikipedia article titles
        tokenizer: Tokenizer to use for processing text
        chunk_size: Size of each chunk in tokens
        use_cache: Whether to use cached chunks if available
        cache_dir: Directory to store cached data
        model_name: Name of the model (used for cache folder)
        min_chunk_tokens: Minimum number of tokens for a valid chunk

    Returns:
        List of text chunks
    """
    print(f"[CHUNKS] Preparing text chunks from {len(articles)} Wikipedia articles (chunk_size={chunk_size}, min_tokens={min_chunk_tokens})...")

    # Create cache directory
    wiki_cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
    os.makedirs(wiki_cache_dir, exist_ok=True)

    # Check for cached chunks
    chunks_cache_file = os.path.join(wiki_cache_dir, f"chunks_{chunk_size}.pkl")
    print(f"[CHUNKS] Cache file: {chunks_cache_file}")
    print(f"[CHUNKS] Cache exists: {os.path.exists(chunks_cache_file)}")

    if use_cache and os.path.exists(chunks_cache_file):
        try:
            print(f"[CHUNKS] Loading cached chunks...")
            with open(chunks_cache_file, "rb") as f:
                cached_data = pickle.load(f)

            # Verify the cache contains what we expect
            if "chunks" in cached_data and "articles" in cached_data:
                cached_articles = set(cached_data["articles"])
                current_articles = set(articles)
                
                print(f"[CHUNKS] Cached articles: {len(cached_articles)}, Current articles: {len(current_articles)}")
                print(f"[CHUNKS] Articles overlap: {len(cached_articles.intersection(current_articles))}")

                # If the cached chunks were generated from the same or a superset of current articles
                if current_articles.issubset(cached_articles):
                    chunks = cached_data["chunks"]
                    print(f"[CHUNKS] HIT: Using {len(chunks)} cached chunks from {len(cached_articles)} articles")
                    if chunks:
                        print(f"[CHUNKS] Sample chunk: {chunks[0][:100]}...")
                    return chunks
                else:
                    # If we have new articles, but can reuse some cached content
                    print(
                        f"[CHUNKS] PARTIAL: Articles list changed. Cached: {len(cached_articles)}, Current: {len(current_articles)}"
                    )
                    print(
                        f"[CHUNKS] Will fetch content for {len(current_articles - cached_articles)} new articles"
                    )
        except Exception as e:
            print(f"[CHUNKS] ERROR: Error reading cached chunks: {e}")

    print(f"[CHUNKS] MISS: Creating new chunks from {len(articles)} articles")
    chunks = []
    processed_articles = []  # Keep track of successfully processed articles

    for i, article_title in enumerate(tqdm(articles)):
        try:
            print(f"[CHUNKS] Processing article {i+1}/{len(articles)}: '{article_title}'")
            
            # Use our cache-aware fetch_article_content method
            content = fetch_article_content(
                article_title, use_cache=use_cache, cache_dir=cache_dir, model_name=model_name
            )

            # Skip if content is empty
            if not content or len(content.strip()) < 100:  # Skip very short content
                print(f"[CHUNKS] SKIP: Article '{article_title}' has insufficient content ({len(content) if content else 0} chars)")
                continue

            # Tokenize the content
            tokens = tokenizer.encode(content)
            print(f"[CHUNKS] Article '{article_title}' tokenized: {len(tokens)} tokens")

            # Split into chunks
            article_has_valid_chunks = False
            article_chunks = 0
            
            for j in range(0, len(tokens), chunk_size):
                if i + chunk_size < len(tokens):
                    chunk_tokens = tokens[j : j + chunk_size]

                    # Only add chunks with sufficient content
                    if len(chunk_tokens) >= min_chunk_tokens:
                        chunk_text = tokenizer.decode(chunk_tokens)
                        chunks.append(chunk_text)
                        article_chunks += 1
                        article_has_valid_chunks = True

            print(f"[CHUNKS] Created {article_chunks} chunks from article '{article_title}'")
            
            # Only record this article as processed if it contributed valid chunks
            if article_has_valid_chunks:
                processed_articles.append(article_title)
                print(f"[CHUNKS] Added article '{article_title}' to processed list")
            else:
                print(f"[CHUNKS] SKIP: No valid chunks created from article '{article_title}'")

            # Sleep to avoid rate limiting
            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"[CHUNKS] ERROR: Error processing article '{article_title}': {e}")
            continue

    # Cache the chunks if we have any
    if chunks:
        try:
            print(f"[CHUNKS] Caching {len(chunks)} chunks from {len(processed_articles)} articles...")
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "chunk_size": chunk_size,
                "articles": processed_articles,
                "chunks": chunks,
            }
            with open(chunks_cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"[CHUNKS] CACHED: Successfully cached {len(chunks)} processed chunks from {len(processed_articles)} articles")
        except Exception as e:
            print(f"[CHUNKS] ERROR: Error caching processed chunks: {e}")

    # Ensure we have at least some chunks
    if not chunks:
        print("[CHUNKS] WARNING: No valid chunks were created. Using fallback text.")
        # Create some simple chunks from hardcoded text
        fallback_text = "The default mode network (DMN) is a large-scale brain network primarily composed of the medial prefrontal cortex, posterior cingulate cortex, and angular gyrus. It is most commonly shown to be active when a person is not focused on the outside world and the brain is at wakeful rest, such as during daydreaming and mind-wandering. It can also be active during detailed thoughts about the past or future, and in social contexts."
        tokens = tokenizer.encode(fallback_text)
        chunks = [tokenizer.decode(tokens)]
        print(f"[CHUNKS] FALLBACK: Created 1 chunk with {len(tokens)} tokens")

    # Shuffle the chunks
    random.shuffle(chunks)
    print(f"[CHUNKS] DONE: Prepared and shuffled {len(chunks)} text chunks")
    
    if chunks:
        print(f"[CHUNKS] Sample chunk: {chunks[0][:100]}...")
    
    return chunks

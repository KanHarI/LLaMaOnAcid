"""
Module for processing and preparing text chunks for analysis.
"""
import os
import pickle
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
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
    min_chunk_tokens: int = MIN_CHUNK_TOKENS
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
    print("Preparing text chunks from Wikipedia articles...")
    
    # Create cache directory
    wiki_cache_dir = os.path.join(cache_dir or CACHE_DIR, model_name.replace("/", "_"))
    os.makedirs(wiki_cache_dir, exist_ok=True)
    
    # Check for cached chunks
    chunks_cache_file = os.path.join(wiki_cache_dir, f"chunks_{chunk_size}.pkl")
    
    if use_cache and os.path.exists(chunks_cache_file):
        try:
            with open(chunks_cache_file, "rb") as f:
                cached_data = pickle.load(f)
            
            # Verify the cache contains what we expect
            if "chunks" in cached_data and "articles" in cached_data:
                cached_articles = set(cached_data["articles"])
                current_articles = set(articles)
                
                # If the cached chunks were generated from the same or a superset of current articles
                if current_articles.issubset(cached_articles):
                    chunks = cached_data["chunks"]
                    print(f"Using {len(chunks)} cached chunks from {len(cached_articles)} articles")
                    return chunks
                else:
                    # If we have new articles, but can reuse some cached content
                    print(f"Articles list changed. Cached: {len(cached_articles)}, Current: {len(current_articles)}")
                    print(f"Will fetch content for {len(current_articles - cached_articles)} new articles")
        except Exception as e:
            print(f"Error reading cached chunks: {e}")
    
    chunks = []
    processed_articles = []  # Keep track of successfully processed articles
    
    for article_title in tqdm(articles):
        try:
            # Use our cache-aware fetch_article_content method
            content = fetch_article_content(
                article_title, 
                use_cache=use_cache,
                cache_dir=cache_dir,
                model_name=model_name
            )
            
            # Skip if content is empty
            if not content or len(content.strip()) < 100:  # Skip very short content
                print(f"Skipping article '{article_title}' due to insufficient content")
                continue
            
            # Tokenize the content
            tokens = tokenizer.encode(content)
            
            # Split into chunks
            article_has_valid_chunks = False
            for i in range(0, len(tokens), chunk_size):
                if i + chunk_size < len(tokens):
                    chunk_tokens = tokens[i:i+chunk_size]
                    
                    # Only add chunks with sufficient content
                    if len(chunk_tokens) >= min_chunk_tokens:
                        chunk_text = tokenizer.decode(chunk_tokens)
                        chunks.append(chunk_text)
                        article_has_valid_chunks = True
            
            # Only record this article as processed if it contributed valid chunks
            if article_has_valid_chunks:
                processed_articles.append(article_title)
                    
            # Sleep to avoid rate limiting
            import time
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing article {article_title}: {e}")
            continue
    
    # Cache the chunks if we have any
    if chunks:
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "chunk_size": chunk_size,
                "articles": processed_articles,
                "chunks": chunks
            }
            with open(chunks_cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Cached {len(chunks)} processed chunks from {len(processed_articles)} articles")
        except Exception as e:
            print(f"Error caching processed chunks: {e}")
    
    # Ensure we have at least some chunks
    if not chunks:
        print("Warning: No valid chunks were created. Using fallback text.")
        # Create some simple chunks from hardcoded text
        fallback_text = "The default mode network (DMN) is a large-scale brain network primarily composed of the medial prefrontal cortex, posterior cingulate cortex, and angular gyrus. It is most commonly shown to be active when a person is not focused on the outside world and the brain is at wakeful rest, such as during daydreaming and mind-wandering. It can also be active during detailed thoughts about the past or future, and in social contexts."
        tokens = tokenizer.encode(fallback_text)
        chunks = [tokenizer.decode(tokens)]
    
    # Shuffle the chunks
    random.shuffle(chunks)
    
    print(f"Prepared {len(chunks)} text chunks")
    return chunks 
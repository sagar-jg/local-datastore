#!/usr/bin/env python3
"""
Command-line interface for the context-aware web scraper
Usage examples:
    python scrape_website.py --url https://example.com
    python scrape_website.py --urls-file urls.txt
    python scrape_website.py --url https://example.com --update
    python scrape_website.py --url https://example.com --use-openai
"""

import argparse
import sys
import os
from typing import List
from urllib.parse import urlparse
import logging
from dotenv import load_dotenv

# Import our scraper (assuming it's in the same directory)
from web_scraper_rag import ContextAwareWebScraper

# Load environment variables
load_dotenv()

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scraper.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def read_urls_from_file(file_path: str) -> List[str]:
    """Read URLs from a text file (one URL per line)"""
    urls = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):  # Skip empty lines and comments
                    if validate_url(url):
                        urls.append(url)
                    else:
                        logging.warning(f"Invalid URL skipped: {url}")
        return urls
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Context-Aware Web Scraper for RAG Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a single URL
  python scrape_website.py --url https://example.com
  
  # Scrape multiple URLs from a file
  python scrape_website.py --urls-file urls.txt
  
  # Scrape with custom delay and update existing
  python scrape_website.py --url https://example.com --delay 2.0 --update
  
  # Scrape with OpenAI instead of Ollama
  python scrape_website.py --url https://example.com --use-openai
  
  # Scrape with custom chunk settings
  python scrape_website.py --url https://example.com --chunk-size 1000 --overlap 150
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--url', help='Single URL to scrape')
    input_group.add_argument('--urls-file', help='File containing URLs (one per line)')
    
    # Scraping options
    parser.add_argument('--update', action='store_true', 
                       help='Update existing chunks (delete old ones first)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--chunk-size', type=int, default=800,
                       help='Maximum chunk size in characters (default: 800)')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Chunk overlap in characters (default: 100)')
    
    # Configuration options
    parser.add_argument('--pinecone-key', 
                       help='Pinecone API key (or set PINECONE_API_KEY env var)')
    parser.add_argument('--openai-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--use-ollama', action='store_true', default=True,
                       help='Use Ollama for embeddings and summaries (default: True)')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI instead of Ollama')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                       help='Ollama host URL (default: http://localhost:11434)')
    parser.add_argument('--namespace', default='website-data',
                       help='Pinecone namespace (default: website-data)')
    
    # Logging options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output (log to file only)')
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.quiet:
        setup_logging(args.log_level)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('scraper.log')]
        )
    
    # Get API keys and provider settings
    pinecone_key = args.pinecone_key or os.getenv('PINECONE_API_KEY')
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    
    # Determine AI provider
    use_ollama = not args.use_openai  # Default to Ollama unless --use-openai is specified
    if args.use_ollama:
        use_ollama = True
    
    ollama_host = args.ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    if not pinecone_key:
        logging.error("Pinecone API key required. Set --pinecone-key or PINECONE_API_KEY env var")
        sys.exit(1)
    
    if not use_ollama and not openai_key:
        logging.error("When using OpenAI, API key is required. Set --openai-key or OPENAI_API_KEY env var")
        sys.exit(1)
    
    # Prepare URLs
    urls = []
    if args.url:
        if validate_url(args.url):
            urls = [args.url]
        else:
            logging.error(f"Invalid URL: {args.url}")
            sys.exit(1)
    elif args.urls_file:
        urls = read_urls_from_file(args.urls_file)
        if not urls:
            logging.error("No valid URLs found in file")
            sys.exit(1)
    
    logging.info(f"Starting to scrape {len(urls)} URL(s)")
    
    try:
        # Initialize scraper
        scraper = ContextAwareWebScraper(
            pinecone_api_key=pinecone_key,
            openai_api_key=openai_key,
            use_ollama=use_ollama,
            ollama_host=ollama_host
        )
        
        # Update namespace if specified
        scraper.namespace = args.namespace
        
        # Update chunk settings if specified
        scraper.max_chunk_size = args.chunk_size
        scraper.chunk_overlap = args.overlap
        
        # Process URLs
        if len(urls) == 1:
            # Single URL
            success = scraper.scrape_and_store(urls[0], update_existing=args.update)
            if success:
                logging.info(f"✅ Successfully processed: {urls[0]}")
                print(f"✅ Successfully scraped and stored: {urls[0]}")
            else:
                logging.error(f"❌ Failed to process: {urls[0]}")
                print(f"❌ Failed to scrape: {urls[0]}")
                sys.exit(1)
        else:
            # Multiple URLs
            results = scraper.scrape_multiple_urls(urls, delay=args.delay)
            
            # Print summary
            successful = sum(1 for success in results.values() if success)
            failed = len(results) - successful
            
            print(f"\n=== Scraping Summary ===")
            print(f"Total URLs: {len(urls)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if not args.quiet:
                print(f"\n=== Detailed Results ===")
                for url, success in results.items():
                    status = "✅" if success else "❌"
                    print(f"{status} {url}")
            
            if failed > 0:
                logging.warning(f"{failed} URLs failed to process")
                sys.exit(1)
        
        logging.info("Scraping completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Scraping failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

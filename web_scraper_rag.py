#!/usr/bin/env python3
"""
Advanced Web Scraper with Context-Aware RAG Storage
Based on best practices from "You're Doing RAG Wrong" article

Features:
- Context-aware chunking with metadata
- Hierarchical content structure preservation
- Dynamic content handling
- Multiple content type support (text, PDFs, videos)
- Pinecone vector storage optimized for retrieval
- Easy updates with smart chunk IDs
- Ollama & OpenAI support
"""

import requests
from bs4 import BeautifulSoup
import pinecone
from pinecone import Pinecone
import openai
import ollama
from urllib.parse import urljoin, urlparse, parse_qs
import hashlib
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import PyPDF2
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContentChunk:
    """Represents a content chunk with context-aware metadata"""
    content: str
    chunk_id: str
    url: str
    title: str
    headers_path: List[str]
    content_type: str
    links: List[Dict[str, str]]
    chunk_index: int
    total_chunks: int
    context_summary: str
    page_summary: str

class ContextAwareWebScraper:
    """Advanced web scraper that creates context-aware chunks for optimal RAG retrieval"""
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str = None, 
                 use_ollama: bool = True, ollama_host: str = "http://localhost:11434"):
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.use_ollama = use_ollama
        self.ollama_host = ollama_host
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "interna-website"
        self.namespace = "website-data"
        
        # Initialize AI providers
        if not use_ollama and not openai_api_key:
            raise ValueError("Either set use_ollama=True or provide openai_api_key")
        
        if use_ollama:
            try:
                # Test Ollama connection
                ollama.Client(host=ollama_host).list()
                self.ollama_client = ollama.Client(host=ollama_host)
                logger.info("Connected to Ollama successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                if not openai_api_key:
                    raise ValueError("Ollama connection failed and no OpenAI key provided")
                self.use_ollama = False
                logger.info("Falling back to OpenAI")
        
        if not use_ollama or (use_ollama and not hasattr(self, 'ollama_client')):
            openai.api_key = openai_api_key
            logger.info("Using OpenAI for embeddings and summaries")
        
        # Connect to existing index
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
        
        # Selenium setup for dynamic content
        self.setup_selenium()
        
        # Chunk settings based on PDF best practices
        self.max_chunk_size = 800  # Optimal for context retention
        self.chunk_overlap = 100   # Prevent context loss
        self.min_chunk_size = 100  # Minimum meaningful chunk size
        
        # Model configurations
        self.embedding_model = "nomic-embed-text" if use_ollama else "text-embedding-3-large"
        self.chat_model = "qwen2.5:7b" if use_ollama else "gpt-3.5-turbo"
        
    def setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.warning(f"Selenium setup failed: {e}. Will use requests only.")
            self.driver = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra whitespace at start and end
        text = text.strip()
        
        # Remove backslashes unless they're meaningful (like URLs)
        text = re.sub(r'\\(?![/\\])', '', text)
        
        # Clean up quotes and special characters
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        
        return text
    
    def remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences from text to improve quality"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and deduplicate
        seen_sentences = set()
        unique_sentences = []
        
        for sentence in sentences:
            cleaned = self.clean_text(sentence)
            if cleaned and len(cleaned) > 10:  # Only keep meaningful sentences
                # Create a normalized version for comparison
                normalized = re.sub(r'\s+', ' ', cleaned.lower().strip())
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    unique_sentences.append(cleaned)
        
        return '. '.join(unique_sentences) + ('.' if unique_sentences else '')
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Ollama or OpenAI"""
        try:
            if self.use_ollama and hasattr(self, 'ollama_client'):
                response = self.ollama_client.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                return response['embedding']
            else:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Fallback to OpenAI if Ollama fails
            if self.use_ollama and self.openai_api_key:
                logger.info("Ollama embedding failed, trying OpenAI fallback")
                try:
                    response = openai.embeddings.create(
                        model="text-embedding-3-large",
                        input=text
                    )
                    return response.data[0].embedding
                except Exception as e2:
                    logger.error(f"OpenAI fallback also failed: {e2}")
            raise
    
    def generate_context_summary(self, content: str, page_context: str) -> str:
        """Generate context-aware summary using Ollama or OpenAI"""
        try:
            prompt = f"""Create a concise context summary for this content chunk that will help with retrieval.

Page Context: {page_context}

Content: {content[:500]}...

Guidelines:
1. Describe what this chunk is about
2. Include relevant keywords
3. Mention the broader context from the page
4. Keep it under 100 words
5. Focus on retrieval relevance

Summary:"""
            
            if self.use_ollama and hasattr(self, 'ollama_client'):
                response = self.ollama_client.chat(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3, "num_predict": 150}
                )
                return response['message']['content'].strip()
            else:
                response = openai.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.warning(f"Failed to generate context summary: {e}")
            # Fallback to OpenAI if Ollama fails
            if self.use_ollama and self.openai_api_key:
                try:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e2:
                    logger.warning(f"OpenAI fallback also failed: {e2}")
            
            return f"Content about: {content[:100]}..."
    
    def get_page_summary(self, soup: BeautifulSoup, url: str) -> str:
        """Generate page-level summary for context"""
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        title_text = self.clean_text(title_text)
        
        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '') if meta_desc else ''
        description = self.clean_text(description)
        
        # Get main headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3'], limit=5):
            heading_text = self.clean_text(h.get_text().strip())
            if heading_text:
                headings.append(heading_text)
        
        summary = f"Page: {title_text}"
        if description:
            summary += f" - {description}"
        if headings:
            summary += f" - Main sections: {', '.join(headings)}"
        
        return self.clean_text(summary)
    
    def extract_content_with_context(self, url: str) -> Tuple[BeautifulSoup, str]:
        """Extract content using both requests and Selenium if needed"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # First try with requests
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if page seems to have dynamic content
            script_tags = soup.find_all('script')
            has_dynamic_content = any('react' in str(script).lower() or 'vue' in str(script).lower() 
                                    or 'angular' in str(script).lower() for script in script_tags)
            
            # If dynamic content detected and Selenium available, use it
            if has_dynamic_content and self.driver:
                logger.info(f"Dynamic content detected for {url}, using Selenium")
                return self.extract_with_selenium(url)
            
            return soup, response.text
            
        except Exception as e:
            logger.warning(f"Requests failed for {url}: {e}")
            if self.driver:
                return self.extract_with_selenium(url)
            raise
    
    def extract_with_selenium(self, url: str) -> Tuple[BeautifulSoup, str]:
        """Extract content using Selenium for dynamic pages"""
        try:
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Execute JavaScript to get final content
            html = self.driver.execute_script("return document.documentElement.outerHTML")
            soup = BeautifulSoup(html, 'html.parser')
            
            return soup, html
            
        except Exception as e:
            logger.error(f"Selenium extraction failed for {url}: {e}")
            raise
    
    def extract_links_and_media(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all relevant links and media for context"""
        links = []
        seen_urls = set()  # Prevent duplicate links
        
        # Extract regular links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = self.clean_text(link.get_text().strip())
            if href and text and href not in seen_urls:
                full_url = urljoin(base_url, href)
                seen_urls.add(href)
                links.append({
                    'type': 'link',
                    'url': full_url,
                    'text': text,
                    'context': 'Internal link'
                })
        
        # Extract PDFs
        for link in soup.find_all('a', href=re.compile(r'\.pdf$', re.I)):
            href = link.get('href')
            if href not in seen_urls:
                text = self.clean_text(link.get_text().strip())
                full_url = urljoin(base_url, href)
                seen_urls.add(href)
                links.append({
                    'type': 'pdf',
                    'url': full_url,
                    'text': text or 'PDF Document',
                    'context': 'PDF resource'
                })
        
        # Extract YouTube links
        for link in soup.find_all('a', href=re.compile(r'youtube\.com|youtu\.be', re.I)):
            href = link.get('href')
            if href not in seen_urls:
                text = self.clean_text(link.get_text().strip())
                seen_urls.add(href)
                links.append({
                    'type': 'youtube',
                    'url': href,
                    'text': text or 'YouTube Video',
                    'context': 'Video content'
                })
        
        return links[:10]  # Limit to top 10 links to avoid noise
    
    def extract_header_hierarchy(self, element, soup: BeautifulSoup) -> List[str]:
        """Extract header hierarchy for context preservation"""
        headers = []
        
        # Find all headers before this element
        all_headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        # Build header stack
        header_stack = []
        for header in all_headers:
            header_text = self.clean_text(header.get_text().strip())
            if header_text:
                level = int(header.name[1])
                
                # Maintain hierarchy
                while header_stack and header_stack[-1]['level'] >= level:
                    header_stack.pop()
                
                header_stack.append({'level': level, 'text': header_text})
                
                # If this header comes before our element, it's part of our context
                try:
                    if element in header.parent.find_all(recursive=True):
                        headers = [h['text'] for h in header_stack]
                        break
                except:
                    pass
        
        return headers[-3:] if len(headers) > 3 else headers  # Keep last 3 levels
    
    def create_context_aware_chunks(self, soup: BeautifulSoup, url: str) -> List[ContentChunk]:
        """Create context-aware chunks following PDF best practices"""
        
        # Clean the soup
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get page-level context
        page_title = soup.find('title')
        page_title_text = self.clean_text(page_title.get_text().strip()) if page_title else urlparse(url).netloc
        
        page_summary = self.get_page_summary(soup, url)
        links_and_media = self.extract_links_and_media(soup, url)
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main', re.I)) or soup.body
        
        if not main_content:
            main_content = soup
        
        # Get all text content with structure, avoiding duplicates
        content_blocks = []
        seen_content = set()
        
        for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
            raw_text = element.get_text().strip()
            cleaned_text = self.clean_text(raw_text)
            
            if cleaned_text and len(cleaned_text) > 20:  # Filter out short/empty content
                # Create normalized version for duplicate detection
                normalized = re.sub(r'\s+', ' ', cleaned_text.lower().strip())
                
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    headers = self.extract_header_hierarchy(element, soup)
                    content_blocks.append({
                        'text': cleaned_text,
                        'headers': headers,
                        'tag': element.name
                    })
        
        # Create chunks with overlap and deduplication
        chunks = []
        current_chunk = ""
        current_headers = []
        chunk_index = 0
        
        for i, block in enumerate(content_blocks):
            block_text = block['text']
            block_headers = block['headers']
            
            # Check if adding this block would exceed chunk size
            potential_chunk = current_chunk + " " + block_text if current_chunk else block_text
            potential_chunk = self.clean_text(potential_chunk)
            
            if len(potential_chunk) > self.max_chunk_size and current_chunk:
                # Deduplicate current chunk before creating
                current_chunk = self.remove_duplicate_sentences(current_chunk)
                
                # Only create chunk if it meets minimum size requirement
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_id = self.generate_chunk_id(url, chunk_index)
                    
                    # Generate context summary
                    context_summary = self.generate_context_summary(current_chunk, page_summary)
                    
                    chunk = ContentChunk(
                        content=current_chunk,
                        chunk_id=chunk_id,
                        url=url,
                        title=page_title_text,
                        headers_path=current_headers,
                        content_type="web_content",
                        links=links_and_media,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will update later
                        context_summary=context_summary,
                        page_summary=page_summary
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = self.clean_text(overlap_text + " " + block_text)
                current_headers = block_headers
            else:
                # Add to current chunk
                current_chunk = potential_chunk
                
                # Update headers if this block has them
                if block_headers:
                    current_headers = block_headers
        
        # Add final chunk if there's content
        if current_chunk and current_chunk.strip():
            current_chunk = self.remove_duplicate_sentences(current_chunk)
            
            if len(current_chunk) >= self.min_chunk_size:
                chunk_id = self.generate_chunk_id(url, chunk_index)
                context_summary = self.generate_context_summary(current_chunk, page_summary)
                
                chunk = ContentChunk(
                    content=current_chunk,
                    chunk_id=chunk_id,
                    url=url,
                    title=page_title_text,
                    headers_path=current_headers,
                    content_type="web_content",
                    links=links_and_media,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks) + 1,
                    context_summary=context_summary,
                    page_summary=page_summary
                )
                
                chunks.append(chunk)
        
        # Update total_chunks for all chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        logger.info(f"Created {total_chunks} context-aware chunks for {url}")
        return chunks
    
    def generate_chunk_id(self, url: str, chunk_index: int) -> str:
        """Generate unique, updateable chunk ID"""
        domain = urlparse(url).netloc.replace('www.', '')
        path_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{domain}_{path_hash}_{chunk_index}"
    
    def create_full_context_for_embedding(self, chunk: ContentChunk) -> str:
        """Create the full context string for embedding (following PDF best practices)"""
        
        context_parts = []
        
        # File/Page summary
        context_parts.append(f"<page_summary>{chunk.page_summary}</page_summary>")
        
        # Chunk summary
        context_parts.append(f"<chunk_summary>{chunk.context_summary}</chunk_summary>")
        
        # Headers hierarchy
        if chunk.headers_path:
            headers_str = " > ".join(chunk.headers_path)
            context_parts.append(f"<headers>{headers_str}</headers>")
        
        # Main content
        context_parts.append(f"<content>{chunk.content}</content>")
        
        return " ".join(context_parts)
    
    def safe_json_serialize(self, obj: Any) -> str:
        """Safely serialize objects to JSON, handling special characters"""
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON serialization failed: {e}")
            # Fallback to string representation
            return str(obj)
    
    def store_in_pinecone(self, chunks: List[ContentChunk]):
        """Store chunks in Pinecone with metadata optimized for retrieval"""
        
        vectors_to_upsert = []
        
        for chunk in chunks:
            try:
                # Create full context for embedding (this gets vectorized)
                full_context = self.create_full_context_for_embedding(chunk)
                
                # Generate embedding
                embedding = self.generate_embeddings(full_context)
                
                # Prepare metadata (this is what gets returned)
                metadata = {
                    "url": chunk.url,
                    "title": chunk.title,
                    "content": chunk.content,  # Return only the main content, not the context
                    "headers_path": " > ".join(chunk.headers_path) if chunk.headers_path else "",
                    "content_type": chunk.content_type,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "context_summary": chunk.context_summary,
                    "page_summary": chunk.page_summary,
                    "links": self.safe_json_serialize(chunk.links[:5]),  # Top 5 links only, safely serialized
                    "scraped_at": datetime.now().isoformat(),
                    "domain": urlparse(chunk.url).netloc
                }
                
                vectors_to_upsert.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
                continue
        
        # Batch upsert to Pinecone
        if vectors_to_upsert:
            try:
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch, namespace=self.namespace)
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
                
                logger.info(f"Successfully stored {len(vectors_to_upsert)} chunks in Pinecone")
                
            except Exception as e:
                logger.error(f"Failed to upsert to Pinecone: {e}")
                raise
    
    def delete_existing_chunks(self, url: str):
        """Delete existing chunks for a URL to enable updates"""
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            path_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            
            # Query for existing chunks
            query_response = self.index.query(
                filter={"url": url},
                top_k=1000,  # Max chunks per URL
                include_metadata=True,
                namespace=self.namespace
            )
            
            if query_response.matches:
                ids_to_delete = [match.id for match in query_response.matches]
                self.index.delete(ids=ids_to_delete, namespace=self.namespace)
                logger.info(f"Deleted {len(ids_to_delete)} existing chunks for {url}")
            
        except Exception as e:
            logger.warning(f"Failed to delete existing chunks for {url}: {e}")
    
    def scrape_and_store(self, url: str, update_existing: bool = True) -> bool:
        """Main method to scrape a URL and store in vector database"""
        
        logger.info(f"Starting to scrape: {url}")
        
        try:
            # Delete existing chunks if updating
            if update_existing:
                self.delete_existing_chunks(url)
            
            # Extract content
            soup, raw_html = self.extract_content_with_context(url)
            
            # Create context-aware chunks
            chunks = self.create_context_aware_chunks(soup, url)
            
            if not chunks:
                logger.warning(f"No content chunks created for {url}")
                return False
            
            # Store in Pinecone
            self.store_in_pinecone(chunks)
            
            logger.info(f"Successfully processed {url} - {len(chunks)} chunks stored")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            return False
    
    def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0) -> Dict[str, bool]:
        """Scrape multiple URLs with rate limiting"""
        results = {}
        
        for i, url in enumerate(urls):
            logger.info(f"Processing {i+1}/{len(urls)}: {url}")
            
            try:
                results[url] = self.scrape_and_store(url)
                
                # Rate limiting
                if delay > 0 and i < len(urls) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                results[url] = False
        
        return results
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass

# Example usage and testing
def main():
    """Example usage of the context-aware web scraper"""
    
    # Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional if using Ollama
    USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() in ("true", "1", "yes")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    if not PINECONE_API_KEY:
        logger.error("Please set PINECONE_API_KEY environment variable")
        return
    
    if not USE_OLLAMA and not OPENAI_API_KEY:
        logger.error("Please set OPENAI_API_KEY environment variable or enable Ollama")
        return
    
    # Initialize scraper
    scraper = ContextAwareWebScraper(
        pinecone_api_key=PINECONE_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        use_ollama=USE_OLLAMA,
        ollama_host=OLLAMA_HOST
    )
    
    # Example URLs to scrape
    urls_to_scrape = [
        "https://example.com",
        "https://docs.example.com/getting-started",
        # Add your URLs here
    ]
    
    # Scrape single URL
    success = scraper.scrape_and_store("https://example.com")
    print(f"Single URL scraping success: {success}")
    
    # Scrape multiple URLs
    results = scraper.scrape_multiple_urls(urls_to_scrape, delay=2.0)
    
    # Print results
    print("\n=== Scraping Results ===")
    for url, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {url}")

if __name__ == "__main__":
    main()

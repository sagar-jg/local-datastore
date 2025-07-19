# Context-Aware Web Scraper for RAG

A sophisticated web scraping system that creates context-aware chunks optimized for Retrieval-Augmented Generation (RAG). This implementation follows best practices from "You're Doing RAG Wrong" to solve common issues like context blindness and first-person confusion.

## 🌟 Features

### Context-Aware Chunking
- **Hierarchical Context Preservation**: Maintains header hierarchy and document structure
- **Smart Overlap**: Prevents context loss between chunks with intelligent overlap
- **Metadata-Rich Storage**: Each chunk includes comprehensive metadata for better retrieval

### AI Provider Support
- **Ollama (Default)**: Free, local AI with `nomic-embed-text` + `qwen2.5:7b`
- **OpenAI (Alternative)**: Cloud-based with `text-embedding-3-large` + `gpt-3.5-turbo`
- **Hybrid Fallback**: Automatically falls back to OpenAI if Ollama fails

### Content Extraction
- **Dynamic Content Support**: Uses Selenium for JavaScript-heavy pages
- **Multi-Format Support**: Handles HTML, PDFs, images, and video links
- **Link Preservation**: Maintains internal links, PDFs, YouTube videos, and images

### Vector Storage Optimization
- **Context-Aware Embeddings**: Embeds full context but returns clean content
- **Smart Chunk IDs**: Easy updates with domain-based, hierarchical IDs
- **Namespace Organization**: Organized storage in Pinecone with dedicated namespaces

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/sagar-jg/local-datastore.git
cd local-datastore

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

Create a `.env` file with your credentials:

```bash
# Pinecone (Required)
PINECONE_API_KEY=your-pinecone-api-key

# AI Provider (Ollama by default)
USE_OLLAMA=true
OLLAMA_HOST=http://localhost:11434

# OpenAI (Optional - fallback or alternative)
OPENAI_API_KEY=your-openai-api-key
```

#### Setting up Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull nomic-embed-text    # For embeddings
ollama pull qwen2.5:7b         # For summarization

# Start Ollama (usually auto-starts)
ollama serve
```

### 3. Basic Usage

#### Scrape a Single URL
```bash
python scrape_website.py --url https://example.com
```

#### Scrape Multiple URLs
```bash
# Create a file with URLs (one per line)
echo "https://example.com" > urls.txt
echo "https://docs.example.com" >> urls.txt

python scrape_website.py --urls-file urls.txt
```

#### Scrape with OpenAI (instead of Ollama)
```bash
python scrape_website.py --url https://example.com --use-openai
```

### 4. Test Retrieval Quality
```bash
python test_retrieval.py
```

## 🎯 **AI Provider Options**

### **Ollama (Default - Recommended)**
- **Free & Local**: No API costs, runs locally
- **Privacy**: Data stays on your machine
- **Models**: Uses `nomic-embed-text` + `qwen2.5:7b`
- **Setup**: Install Ollama and pull models

### **OpenAI (Alternative/Fallback)**
- **Hosted**: Cloud-based, requires API key
- **High Quality**: Latest embedding models
- **Models**: Uses `text-embedding-3-large` + `gpt-3.5-turbo`
- **Cost**: Pay per token usage

### **Hybrid Approach**
The system automatically falls back to OpenAI if Ollama fails:
```python
scraper = ContextAwareWebScraper(
    pinecone_api_key="...",
    openai_api_key="...",  # Optional fallback
    use_ollama=True        # Try Ollama first
)
```

## 📚 Advanced Usage

### Custom Chunk Settings
```bash
python scrape_website.py --url https://example.com \
  --chunk-size 1000 \
  --overlap 150 \
  --delay 2.0
```

### Programmatic Usage
```python
from web_scraper_rag import ContextAwareWebScraper

# Using Ollama (default)
scraper = ContextAwareWebScraper(
    pinecone_api_key="your-key",
    use_ollama=True,
    ollama_host="http://localhost:11434"
)

# Using OpenAI
scraper = ContextAwareWebScraper(
    pinecone_api_key="your-key",
    openai_api_key="your-openai-key",
    use_ollama=False
)

# Hybrid (Ollama with OpenAI fallback)
scraper = ContextAwareWebScraper(
    pinecone_api_key="your-key",
    openai_api_key="your-openai-key",  # For fallback
    use_ollama=True
)

# Scrape single URL
success = scraper.scrape_and_store("https://example.com")

# Scrape multiple URLs
urls = ["https://example.com", "https://docs.example.com"]
results = scraper.scrape_multiple_urls(urls, delay=1.0)
```

## 🏗️ Architecture

### Chunk Structure
Each content chunk includes:

```python
{
    "content": "Main content text",
    "chunk_id": "domain_hash_index",
    "url": "source URL",
    "title": "Page title",
    "headers_path": ["H1", "H2", "H3"],
    "context_summary": "GPT-generated summary",
    "page_summary": "Overall page context",
    "links": [{"type": "link", "url": "...", "text": "..."}],
    "chunk_index": 0,
    "total_chunks": 5
}
```

### Context-Aware Embedding
The system embeds rich context but returns clean content:

**Embedded (for search):**
```xml
<page_summary>
Page overview and main topics
</page_summary>

<chunk_summary>
This chunk covers API authentication methods
</chunk_summary>

<headers>
Documentation > API > Authentication
</headers>

<content>
Actual content text here...
</content>
```

**Returned (for use):**
```
Clean content text without metadata markup
```

## 🛠️ File Structure

```
local-datastore/
├── web_scraper_rag.py      # Main scraper class
├── scrape_website.py       # CLI interface
├── test_retrieval.py       # Retrieval testing
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── README.md              # This file
└── logs/
    └── scraper.log        # Application logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on best practices from "You're Doing RAG Wrong" by DarkBones
- Inspired by context-aware chunking techniques
- Built for optimal retrieval performance

---

**Remember**: Good RAG starts with good chunking. This system prioritizes context preservation over chunk size limits to ensure your retrieval system gets meaningful, contextual results.

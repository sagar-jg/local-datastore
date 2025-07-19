#!/usr/bin/env python3
"""
Test script for the context-aware RAG retrieval system
This helps verify that the stored chunks are optimally retrievable
"""

import os
import json
from typing import List, Dict, Any
import openai
import ollama
from pinecone import Pinecone
from dotenv import load_dotenv
import logging

load_dotenv()

class RAGRetriever:
    """Test class for RAG retrieval from Pinecone"""
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str = None, 
                 use_ollama: bool = True, ollama_host: str = "http://localhost:11434"):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("interna-website")
        self.namespace = "website-data"
        
        self.use_ollama = use_ollama
        self.ollama_host = ollama_host
        
        if use_ollama:
            try:
                self.ollama_client = ollama.Client(host=ollama_host)
                self.ollama_client.list()  # Test connection
                self.embedding_model = "nomic-embed-text"
            except Exception as e:
                if not openai_api_key:
                    raise ValueError(f"Ollama connection failed and no OpenAI key provided: {e}")
                self.use_ollama = False
        
        if not use_ollama:
            openai.api_key = openai_api_key
            self.embedding_model = "text-embedding-3-large"
        
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        try:
            if self.use_ollama and hasattr(self, 'ollama_client'):
                response = self.ollama_client.embeddings(
                    model=self.embedding_model,
                    prompt=query
                )
                return response['embedding']
            else:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=query
                )
                return response.data[0].embedding
        except Exception as e:
            logging.error(f"Failed to generate query embedding: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for relevant chunks"""
        
        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_dict
        )
        
        # Format results
        results = []
        for match in search_results.matches:
            result = {
                'id': match.id,
                'score': match.score,
                'url': match.metadata.get('url', ''),
                'title': match.metadata.get('title', ''),
                'content': match.metadata.get('content', ''),
                'headers': match.metadata.get('headers_path', ''),
                'context_summary': match.metadata.get('context_summary', ''),
                'domain': match.metadata.get('domain', ''),
                'chunk_info': f"{match.metadata.get('chunk_index', 0) + 1}/{match.metadata.get('total_chunks', 1)}"
            }
            results.append(result)
        
        return results
    
    def print_search_results(self, query: str, results: List[Dict]):
        """Pretty print search results"""
        print(f"\n🔍 Query: '{query}'")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📄 {result['title']} (Score: {result['score']:.3f})")
            print(f"   🔗 {result['url']}")
            if result['headers']:
                print(f"   📋 Section: {result['headers']}")
            print(f"   📊 Chunk: {result['chunk_info']}")
            print(f"   💡 Context: {result['context_summary']}")
            print(f"   📝 Content preview: {result['content'][:200]}...")
            print("-" * 60)
    
    def test_various_queries(self):
        """Test various types of queries to evaluate retrieval quality"""
        
        test_queries = [
            # General queries
            "What is this website about?",
            "How do I get started?",
            "What are the main features?",
            
            # Specific queries
            "API documentation",
            "pricing information",
            "contact details",
            "troubleshooting guide",
            
            # Technical queries
            "installation instructions",
            "configuration settings",
            "authentication setup",
            "error codes and solutions",
            
            # Contextual queries
            "frequently asked questions",
            "terms of service",
            "privacy policy",
            "support and help"
        ]
        
        print("🧪 Testing RAG Retrieval Quality")
        print("=" * 80)
        
        for query in test_queries:
            try:
                results = self.search(query, top_k=3)
                self.print_search_results(query, results)
                print("\n" + "="*80 + "\n")
                
            except Exception as e:
                print(f"❌ Error testing query '{query}': {e}")
        
        # Test domain-specific search
        print("\n🏷️  Testing Domain-Specific Search")
        print("=" * 80)
        
        # This would filter by specific domain
        domain_filter = {"domain": "example.com"}  # Replace with actual domain
        try:
            results = self.search("getting started", top_k=5, filter_dict=domain_filter)
            self.print_search_results("getting started (domain-filtered)", results)
        except Exception as e:
            print(f"❌ Error in domain-filtered search: {e}")
    
    def analyze_index_stats(self):
        """Analyze the current state of the index"""
        try:
            stats = self.index.describe_index_stats()
            print("\n📊 Index Statistics")
            print("=" * 50)
            print(f"Total vectors: {stats.total_vector_count}")
            print(f"Dimension: {stats.dimension}")
            
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for namespace, info in stats.namespaces.items():
                    print(f"Namespace '{namespace}': {info.vector_count} vectors")
            
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")

def main():
    """Test the RAG retrieval system"""
    
    # Get API keys and settings
    pinecone_key = os.getenv('PINECONE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    use_ollama = os.getenv('USE_OLLAMA', 'true').lower() in ('true', '1', 'yes')
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    if not pinecone_key:
        print("❌ Error: Please set PINECONE_API_KEY environment variable")
        return
    
    if not use_ollama and not openai_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable or enable Ollama")
        return
    
    try:
        # Initialize retriever
        retriever = RAGRetriever(
            pinecone_key, 
            openai_key, 
            use_ollama=use_ollama,
            ollama_host=ollama_host
        )
        
        # Show index stats
        retriever.analyze_index_stats()
        
        # Test retrieval
        retriever.test_various_queries()
        
        # Interactive mode
        print("\n🎯 Interactive Query Mode")
        print("Enter queries to test retrieval (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = retriever.search(query, top_k=5)
                retriever.print_search_results(query, results)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")

if __name__ == "__main__":
    main()

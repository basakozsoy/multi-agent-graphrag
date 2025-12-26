"""
Language Model and Embedding Model Configuration

Initializes LLM and embedding models based on environment configuration.
Supports multiple providers:
- OpenAI: Cloud-based API with state-of-the-art models
- Ollama: Local deployment for privacy and cost optimization (default)
"""
import os
from llama_index.core import Settings


def create_models():
    """
    Initialize and configure language model and embedding model.
    
    Returns:
        tuple: (llm, embedding_model) configured based on MODEL_PROVIDER environment variable
        
    Environment Variables:
        MODEL_PROVIDER: "openai" or "ollama" (default: "ollama")
    """
    provider = os.getenv("MODEL_PROVIDER", "ollama")
    
    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        llm = OpenAI(model="gpt-4-turbo-preview")
        embed = OpenAIEmbedding(model="text-embedding-3-small")
    
    else:
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        ollama_llm = os.getenv('OLLAMA_LLM_MODEL', 'qwen2.5:7b')
        ollama_embed = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        llm = Ollama(model=ollama_llm, base_url=ollama_url)
        embed = OllamaEmbedding(model_name=ollama_embed, base_url=ollama_url)
    
    Settings.llm = llm
    Settings.embed_model = embed
    
    return llm, embed

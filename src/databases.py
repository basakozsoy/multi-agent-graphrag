"""
Database Interface Layer

Provides unified interfaces for vector and graph database operations.

Classes:
    VectorStore: Manages document embeddings in Qdrant for semantic similarity search
    GraphStore: Manages entity relationships in Neo4j for graph-based retrieval
    
The hybrid approach combines semantic search with relationship-based queries
for comprehensive document retrieval.
"""
import os
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from qdrant_client import QdrantClient


class VectorStore:
    """
    Vector database interface for Qdrant.
    
    Manages document indexing and semantic similarity search using vector embeddings.
    Utilizes cosine similarity for document retrieval.
    """
    def __init__(self, collection_name="documents"):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=collection_name
        )
        
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
    
    def index_documents(self, documents):
        """
        Index documents using semantic chunking for topic-boundary detection.
        
        Args:
            documents: List of documents to index
            
        Returns:
            VectorStoreIndex: Configured index with semantic chunking
        """
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        Settings.text_splitter = semantic_splitter
        
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=self.storage_context,
            transformations=[semantic_splitter]
        )
        return self.index
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)


class GraphStore:
    """
    Graph database interface for Neo4j.
    
    Manages entity extraction, relationship mapping, and graph-based document retrieval.
    Enables complex queries based on entity relationships.
    """
    def __init__(self):
        self.graph_store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password123"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        self.index = None
    
    def build_graph(self, documents, max_triplets=3):
        """
        Build knowledge graph from documents
        
        Uses LLM to extract:
        - Entities: Alice, Bob, Project Phoenix
        - Relationships: (Alice)-[:CEO_OF]->(Company), (Bob)-[:WORKS_ON]->(Project Phoenix)
        """
        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=max_triplets,  # How many relationships to extract per chunk
            include_embeddings=True  # Also store text for retrieval
        )
        return self.index
    
    def search(self, query, top_k=5):
        """Search the knowledge graph"""
        if not self.index:
            self.index = KnowledgeGraphIndex([], storage_context=self.storage_context)
        
        retriever = self.index.as_retriever(
            retriever_mode="keyword",  # Search by keywords in the graph
            similarity_top_k=top_k
        )
        return retriever.retrieve(query)

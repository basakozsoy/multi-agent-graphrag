"""
Document Indexing Script

Indexes documents into vector and graph databases:
1. Loads documents from data/sample_documents/
2. Creates vector embeddings and stores in Qdrant
3. Extracts entities and relationships for Neo4j knowledge graph

Must be executed before running the demo.

Usage:
    python scripts/index_documents.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import SimpleDirectoryReader
from src.databases import VectorStore, GraphStore
from src.models import create_models

load_dotenv()


def main():
    print("\nIndexing Documents\n" + "="*60)
    
    create_models()
    
    docs_path = Path(__file__).parent.parent / "data" / "sample_documents"
    documents = SimpleDirectoryReader(str(docs_path)).load_data()
    print(f"Loaded {len(documents)} documents")
    
    vector_store = VectorStore(collection_name="acme_docs")
    vector_store.index_documents(documents)
    print("Vector indexing complete")
    
    skip_graph = os.getenv("SKIP_GRAPH_BUILD", "false").lower() == "true"
    
    if skip_graph:
        print("Skipping graph build (Vector + BM25 only)")
    else:
        print("Building knowledge graph...")
        graph_store = GraphStore()
        
        success_count = 0
        for i, doc in enumerate(documents, 1):
            try:
                if i == 1:
                    graph_store.build_graph([doc], max_triplets=2)
                else:
                    from llama_index.core import KnowledgeGraphIndex
                    new_index = KnowledgeGraphIndex.from_documents(
                        [doc],
                        storage_context=graph_store.storage_context,
                        max_triplets_per_chunk=2,
                        include_embeddings=True,
                        show_progress=False
                    )
                    graph_store.index = new_index
                success_count += 1
            except Exception as e:
                continue
        
        if success_count > 0:
            print(f"Graph building complete ({success_count}/{len(documents)} documents)")
        else:
            print("Graph building failed (Vector + BM25 only)")
    
    print("\n" + "="*60)
    print("Indexing complete. Run: python scripts/demo.py")
    print("="*60)


if __name__ == "__main__":
    main()

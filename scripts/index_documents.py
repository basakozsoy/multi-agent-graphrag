"""
Document Indexing Script

Indexes documents into vector and graph databases:
1. Loads documents from data/sample_documents/ (with layout-aware PDF parsing)
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

from llama_index.core import Document
from src.databases import VectorStore, GraphStore
from src.models import create_models

load_dotenv()


def load_documents_with_docling(docs_path):
    """
    Load documents using Docling for layout-aware parsing.
    
    Docling converts PDFs to structured Markdown, preserving:
    - Headers (# ## ###) for hierarchical structure
    - Tables (| | |) for tabular data
    - Multi-column layouts
    
    For non-PDF files, falls back to simple text loading.
    """
    from docling.document_converter import DocumentConverter
    
    converter = DocumentConverter()
    documents = []
    
    for file_path in Path(docs_path).glob("**/*"):
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == '.pdf':
                    # Use Docling for PDFs - converts to structured Markdown
                    result = converter.convert(str(file_path))
                    markdown_text = result.document.export_to_markdown()
                    
                    doc = Document(
                        text=markdown_text,
                        metadata={"file_name": file_path.name, "file_path": str(file_path)}
                    )
                    documents.append(doc)
                    print(f"✓ Loaded (Docling): {file_path.name}")
                else:
                    # Plain text files
                    text = file_path.read_text(encoding='utf-8')
                    doc = Document(
                        text=text,
                        metadata={"file_name": file_path.name, "file_path": str(file_path)}
                    )
                    documents.append(doc)
                    print(f"✓ Loaded (text): {file_path.name}")
            except Exception as e:
                print(f"✗ Failed to load {file_path.name}: {e}")
                continue
    
    return documents


def main():
    print("\nIndexing Documents (Hierarchical Parent-Child Chunks)\n" + "="*60)
    
    create_models()
    
    docs_path = Path(__file__).parent.parent / "data" / "sample_documents"
    documents = load_documents_with_docling(str(docs_path))
    print(f"\nLoaded {len(documents)} documents")
    
    print("\nCreating hierarchical chunks (Parent → Children)...")
    vector_store = VectorStore(collection_name="acme_docs")
    vector_store.index_documents(documents, use_hierarchical=True)
    print("✓ Vector indexing complete with parent-child hierarchy")
    
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

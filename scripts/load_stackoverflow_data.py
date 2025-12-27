"""
Python Stack Overflow Data Loader

Loads Python programming Q&A from Stack Overflow.

Usage:
    # Load 100 Python questions
    python scripts/load_engineering_data.py
    
    # Load more questions
    python scripts/load_engineering_data.py --limit 200
    
    # Higher quality threshold
    python scripts/load_engineering_data.py --limit 50 --min-score 20
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Document
from src.databases import VectorStore, GraphStore
from src.models import create_models

load_dotenv()


class StackOverflowLoader:
    """Load Q&A from Stack Overflow via Stack Exchange API"""
    
    def __init__(self):
        self.base_url = "https://api.stackexchange.com/2.3"
    
    def load_questions(self, tag, max_results=50, min_score=10):
        """
        Load questions by tag
        
        Args:
            tag: Tag to filter (e.g., 'python', 'machine-learning', 'docker')
            max_results: Maximum number of questions to fetch
            min_score: Minimum question score to include
        """
        import requests
        import time
        
        print(f"\n📥 Loading Stack Overflow questions tagged '{tag}'...")
        
        documents = []
        page = 1
        page_size = 100
        
        while len(documents) < max_results:
            params = {
                'page': page,
                'pagesize': min(page_size, max_results - len(documents)),
                'order': 'desc',
                'sort': 'votes',
                'tagged': tag,
                'site': 'stackoverflow',
                'filter': 'withbody',  # Include question body
                'min': min_score
            }
            
            try:
                response = requests.get(f"{self.base_url}/questions", params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'items' not in data or not data['items']:
                    break
                
                for item in data['items']:
                    # Combine question title and body
                    text = f"""# {item['title']}

**Score**: {item['score']} | **Views**: {item['view_count']} | **Answers**: {item['answer_count']}

## Question
{item.get('body_markdown', item.get('body', ''))}

**Tags**: {', '.join(item.get('tags', []))}
**Asked**: {item.get('creation_date', 'N/A')}
**Link**: {item.get('link', 'N/A')}
"""
                    
                    doc = Document(
                        text=text,
                        metadata={
                            'source': 'stackoverflow',
                            'question_id': item['question_id'],
                            'title': item['title'],
                            'tags': item.get('tags', []),
                            'score': item['score'],
                            'answer_count': item['answer_count'],
                            'link': item.get('link', '')
                        }
                    )
                    documents.append(doc)
                
                print(f"   Loaded {len(documents)}/{max_results} questions...", end='\r')
                
                # Check if there are more pages
                if not data.get('has_more', False):
                    break
                
                page += 1
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"\n   Error loading page {page}: {e}")
                break
        
        print(f"\n✓ Loaded {len(documents)} Stack Overflow questions")
        return documents


def main():
    parser = argparse.ArgumentParser(description='Load Python Stack Overflow Q&A data')
    parser.add_argument('--limit', type=int, default=100,
                       help='Number of questions to load (default: 100)')
    parser.add_argument('--min-score', type=int, default=10,
                       help='Minimum score for questions (default: 10)')
    parser.add_argument('--skip-graph', action='store_true',
                       help='Skip knowledge graph building (faster)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Python Stack Overflow Data Loader")
    print("="*70)
    
    # Initialize models
    create_models()
    
    # Load Python Stack Overflow questions
    print("\nLoading Python programming Q&A from Stack Overflow...\n")
    so_loader = StackOverflowLoader()
    all_documents = so_loader.load_questions(
        tag='python',
        max_results=args.limit,
        min_score=args.min_score
    )
    
    if not all_documents:
        print("\n✗ No documents loaded. Exiting.")
        return
    
    print(f"\n{'='*70}")
    print(f"Total documents loaded: {len(all_documents)}")
    print("="*70)
    
    # Index documents
    print("\n📊 Indexing into vector database...")
    vector_store = VectorStore(collection_name='stackoverflow_docs')
    vector_store.index_documents(all_documents)
    print("✓ Vector indexing complete")
    
    # Build knowledge graph (optional)
    if not args.skip_graph:
        print("\n🕸️  Building knowledge graph...")
        graph_store = GraphStore()
        
        # Process in batches to avoid memory issues
        batch_size = 5
        success_count = 0
        
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i+batch_size]
            try:
                if i == 0:
                    graph_store.build_graph(batch, max_triplets=2)
                else:
                    from llama_index.core import KnowledgeGraphIndex
                    new_index = KnowledgeGraphIndex.from_documents(
                        batch,
                        storage_context=graph_store.storage_context,
                        max_triplets_per_chunk=2,
                        include_embeddings=True,
                        show_progress=False
                    )
                    graph_store.index = new_index
                success_count += len(batch)
                print(f"   Processed {min(i+batch_size, len(all_documents))}/{len(all_documents)} documents...", end='\r')
            except Exception as e:
                continue
        
        print(f"\n✓ Graph building complete ({success_count}/{len(all_documents)} documents)")
    else:
        print("\n⏭️  Skipping graph build (use --skip-graph=false to enable)")
    
    print("\n" + "="*70)
    print("✓ Indexing complete!")
    print("\nRun demo: python scripts/demo_engineering.py")
    print("="*70)


if __name__ == "__main__":
    main()

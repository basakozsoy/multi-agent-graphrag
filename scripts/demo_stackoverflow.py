"""
Python RAG Demo

Demonstrates the multi-agent system with Python Q&A from Stack Overflow.

Usage:
    # First, load data:
    python scripts/load_engineering_data.py
    
    # Then run demo:
    python scripts/demo_engineering.py
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.databases import VectorStore, GraphStore
from src.retriever import HybridRetriever
from src.orchestrator import create_multi_agent_system
from src.models import create_models

load_dotenv()


def main():
    print("\n� Python Multi-Agent RAG Demo\n" + "="*70)
    
    create_models()
    
    # Use Python Q&A collection
    vector_store = VectorStore(collection_name="stackoverflow_docs")
    graph_store = GraphStore()
    retriever = HybridRetriever(vector_store, graph_store, verbose=True)
    agent_system = create_multi_agent_system(retriever=retriever, max_iterations=3)
    
    print("System initialized with Python Q&A data\n")
    
    # Python-specific demo queries
    demo_queries = [
        "How do I sort a dictionary by value in Python?",
        "What is the difference between append and extend?",
        "How to read a CSV file in Python?",
    ]
    
    print("Running demo queries...\n")
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(demo_queries)}: {query}")
        print("="*70)
        
        result = agent_system.invoke({
            "query": query,
            "plan": "",
            "retrieved_docs": [],
            "retrieval_quality": 0.0,
            "feedback": "",
            "iteration": 0,
            "final_answer": "",
            "max_iterations": 3,
            "skip_planning": True
        })
        
        print(f"\n📊 ANSWER:\n{result['final_answer']}")
        print(f"\n[Iterations: {result['iteration']} | Quality: {result['retrieval_quality']:.2f}]")
        
        if i < len(demo_queries):
            input("\nPress Enter for next query...")
    
    # Interactive mode
    print("\n" + "="*70)
    print("🤔 Interactive Mode - Ask Python questions!")
    print("="*70)
    print("\nExamples:")
    print("  • How to use list comprehensions?")
    print("  • What is the difference between == and is?")
    print("  • How to handle exceptions in Python?")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("\n💡 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = agent_system.invoke({
                "query": question,
                "plan": "",
                "retrieved_docs": [],
                "retrieval_quality": 0.0,
                "feedback": "",
                "iteration": 0,
                "final_answer": "",
                "max_iterations": 3,
                "skip_planning": True
            })
            
            print(f"\n📊 ANSWER:\n{result['final_answer']}")
            print(f"\n[Iterations: {result['iteration']} | Quality: {result['retrieval_quality']:.2f}]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n👋 Thanks for using Python RAG!")


if __name__ == "__main__":
    main()

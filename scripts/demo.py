"""
Multi-Agent RAG System Demonstration

Demonstrates the complete multi-agent retrieval workflow including:
1. Triple hybrid retrieval (Vector + Graph + BM25)
2. Multi-agent orchestration with LangGraph
3. Cyclic feedback mechanism between Retriever and Reviewer
4. Specialized agent collaboration: Planner, Retriever, Reviewer, Analyst

Usage:
    python scripts/demo.py
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
    print("\nMulti-Agent Self-Correcting RAG Demo\n" + "="*70)
    
    create_models()
    
    vector_store = VectorStore(collection_name="acme_docs")
    graph_store = GraphStore()
    retriever = HybridRetriever(vector_store, graph_store)
    agent_system = create_multi_agent_system(retriever=retriever, max_iterations=3)
    
    print("System initialized\n")
    
    demo_queries = [
        "Who is the CEO of Acme Corporation?",
        "What is Project Phoenix and who leads it?",
        "What are the parental leave benefits for new fathers?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\nQuery {i}/{len(demo_queries)}: {query}")
        print("-" * 70)
        
        result = agent_system.invoke({
            "query": query,
            "plan": "",
            "retrieved_docs": [],
            "retrieval_quality": 0.0,
            "feedback": "",
            "iteration": 0,
            "final_answer": "",
            "max_iterations": 3
        })
        
        print(f"\n{result['final_answer']}")
        print(f"\n[Iterations: {result['iteration']} | Quality: {result['retrieval_quality']:.2f}]")
        
        if i < len(demo_queries):
            input("\nPress Enter for next query...")
    
    print("\n" + "="*70)
    print("Interactive Mode (type 'quit' to exit)\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
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
                "max_iterations": 3
            })
            
            print(f"\n{result['final_answer']}")
            print(f"\n[Iterations: {result['iteration']} | Quality: {result['retrieval_quality']:.2f}]\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Hybrid Retrieval System

Implements triple hybrid retrieval combining:
1. Vector search - semantic similarity using Qdrant
2. Graph search - entity relationship traversal using Neo4j  
3. BM25 search - keyword-based statistical ranking

Results are merged using Reciprocal Rank Fusion (RRF) with optimized weighting.
"""
import os
from pathlib import Path
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever


class HybridRetriever:
    """
    Triple hybrid retrieval system combining vector, graph, and keyword search.
    
    Supports multiple retrieval strategies:
        - hybrid: Combines all three methods with weighted fusion (default)
        - vector_only: Semantic similarity search only
        - graph_only: Entity relationship search only
        - bm25_only: Keyword-based search only
    """
    
    def __init__(self, vector_store, graph_store):
    
    def retrieve(self, query, top_k=5, strategy="hybrid"):
        """
        Main retrieval method
        
        Args:
            query: User's question
            top_k: How many documents to return
            strategy: "hybrid", "vector_only", "graph_only", or "bm25_only"
        """
        if strategy == "vector_only":
            return self.vector_store.search(query, top_k)
        
        elif strategy == "graph_only":
            return self.graph_store.search(query, top_k)
        
        elif strategy == "bm25_only":
            return self._bm25_search(query, top_k)
        
        else:
            vector_results = self.vector_store.search(query, top_k)
            graph_results = self.graph_store.search(query, top_k)
            bm25_results = self._bm25_search(query, top_k)
            
            return self._merge_results(
                [vector_results, graph_results, bm25_results], 
                weights=[0.6, 0.1, 0.3],
                top_k=top_k
            )
    
    def _bm25_search(self, query, top_k=5):
        """
        Execute BM25 keyword-based search.
        
        BM25 (Best Match 25) is a probabilistic retrieval function used for ranking.
        Optimized for exact keyword matches.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        if not self.bm25_retriever:
            docs_path = Path(__file__).parent.parent / "data" / "sample_documents"
            if docs_path.exists():
                documents = SimpleDirectoryReader(str(docs_path)).load_data()
                nodes = [TextNode(text=doc.text, metadata=doc.metadata) for doc in documents]
                if nodes:
                    self.bm25_retriever = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=top_k
                    )
        
        if self.bm25_retriever:
            return self.bm25_retriever.retrieve(query)
        return []
    
    def _merge_results(self, results_list, weights, top_k):
        """
        Reciprocal Rank Fusion (RRF) for multiple retrievers
        
        Handles 3 result lists instead of 2.
        
        RRF formula: score = weight / (k + rank)
        
        Args:
            results_list: List of [vector_results, graph_results, bm25_results]
            weights: List of [vector_weight, graph_weight, bm25_weight]
            top_k: Final number of results
        """
        scores = {}
        k = 60  # RRF constant
        
        # Process each retriever's results
        for results, weight in zip(results_list, weights):
            for rank, node in enumerate(results):
                node_id = node.node.node_id
                rrf_score = weight / (k + rank + 1)
                scores[node_id] = scores.get(node_id, 0) + rrf_score
        
        # Combine all unique nodes
        all_nodes = {}
        for results in results_list:
            for node in results:
                all_nodes[node.node.node_id] = node
        
        # Sort by merged score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return top-k with new scores
        return [
            NodeWithScore(node=all_nodes[node_id].node, score=scores[node_id])
            for node_id in sorted_ids[:top_k]
            if node_id in all_nodes
        ]

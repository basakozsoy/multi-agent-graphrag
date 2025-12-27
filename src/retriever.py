"""
Hybrid Retrieval System

Implements triple hybrid retrieval combining:
1. Vector search - semantic similarity using Qdrant
2. Graph search - entity relationship traversal using Neo4j  
3. BM25 search - keyword-based statistical ranking

Results are merged using Reciprocal Rank Fusion (RRF) with optimized weighting
and automatic deduplication to ensure diverse, non-redundant results.

Deduplication Strategy:
- Identifies duplicate documents by text content (first 200 chars)
- Merges scores for duplicate content across different retrieval methods
- Ensures final result set contains only unique documents
"""
import os
from pathlib import Path
from langsmith import traceable
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
    
    def __init__(self, vector_store, graph_store, verbose=False):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: VectorStore instance for semantic search
            graph_store: GraphStore instance for relationship-based search
            verbose: If True, log deduplication statistics
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.bm25_retriever = None  # Lazy initialization
        self.verbose = verbose
    
    @traceable(name="hybrid_retrieve", tags=["retrieval", "hybrid"])
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
            vector_results = self._vector_search_traced(query, top_k)
            graph_results = self._graph_search_traced(query, top_k)
            bm25_results = self._bm25_search_traced(query, top_k)
            
            return self._merge_results(
                [vector_results, graph_results, bm25_results], 
                weights=[0.6, 0.1, 0.3],
                top_k=top_k
            )
    
    @traceable(name="vector_search", tags=["retrieval", "vector"])
    def _vector_search_traced(self, query, top_k):
        """Vector search with tracing"""
        results = self.vector_store.search(query, top_k)
        return results
    
    @traceable(name="graph_search", tags=["retrieval", "graph"])
    def _graph_search_traced(self, query, top_k):
        """Graph search with tracing"""
        results = self.graph_store.search(query, top_k)
        return results
    
    @traceable(name="bm25_search", tags=["retrieval", "bm25"])
    def _bm25_search_traced(self, query, top_k):
        """BM25 search with tracing"""
        return self._bm25_search(query, top_k)
    
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
    
    @traceable(name="rrf_merge", tags=["retrieval", "fusion"])
    def _merge_results(self, results_list, weights, top_k):
        """
        Reciprocal Rank Fusion (RRF) for multiple retrievers with deduplication.
        
        Handles 3 result lists instead of 2.
        
        RRF formula: score = weight / (k + rank)
        
        Deduplication strategy:
        1. Group by node_id (primary dedup key)
        2. Additionally deduplicate by text content hash to catch duplicates
           with different node_ids from different retrieval methods
        3. Merge scores for true duplicates
        
        Args:
            results_list: List of [vector_results, graph_results, bm25_results]
            weights: List of [vector_weight, graph_weight, bm25_weight]
            top_k: Final number of results
        """
        scores = {}
        k = 60  # RRF constant
        
        # Track individual scores per method for tracing
        method_names = ["vector", "graph", "bm25"]
        score_breakdown = {method: {} for method in method_names}
        
        # Process each retriever's results
        for method_name, results, weight in zip(method_names, results_list, weights):
            for rank, node in enumerate(results):
                node_id = node.node.node_id
                rrf_score = weight / (k + rank + 1)
                scores[node_id] = scores.get(node_id, 0) + rrf_score
                
                # Track breakdown per method
                score_breakdown[method_name][node_id] = {
                    "rank": rank + 1,
                    "weight": weight,
                    "rrf_score": rrf_score
                }
        
        # Log score breakdown if verbose
        if self.verbose:
            print(f"\n   📊 RRF Score Breakdown (top 3):")
            sorted_by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for node_id, total_score in sorted_by_score:
                print(f"   Node {node_id[:12]}... : {total_score:.4f}")
                for method in method_names:
                    if node_id in score_breakdown[method]:
                        info = score_breakdown[method][node_id]
                        print(f"      {method:8} → rank {info['rank']:2} × {info['weight']:.1f} = {info['rrf_score']:.4f}")
        
        # Combine all unique nodes
        all_nodes = {}
        for results in results_list:
            for node in results:
                all_nodes[node.node.node_id] = node
        
        # DEDUPLICATION STEP: Identify duplicates by text content
        # Map text hash -> list of node_ids with that text
        text_to_nodes = {}
        for node_id, node_with_score in all_nodes.items():
            # Use first 200 chars as hash key (handles slight variations)
            text_key = node_with_score.node.text[:200].strip()
            if text_key not in text_to_nodes:
                text_to_nodes[text_key] = []
            text_to_nodes[text_key].append(node_id)
        
        # Count duplicates found
        total_before_dedup = len(all_nodes)
        num_duplicate_groups = sum(1 for node_ids in text_to_nodes.values() if len(node_ids) > 1)
        
        # Merge scores for duplicate content, keep highest-scored node_id
        deduped_scores = {}
        deduped_nodes = {}
        for text_key, node_ids in text_to_nodes.items():
            # Sum scores from all duplicate node_ids
            merged_score = sum(scores.get(nid, 0) for nid in node_ids)
            # Keep the node with highest individual score as representative
            best_node_id = max(node_ids, key=lambda nid: scores.get(nid, 0))
            deduped_scores[best_node_id] = merged_score
            deduped_nodes[best_node_id] = all_nodes[best_node_id]
        
        # Log deduplication stats if verbose
        if self.verbose:
            total_after_dedup = len(deduped_nodes)
            duplicates_removed = total_before_dedup - total_after_dedup
            if duplicates_removed > 0:
                print(f"   🔄 Deduplication: {total_before_dedup} → {total_after_dedup} docs "
                      f"({duplicates_removed} duplicates removed in {num_duplicate_groups} groups)")
        
        # Sort by merged score
        sorted_ids = sorted(deduped_scores.keys(), key=lambda x: deduped_scores[x], reverse=True)
        
        # Return top-k with new scores
        return [
            NodeWithScore(node=deduped_nodes[node_id].node, score=deduped_scores[node_id])
            for node_id in sorted_ids[:top_k]
            if node_id in deduped_nodes
        ]

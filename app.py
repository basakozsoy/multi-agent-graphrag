"""
Streamlit UI for Multi-Agent RAG System

Interactive interface to visualize:
- Hybrid retrieval with score breakdowns
- Multi-agent workflow execution
- Entity graph visualization
- Real-time query processing

Usage:
    streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.databases import VectorStore, GraphStore
from src.retriever import HybridRetriever
from src.orchestrator import create_multi_agent_system
from src.models import create_models

load_dotenv()

# Page config
st.set_page_config(
    page_title="Multi-Agent RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    with st.spinner("Initializing RAG system..."):
        create_models()
        st.session_state.vector_store = VectorStore(collection_name="acme_docs")
        st.session_state.graph_store = GraphStore()
        st.session_state.retriever = HybridRetriever(
            st.session_state.vector_store, 
            st.session_state.graph_store,
            verbose=False
        )
        st.session_state.agent_system = create_multi_agent_system(
            st.session_state.retriever
        )
        st.session_state.initialized = True
        st.session_state.history = []

# Header
st.title("Multi-Agent RAG System")
st.markdown("**GraphRAG + Hierarchical Chunking + Entity Resolution**")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    mode = st.radio(
        "Mode",
        ["Multi-Agent (Full)", "Retrieval Only"],
        help="Multi-Agent: Full pipeline with self-correction. Retrieval: Just search."
    )
    
    strategy = st.selectbox(
        "Retrieval Strategy",
        ["hybrid", "vector_only", "bm25_only", "graph_only"],
        help="Hybrid combines all 3 methods (Vector 60%, BM25 30%, Graph 10%)"
    )
    
    top_k = st.slider("Results per method", 3, 10, 5)
    
    show_scores = st.checkbox("Show score breakdown", value=True)
    show_sources = st.checkbox("Show source documents", value=True)
    
    st.divider()
    st.markdown("### System Stats")
    
    # Get collection info
    try:
        collection_info = st.session_state.vector_store.client.get_collection("acme_docs")
        st.metric("Vector Documents", collection_info.points_count)
    except:
        st.metric("Vector Documents", "N/A")
    
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("• Triple Hybrid Retrieval")
    st.markdown("• Hierarchical Chunking")
    st.markdown("• Entity Resolution")
    st.markdown("• Self-Correcting Agents")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Query Interface")
    
    # Sample queries
    sample_queries = [
        "Who is the CEO of Acme Corporation?",
        "What is Project Phoenix and who leads it?",
        "What are the parental leave benefits?",
        "Custom query..."
    ]
    
    selected_sample = st.selectbox("Sample Queries", sample_queries)
    
    if selected_sample == "Custom query...":
        query = st.text_input("", "", placeholder="Enter your question...")
    else:
        query = st.text_input("", selected_sample, label_visibility="collapsed")
    
    if st.button("Search", type="primary", use_container_width=True):
        if query:
            with st.spinner("Processing query..."):
                if mode == "Retrieval Only":
                    # Just retrieval
                    results = st.session_state.retriever.retrieve(
                        query, 
                        top_k=top_k, 
                        strategy=strategy
                    )
                    
                    st.success(f"Retrieved {len(results)} results")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - Score: {result.score:.4f}"):
                            st.markdown(result.node.text)
                            if result.node.metadata:
                                st.json(result.node.metadata)
                    
                    # Store in history
                    st.session_state.history.append({
                        "query": query,
                        "mode": mode,
                        "strategy": strategy,
                        "results": len(results)
                    })
                    
                else:
                    # Full multi-agent system
                    state = {
                        "query": query,
                        "plan": "",
                        "retrieved_docs": [],
                        "best_docs": [],
                        "best_quality": 0.0,
                        "retrieval_quality": 0.0,
                        "feedback": "",
                        "iteration": 0,
                        "final_answer": "",
                        "max_iterations": 3,
                        "skip_planning": False
                    }
                    
                    # Run agent system
                    final_state = st.session_state.agent_system.invoke(state)
                    
                    # Display answer
                    st.success("Answer Generated")
                    st.markdown("### Answer")
                    st.markdown(final_state.get("final_answer", "No answer generated"))
                    
                    # Show iterations
                    st.info(f"Completed in {final_state.get('iteration', 0)} iterations | Quality Score: {final_state.get('retrieval_quality', 0):.2f}")
                    
                    # Show documents used
                    if show_sources and final_state.get("best_docs"):
                        st.markdown("### Source Documents")
                        for i, doc in enumerate(final_state["best_docs"][:3], 1):
                            with st.expander(f"Source {i}"):
                                st.markdown(doc.node.text[:500] + "...")
                    
                    # Store in history
                    st.session_state.history.append({
                        "query": query,
                        "mode": mode,
                        "iterations": final_state.get('iteration', 0),
                        "score": final_state.get('retrieval_quality', 0)
                    })

with col2:
    st.header("Visualizations")
    
    if show_scores and st.session_state.history:
        st.markdown("### Retrieval Weights")
        
        # Create pie chart for hybrid retrieval weights
        if strategy == "hybrid":
            fig = go.Figure(data=[go.Pie(
                labels=['Vector Search', 'BM25 Keyword', 'Graph Traversal'],
                values=[60, 30, 10],
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1']),
                hole=.3
            )])
            fig.update_layout(
                title="Hybrid Retrieval Distribution",
                height=300,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Query History")
        
        # Show recent queries
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.history) - i + 1}", expanded=(i==1)):
                st.markdown(f"**Q:** {item['query'][:50]}...")
                st.markdown(f"**Mode:** {item['mode']}")
                if 'iterations' in item:
                    st.markdown(f"**Iterations:** {item['iterations']}")
                    st.markdown(f"**Score:** {item.get('score', 0):.2f}")
                if 'results' in item:
                    st.markdown(f"**Results:** {item['results']}")

# Architecture diagram
with st.expander("System Architecture"):
    st.markdown("""
    ### Multi-Agent Workflow
    ```
    User Query
        │
        ▼
    [RETRIEVER] ──→ Vector (60%) + BM25 (30%) + Graph (10%)
        │
        ▼
    [REVIEWER] ──→ Quality Score (0-1)
        │
        ├─ Score ≥ 0.5 ──→ [ANALYST] ──→ Final Answer
        └─ Score < 0.5 ──→ Retry (max 3×)
    ```
    
    ### Document Processing
    - **Hierarchical Chunking**: Parent (1500 tokens) → Children (300 tokens)
    - **Entity Resolution**: Fuzzy matching deduplicates entities (85% threshold)
    - **Layout-Aware**: Docling parses PDFs to structured Markdown
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Multi-Agent RAG System | LangGraph + LlamaIndex + Neo4j + Qdrant</p>
</div>
""", unsafe_allow_html=True)

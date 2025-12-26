# Multi-Agent Self-Correcting RAG System

A production-ready retrieval-augmented generation system featuring multi-agent orchestration with LangGraph, cyclic feedback loops, semantic chunking, and triple hybrid retrieval.

**Implements**: GraphRAG + Hybrid Indexing + Agentic Self-Correction with LlamaIndex, Neo4j, and Qdrant

## Key Features

- **Multi-Agent Orchestration**: Four specialized agents (Planner, Retriever, Reviewer, Analyst) coordinated via LangGraph state machine
- **Triple Hybrid Retrieval**: Combines vector search (60%), BM25 keyword matching (30%), and graph traversal (10%)
- **Cyclic Feedback Loop**: Automatic quality evaluation with intelligent retry and strategy rotation
- **Semantic Chunking**: Topic-boundary detection that preserves contextual relationships
- **Knowledge Graph Integration**: Neo4j-powered entity relationship mapping for complex queries
- **Local or Cloud**: Runs locally with Ollama or use OpenAI API for enhanced quality

## Architecture

### Multi-Agent Workflow
```
                    User Query
                        │
                        ▼
        ┌───────────────────────────────┐
        │   PLANNER (Optional)          │
        │   Analyzes & strategizes      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   RETRIEVER                   │
        │   Hybrid search + caching     │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   REVIEWER                    │
        │   Scores quality (0-1)        │
        └───────────────┬───────────────┘
                        │
                ┌───────┴────────┐
                │                │
            Quality          Quality
            < 0.5            ≥ 0.5
                │                │
                │                ▼
                │    ┌───────────────────────┐
                │    │   ANALYST             │
                │    │   Synthesizes answer  │
                │    └───────────┬───────────┘
                │                │
                │                ▼
                │         Final Answer
                │
                └─→ RETRY (max 2x)
                        │
                        └─→ [back to RETRIEVER]
```

### Triple Hybrid Retrieval
```
                    Query
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   Vector (60%)   BM25 (30%)   Graph (10%)
    Semantic       Keywords    Relationships
    (Qdrant)                     (Neo4j)
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
          Reciprocal Rank Fusion
                      │
                      ▼
              Top-5 Documents
```

## Getting Started

```bash
# Run the automated setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate

# Index sample documents
python scripts/index_documents.py

# Run the multi-agent demo
python scripts/demo.py
```

## Project Structure

```
agentic-rag/
├── src/
│   ├── models.py        # LLM & embedding initialization
│   ├── databases.py     # Vector (Qdrant) & Graph (Neo4j) wrappers
│   ├── retriever.py     # Triple hybrid retriever (Vector+BM25+Graph)
│   ├── orchestrator.py  # Multi-agent system with LangGraph
│   └── agent.py         # Legacy single-agent (for comparison)
├── scripts/
│   ├── index_documents.py  # Load and index data
│   └── demo.py             # Multi-agent demo
├── data/
│   └── sample_documents/   # Sample company docs
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Agent Components

- **Planner Agent**: Analyzes query and creates retrieval strategy (optional, can be skipped for speed)
- **Retriever Agent**: Executes hybrid search, rotates strategies based on feedback, caches results
- **Reviewer Agent**: Evaluates quality (0-1 score), triggers retry if < 0.5, tracks best documents
- **Analyst Agent**: Synthesizes final answer from best documents with source citations

## System Operation

**Example Query**: "What is Project Phoenix and who leads it?"

1. **Iteration 1** (Hybrid): Quality 0.0 - missing specific details → Retry
2. **Iteration 2** (Vector-only): Quality 0.0 - still insufficient → Retry  
3. **Iteration 3** (BM25 keyword): Quality 0.8 - found project details → **Approved**
4. **Result**: "Project Phoenix is Acme's AI initiative led by Bob Smith (CTO) and Dr. Emily Chen (Technical Lead) [Sources: 1, 2]"

## Production Readiness

- Multi-agent architecture with LangGraph state machine (373 lines)
- Cyclic feedback loop with up to 2 retry iterations
- Triple hybrid retrieval (Vector 60%, BM25 30%, Graph 10%)
- Semantic chunking at topic boundaries
- Result caching and quality-based routing

## Configuration

The system configuration is managed through environment variables in `.env`:

```env
# Model Provider: "ollama" (local) or "openai" (cloud)
MODEL_PROVIDER=ollama

# Ollama Configuration (local deployment)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# OpenAI Configuration (cloud API)
OPENAI_API_KEY=your_key_here

# Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=acme_docs
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
```

## Example Queries (from demo.py)

1. **"Who is the CEO of Acme Corporation?"** - Quality 1.00, immediate approval
2. **"What is Project Phoenix and who leads it?"** - Self-corrects over 3 iterations (hybrid → vector → BM25)
3. **"What are the parental leave benefits?"** - BM25 finds exact section with semantic chunking


## Tech Stack

**Framework:** LangGraph 0.2 • LlamaIndex 0.12 • Python 3.12+  
**LLMs:** Ollama (qwen2.5:7b, nomic-embed-text) • OpenAI (gpt-4, text-embedding-3)  
**Databases:** Qdrant (vector) • Neo4j (graph)  
**Retrieval:** BM25 keyword ranking • Semantic chunking  
**Infrastructure:** Docker & Docker Compose

## Potential Feature Roadmap

**Multimodal Support**
- [ ] Image and video document indexing
- [ ] Audio transcription and retrieval
- [ ] Cross-modal search capabilities

**Scalability**
- [ ] Separate UI, RAG application, and vector store using REST API design
- [ ] Implement API gateway and load balancing
- [ ] Horizontal scaling with container orchestration (Kubernetes)
- [ ] Database connection pooling and replication

**Response Time Optimization**
- [ ] Cache frequent queries and their results (Redis)
- [ ] Cache commonly retrieved documents
- [ ] Hardware scaling (GPU acceleration for embeddings)
- [ ] Async processing and streaming responses
- [ ] Pre-compute embeddings for static documents
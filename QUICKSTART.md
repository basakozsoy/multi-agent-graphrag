# Quick Start Guide

## 🚀 5-Minute Setup

```bash
# 1. Run automated setup (installs everything)
./setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Index sample documents
python scripts/index_documents.py

# 4. Run the demo!
python scripts/demo.py
```

## 📋 What You'll See

The demo runs 3 queries showing the multi-agent system:

### Query 1: "Who is the CEO?"
✅ **1 iteration** - Perfect retrieval (Quality: 1.00)

### Query 2: "What is Project Phoenix?"
✅ **3 iterations** - Self-correcting (hybrid → vector → BM25 ✓)

### Query 3: "Parental leave for fathers?"
✅ **Finds answer** - "Paternity Leave: 8 weeks fully paid"

## 🎮 Interactive Mode

After the demo, you can ask your own questions:

```
🤔 Your question: Who founded Acme Corporation?

[Agent workflow runs...]

📊 ANSWER:
Alice Johnson founded Acme Corporation in 2010...
```

## 🔧 Quick Config Changes

### Switch to OpenAI (Better Quality)
```bash
# Edit .env
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Re-run
python scripts/demo.py
```

### Skip Planning (Faster)
Edit [scripts/demo.py](scripts/demo.py#L67):
```python
agent_system = create_multi_agent_system(
    retriever,
    skip_planning=True,  # ← Change to True
    max_iterations=2
)
```

### Adjust Quality Threshold
Edit [src/orchestrator.py](src/orchestrator.py#L166):
```python
quality = float(result.get('quality', 0))
if quality >= 0.5:  # ← Change threshold (lower = less strict)
```

## 🐛 Troubleshooting

### "Connection refused" (Qdrant)
```bash
docker-compose up -d
```

### "Ollama not responding"
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart if needed
ollama serve
```

### "Out of memory" during indexing
```bash
# Skip graph building (faster, less memory)
echo "SKIP_GRAPH_BUILD=true" >> .env
python scripts/index_documents.py
```

## 📊 Performance Tips

| Optimization | Speed Gain | Trade-off |
|-------------|------------|-----------|
| `skip_planning=True` | ~2s/query | Slightly less intelligent routing |
| `max_iterations=1` | ~50% faster | No self-correction |
| `top_k=3` | ~20% faster | Might miss relevant docs |
| Skip graph build | ~70% faster indexing | No relationship queries |

## 📝 Your Own Documents

Replace files in `data/sample_documents/`:
```bash
rm data/sample_documents/*.txt
cp your-docs/*.txt data/sample_documents/

# Re-index
python scripts/index_documents.py

# Query
python scripts/demo.py
```

## 💡 Next Steps

- **Add your data**: Replace sample documents
- **Tune parameters**: Adjust quality thresholds, top_k, weights
- **Monitor**: Track quality scores, iteration counts
- **Scale**: Add more agents, retrieval strategies

## 📚 Learn More

- Full docs: [README.md](README.md)
- Architecture: [README.md#architecture](README.md#-architecture)
- Configuration: [README.md#configuration](README.md#-configuration)

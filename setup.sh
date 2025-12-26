#!/bin/bash
# Quick Setup Script for Multi-Agent RAG System

echo "======================================"
echo "Multi-Agent RAG - Quick Setup"
echo "======================================"

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "  ✓ venv already exists"
else
    python3 -m venv venv
    echo "  ✓ venv created"
fi

# Step 2: Activate and install dependencies
echo ""
echo "Step 2: Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Step 3: Create .env if needed
echo ""
echo "Step 3: Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  ✓ .env created (using Ollama by default - FREE!)"
else
    echo "  ✓ .env already exists"
fi

# Step 4: Download models
echo ""
echo "Step 4: Downloading models..."
MODEL_PROVIDER=$(grep "^MODEL_PROVIDER=" .env | cut -d'=' -f2)

if [ "$MODEL_PROVIDER" = "ollama" ]; then
    echo "  Downloading Ollama models..."
    if command -v ollama &> /dev/null; then
        ollama pull qwen2.5:7b
        ollama pull nomic-embed-text
        echo "  ✓ Ollama models downloaded"
    else
        echo "  ⚠️  Ollama not found. Please install: https://ollama.com"
        echo "  Then run: ollama pull qwen2.5:7b && ollama pull nomic-embed-text"
    fi
else
    echo "  Using OpenAI - no models to download"
fi

# Step 5: Start databases
echo ""
echo "Step 5: Starting databases (Docker)..."
# Check if containers are already running
if docker ps --format '{{.Names}}' | grep -q 'rag-neo4j\|rag-qdrant'; then
    echo "  ✓ Databases already running"
else
    docker-compose up -d
    echo "  ✓ Databases starting..."
    echo ""
    echo "Waiting for databases to be ready (10 seconds)..."
    sleep 10
fi

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
if [ "$MODEL_PROVIDER" = "ollama" ]; then
    echo "2. Index sample documents:"
    echo "   python scripts/index_documents.py"
    echo ""
    echo "   (Ollama is already running in background)"
else
    echo "2. Set your OPENAI_API_KEY in .env"
    echo ""
    echo "3. Index documents:"
    echo "   python scripts/index_documents.py"
fi
echo ""
echo "4. Run the multi-agent demo:"
echo "   python scripts/demo.py"
echo ""
echo "======================================"
echo "Database URLs:"
echo "======================================"
echo "  Qdrant:  http://localhost:6333"
echo "  Neo4j:   http://localhost:7474"
echo "           (user: neo4j, pass: password123)"

echo ""
echo "To stop databases: docker-compose down"
echo "To deactivate venv: deactivate"
echo ""

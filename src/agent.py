"""
Self-Correcting Agent - The Core Innovation!

This is what makes the system "agentic" and "self-correcting"

The agent loop:
1. Retrieve documents using hybrid search
2. Evaluate: "Can these documents answer the question?" (uses LLM to judge)
3. If score < threshold → try a different strategy
4. Repeat until good enough OR max iterations

Why this is powerful:
- Traditional RAG: retrieves once, hopes it's good
- Self-correcting RAG: keeps trying until quality threshold met
- Can adapt strategy based on what's working
"""
from .models import create_models


class SelfCorrectingAgent:
    """
    Agent that evaluates and improves its own retrieval
    
    Parameters:
    - retriever: The HybridRetriever instance
    - max_iterations: Max attempts before giving up (default: 3)
    - threshold: Minimum quality score to accept (default: 0.7)
    """
    
    def __init__(self, retriever, max_iterations=3, threshold=0.7):
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.llm, _ = create_models()  # Get LLM for evaluation
    
    def query(self, question):
        """
        Main query method with self-correction loop
        
        Returns:
            dict with 'answer', 'score', 'iterations'
        """
        print(f"\n{'='*60}\nQUERY: {question}\n{'='*60}")
        
        best_nodes = None
        best_score = 0
        strategy = "hybrid"  # Start with hybrid
        
        # Self-correction loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration} (Strategy: {strategy}) ---")
            
            # STEP 1: Retrieve documents
            nodes = self.retriever.retrieve(question, strategy=strategy)
            print(f"Retrieved {len(nodes)} documents")
            
            # STEP 2: Evaluate quality (ask LLM to judge)
            score = self._evaluate(question, nodes)
            print(f"Relevance score: {score:.2f}")
            
            # Track best attempt
            if score > best_score:
                best_score = score
                best_nodes = nodes
            
            # STEP 3: Check if good enough
            if score >= self.threshold:
                print(f"✓ Threshold met! ({score:.2f} >= {self.threshold})")
                break
            
            # STEP 4: Try different strategy
            print(f"✗ Below threshold, trying different strategy...")
            # Rotate through strategies: hybrid → vector_only → graph_only → hybrid...
            strategy = ["vector_only", "graph_only", "hybrid"][iteration % 3]
        
        # Generate final answer using best retrieved documents
        print(f"\n--- Generating Answer (best score: {best_score:.2f}) ---")
        answer = self._generate_answer(question, best_nodes)
        
        return {
            'answer': answer,
            'score': best_score,
            'iterations': iteration
        }
    
    def _evaluate(self, question, nodes):
        """
        Evaluate retrieval quality using LLM
        
        How it works:
        1. Show LLM the question + retrieved documents
        2. Ask: "Can these documents answer this question?"
        3. LLM returns a score 0.0 to 1.0
        
        This is the "self-evaluation" - the agent judges itself!
        """
        if not nodes:
            return 0.0
        
        # Take first 3 documents, truncate to 200 chars each
        context = "\n\n".join([
            f"Doc {i+1}: {n.text[:200]}..." 
            for i, n in enumerate(nodes[:3])
        ])
        
        # Prompt for LLM
        prompt = f"""Rate how well these documents can answer the question.
Question: {question}

Documents:
{context}

Respond with ONLY a number between 0.0 and 1.0 (e.g., 0.8)
- 1.0 = perfect answer possible
- 0.5 = partial answer possible
- 0.0 = cannot answer"""
        
        try:
            response = self.llm.complete(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))  # Clamp to 0-1 range
        except:
            return 0.5  # Default if parsing fails
    
    def _generate_answer(self, question, nodes):
        """
        Generate final answer using best retrieved documents
        
        Standard RAG approach:
        1. Take retrieved documents as context
        2. Give to LLM with the question
        3. LLM generates answer based on context
        """
        if not nodes:
            return "Sorry, I couldn't find relevant information to answer this question."
        
        # Build context from top 5 documents
        context = "\n\n".join([
            f"[Document {i+1}]\n{n.text}" 
            for i, n in enumerate(nodes[:5])
        ])
        
        # Prompt for answer generation
        prompt = f"""Answer the question based ONLY on the context below.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.llm.complete(prompt)
        return response.text.strip()

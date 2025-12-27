"""
Multi-Agent Orchestration System

Implements a state machine with four specialized agents coordinated via LangGraph:

1. Planner Agent - Analyzes queries and formulates retrieval strategies
2. Retriever Agent - Executes multi-strategy document retrieval
3. Reviewer Agent - Evaluates retrieval quality and provides feedback
4. Analyst Agent - Synthesizes final answers from retrieved documents

The system implements a cyclic feedback mechanism where the Reviewer can
trigger additional retrieval iterations with strategy adjustments.
"""
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from .models import create_models

# Define the state that flows through the graph
class AgentState(TypedDict, total=False):
    """
    State object passed between agents in the workflow.
    
    Attributes:
        query: Original user question
        plan: Retrieval strategy from Planner
        retrieved_docs: Documents from current iteration
        best_docs: Highest quality documents across all iterations
        best_quality: Best quality score achieved
        retrieval_quality: Current iteration quality score (0-1)
        feedback: Reviewer feedback for strategy refinement
        iteration: Current retrieval attempt number
        final_answer: Generated answer from Analyst
        max_iterations: Maximum allowed retrieval attempts
    """
    query: str
    plan: str
    retrieved_docs: List
    best_docs: List
    best_quality: float
    retrieval_quality: float
    feedback: str
    iteration: int
    final_answer: str
    max_iterations: int
    skip_planning: bool

class PlannerAgent:
    """
    Query analysis and retrieval strategy formulation agent.
    
    Analyzes incoming queries to determine optimal retrieval approach.
    Can be disabled for reduced latency in performance-critical scenarios.
    """
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute planning (optional - can be skipped for speed)"""
        # Fast path: skip planning for speed, just use hybrid
        if state.get('skip_planning', False):
            print("\n🎯 PLANNER AGENT (skipped for speed)")
            state['plan'] = "Using hybrid retrieval (all methods combined)"
            state['iteration'] = 0
            return state
        
        print("\n🎯 PLANNER AGENT")
        print(f"   Analyzing query: {state['query']}")
        
        # Shorter prompt for speed
        prompt = f"""Query: {state['query']}

Choose strategy (1 sentence):
- hybrid: general questions
- vector_only: conceptual/semantic
- graph_only: relationships/entities
- bm25_only: exact terms

Strategy:"""
        
        response = self.llm.complete(prompt)
        plan = response.text.strip()
        
        print(f"   Plan: {plan[:100]}...")
        
        state['plan'] = plan
        state['iteration'] = 0
        return state

class RetrieverAgent:
    """
    Document retrieval execution agent.
    
    Executes retrieval using various strategies based on planner guidance
    or reviewer feedback. Implements result caching and adaptive strategy rotation.
    Supports multiple invocations for iterative refinement.
    """
    def __init__(self, retriever):
        self.retriever = retriever
        self._cache = {}
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute document retrieval with adaptive strategy selection."""
        # Initialize iteration if not present, then increment
        state['iteration'] = state.get('iteration', 0) + 1
        iteration = state['iteration']
        query = state['query']
        
        print(f"\n🔍 RETRIEVER AGENT (Iteration {iteration})")
        
        cache_key = f"{query}:hybrid"
        if iteration == 1 and cache_key in self._cache:
            print("   Using cached results")
            state['retrieved_docs'] = self._cache[cache_key]
            return state
        
        if iteration == 1:
            strategy = "hybrid"
            print(f"   Using initial strategy: {strategy}")
        else:
            strategies = ["vector_only", "bm25_only", "graph_only"]
            strategy = strategies[(iteration - 2) % len(strategies)]
            print(f"   Adjusting based on feedback: {strategy}")
            feedback = state.get('feedback', '')
            if feedback:
                print(f"   Feedback: {feedback[:100]}...")
        
        # Execute retrieval
        docs = self.retriever.retrieve(
            query, 
            top_k=5,  # Get more chunks to ensure we capture the right section
            strategy=strategy
        )
        
        # Cache hybrid results for reuse
        if strategy == "hybrid" and iteration == 1:
            self._cache[cache_key] = docs
        
        print(f"   Retrieved {len(docs)} documents")
        
        state['retrieved_docs'] = docs
        return state

class ReviewerAgent:
    """
    REVIEWER AGENT
    
    Role: Evaluate retrieval quality and provide feedback
    
    What it does:
    1. Examines retrieved documents
    2. Scores quality (0.0 to 1.0)
    3. If score < threshold: provides feedback to improve retrieval
    4. If score >= threshold: approves to move to Analyst
    
    This is the KEY INNOVATION - creates the feedback loop!
    """
    def __init__(self, llm, quality_threshold=0.7):
        self.llm = llm
        self.quality_threshold = quality_threshold
    
    def __call__(self, state: AgentState) -> AgentState:
        """Evaluate retrieval quality"""
        print(f"\n✅ REVIEWER AGENT")
        
        docs = state['retrieved_docs']
        if not docs:
            state['retrieval_quality'] = 0.0
            state['feedback'] = "No documents retrieved. Try different search strategy."
            print(f"   Quality: 0.0 - No documents found")
            return state
        
        # Create evaluation prompt (shortened for speed)
        context = "\n".join([
            f"Doc{i+1}: {doc.text[:150]}"
            for i, doc in enumerate(docs[:3])
        ])
        
        prompt = f"""Query: {state['query']}

Docs:
{context}

Can these docs answer the query?
SCORE: <0.0-1.0>
FEEDBACK: <if needed>
"""
        
        response = self.llm.complete(prompt)
        result = response.text.strip()
        
        # Parse response
        score = self._extract_score(result)
        feedback = self._extract_feedback(result)
        
        state['retrieval_quality'] = score
        state['feedback'] = feedback
        
        # Track best documents so far
        current_best = state.get('best_quality', 0.0)
        if score > current_best and docs:
            state['best_docs'] = docs
            state['best_quality'] = score
        
        if score >= self.quality_threshold:
            print(f"   Quality: {score:.2f} ✓ APPROVED")
        else:
            print(f"   Quality: {score:.2f} ✗ NEEDS IMPROVEMENT")
            print(f"   Feedback: {feedback[:100]}...")
        
        return state
    
    def _extract_score(self, text: str) -> float:
        """Extract score from response"""
        try:
            if "SCORE:" in text:
                score_line = [l for l in text.split('\n') if 'SCORE:' in l][0]
                score = float(score_line.split('SCORE:')[1].strip().split()[0])
                return max(0.0, min(1.0, score))
        except:
            pass
        return 0.5  # Default
    
    def _extract_feedback(self, text: str) -> str:
        """Extract feedback from response"""
        try:
            if "FEEDBACK:" in text:
                feedback = text.split('FEEDBACK:')[1].strip()
                return feedback
        except:
            pass
        return "Try different retrieval strategy"

class AnalystAgent:
    """
    ANALYST AGENT
    
    Role: Generate final answer from approved retrieval results
    
    What it does:
    1. Takes the approved documents (quality >= threshold)
    2. Synthesizes information
    3. Generates comprehensive answer
    4. Cites sources
    
    This only runs when Reviewer approves the retrieval!
    """
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate final answer"""
        print(f"\n📊 ANALYST AGENT")
        
        # Use best documents found across all iterations
        docs = state.get('best_docs', state.get('retrieved_docs', []))
        best_quality = state.get('best_quality', 0.0)
        print(f"   Synthesizing answer from {len(docs)} documents (quality: {best_quality:.2f})...")
        
        # If no documents or very low quality, return early
        if not docs or best_quality < 0.1:
            state['final_answer'] = "I couldn't find relevant information to answer your question. The retrieved documents don't contain information about this topic."
            return state
        
        # Build context (limit length for low-quality docs to prevent timeouts)
        max_docs = 3 if best_quality < 0.3 else 5
        max_chars_per_doc = 300 if best_quality < 0.3 else 1000
        
        context = "\n\n".join([
            f"[Source {i+1}]\n{doc.text[:max_chars_per_doc]}"
            for i, doc in enumerate(docs[:max_docs])
        ])
        
        prompt = f"""You are an expert analyst. Provide a comprehensive answer based on the retrieved information.

Original Query: {state['query']}

Retrieved Information:
{context}

Instructions:
1. Answer the query thoroughly
2. Cite specific sources [Source X]
3. If information is incomplete or irrelevant, acknowledge it clearly
4. Be concise but complete

Answer:"""
        
        try:
            response = self.llm.complete(prompt)
            answer = response.text.strip()
            print(f"   Generated answer ({len(answer)} chars)")
            state['final_answer'] = answer
        except Exception as e:
            print(f"   ⚠️  Error generating answer: {str(e)[:100]}")
            state['final_answer'] = f"I encountered an issue generating the answer. The retrieved documents may not contain sufficient information about: {state['query']}"
        
        return state


# Routing functions - decide which agent to call next
def should_continue_retrieval(state: AgentState) -> Literal["analyst", "retriever"]:
    """
    Decision point: Continue retrieving or move to analysis?
    
    Logic:
    - If quality >= 0.5 → go to ANALYST (good enough!)
    - If iterations < max AND quality < 0.5 → go to RETRIEVER (retry!)
    - Otherwise → go to ANALYST (give up, use best we have)
    """
    quality = state.get('retrieval_quality', 0)
    iteration = state.get('iteration', 0)
    max_iter = state.get('max_iterations', 2)
    
    # Accept if quality meets threshold
    if quality >= 0.5:
        print(f"\n→ Quality sufficient ({quality:.2f}), moving to ANALYST")
        return "analyst"
    # Retry if we have attempts left
    elif iteration < max_iter:
        print(f"\n→ Quality low ({quality:.2f}), retrying RETRIEVAL (attempt {iteration + 1}/{max_iter})")
        return "retriever"
    # Out of attempts, use best available
    else:
        print(f"\n→ Max iterations reached, proceeding to ANALYST with best available")
        return "analyst"

def create_multi_agent_system(retriever, max_iterations=2, skip_planning=True):
    """
    Build the LangGraph state machine
    
    Args:
        retriever: HybridRetriever instance
        max_iterations: Max retrieval attempts (default: 2)
        skip_planning: Skip planner for speed (default: True)
    
    Returns a compiled graph that orchestrates all agents
    """
    # Initialize agents
    llm, _ = create_models()
    
    planner = PlannerAgent(llm)
    retriever_agent = RetrieverAgent(retriever)
    reviewer = ReviewerAgent(llm, quality_threshold=0.5)
    analyst = AnalystAgent(llm)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    if not skip_planning:
        workflow.add_node("planner", planner)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("analyst", analyst)
    
    # Define edges (flow)
    if skip_planning:
        workflow.set_entry_point("retriever")              # Skip planner for speed
    else:
        workflow.set_entry_point("planner")                # Start with planner
        workflow.add_edge("planner", "retriever")          # Planner → Retriever
    
    workflow.add_edge("retriever", "reviewer")             # Retriever → Reviewer
    workflow.add_conditional_edges(
        "reviewer",                                         # From Reviewer...
        should_continue_retrieval,                          # ...decide where to go
        {
            "retriever": "retriever",                       # Loop back if quality low
            "analyst": "analyst"                            # Or proceed if quality good
        }
    )
    workflow.add_edge("analyst", END)                      # Analyst → Done
    
    # Compile
    app = workflow.compile()
    
    return app

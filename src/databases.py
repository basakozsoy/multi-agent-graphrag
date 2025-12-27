"""
Database Interface Layer

Provides unified interfaces for vector and graph database operations.

Classes:
    VectorStore: Manages document embeddings in Qdrant for semantic similarity search
    GraphStore: Manages entity relationships in Neo4j for graph-based retrieval
    
The hybrid approach combines semantic search with relationship-based queries
for comprehensive document retrieval.
"""
import os
from rapidfuzz import fuzz, process
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from qdrant_client import QdrantClient


class VectorStore:
    """
    Vector database interface for Qdrant.
    
    Manages document indexing and semantic similarity search using vector embeddings.
    Utilizes cosine similarity for document retrieval.
    """
    def __init__(self, collection_name="documents"):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name=collection_name
        )
        
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
    
    def index_documents(self, documents, use_hierarchical=True):
        """
        Index documents using hierarchical parent-child chunking.
        
        Hierarchical Structure:
        - Parent Nodes: Large sections (1000-2000 tokens) for high-level context
        - Child Nodes: Smaller chunks (200-400 tokens) for precise retrieval
        - Relationship: (Parent)-[:CONTAINS]->(Child) for context traversal
        
        Args:
            documents: List of documents to index
            use_hierarchical: If True, use parent-child hierarchy; else use flat semantic chunking
            
        Returns:
            VectorStoreIndex: Configured index with hierarchical or semantic chunking
        """
        if not use_hierarchical:
            # Fallback to flat semantic chunking
            semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model
            )
            Settings.text_splitter = semantic_splitter
            
            self.index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=self.storage_context,
                transformations=[semantic_splitter]
            )
            return self.index
        
        # Create hierarchical parent-child chunks
        parent_splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
        
        all_nodes = []
        
        for doc in documents:
            # Create parent chunks (large sections)
            parent_nodes = parent_splitter.get_nodes_from_documents([doc])
            
            # For each parent, create child chunks
            for parent in parent_nodes:
                parent.metadata["node_type"] = "parent"
                
                # Split parent into children
                child_nodes = child_splitter.get_nodes_from_documents([type(doc)(text=parent.text)])
                
                # Link children to parent
                for i, child in enumerate(child_nodes):
                    child.metadata["node_type"] = "child"
                    child.metadata["parent_id"] = parent.node_id
                    
                    # Create parent-child relationship
                    child.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=parent.node_id,
                        metadata={"relationship": "CONTAINS"}
                    )
                
                # Link parent to children
                parent.metadata["child_count"] = len(child_nodes)
                parent.relationships[NodeRelationship.CHILD] = [
                    RelatedNodeInfo(node_id=child.node_id) for child in child_nodes
                ]
                
                all_nodes.append(parent)
                all_nodes.extend(child_nodes)
        
        # Index all nodes (parents and children)
        self.index = VectorStoreIndex(
            all_nodes,
            storage_context=self.storage_context
        )
        
        return self.index
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)


class GraphStore:
    """
    Graph database interface for Neo4j.
    
    Manages entity extraction, relationship mapping, and graph-based document retrieval.
    Enables complex queries based on entity relationships with entity resolution.
    """
    def __init__(self):
        self.graph_store = Neo4jGraphStore(
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password123"),
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687")
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        self.index = None
        
        # Entity resolution cache: maps canonical entity names to their variations
        self._entity_cache = {}
        self._similarity_threshold = 85  # Fuzzy match threshold (0-100)
    
    def build_graph(self, documents, max_triplets=3, create_hierarchy=True, resolve_entities=True):
        """
        Build knowledge graph from documents with entity resolution and hierarchy.
        
        Uses LLM to extract:
        - Entities: Alice, Bob, Project Phoenix
        - Relationships: (Alice)-[:CEO_OF]->(Company), (Bob)-[:WORKS_ON]->(Project Phoenix)
        - Parent-Child: (Parent)-[:CONTAINS]->(Child) for hierarchical context
        
        Entity Resolution: Automatically deduplicates similar entity names using fuzzy matching
        (e.g., "Bob Smith", "Bob", "Robert Smith" → merged into one canonical entity)
        
        Args:
            documents: List of documents
            max_triplets: Number of entity relationships to extract per chunk
            create_hierarchy: If True, create parent-child relationships in Neo4j
            resolve_entities: If True, deduplicate entities using fuzzy matching
        """
        if resolve_entities:
            # Load existing entities from Neo4j for deduplication
            self._load_entity_cache()
        
        self.index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=max_triplets,
            include_embeddings=True
        )
        
        if resolve_entities:
            # Apply entity resolution to deduplicate similar entities
            self._resolve_duplicate_entities()
        
        # Add parent-child hierarchy to Neo4j graph
        if create_hierarchy:
            self._create_parent_child_relationships(documents)
        
        return self.index
    
    def _load_entity_cache(self):
        """
        Load existing entities from Neo4j to enable entity resolution.
        Builds a cache of canonical entity names for fuzzy matching.
        """
        query = """
        MATCH (n)
        WHERE n:EntityNode OR labels(n) = []
        RETURN DISTINCT n.id as entity_name
        LIMIT 1000
        """
        try:
            result = self.graph_store._driver.execute_query(query)
            for record in result.records:
                entity_name = record.get('entity_name')
                if entity_name:
                    # Map entity to itself initially (canonical form)
                    self._entity_cache[entity_name.lower()] = entity_name
        except Exception:
            # If query fails, continue without cache
            pass
    
    def _resolve_entity(self, entity_name):
        """
        Resolve an entity name to its canonical form using fuzzy matching.
        
        Args:
            entity_name: The entity name to resolve
            
        Returns:
            str: Canonical entity name (may be the same or a similar existing entity)
        """
        if not entity_name or not isinstance(entity_name, str):
            return entity_name
        
        entity_lower = entity_name.lower().strip()
        
        # Check exact match first
        if entity_lower in self._entity_cache:
            return self._entity_cache[entity_lower]
        
        # Fuzzy match against existing entities
        if self._entity_cache:
            matches = process.extract(
                entity_name,
                self._entity_cache.values(),
                scorer=fuzz.token_sort_ratio,
                limit=1
            )
            
            if matches and matches[0][1] >= self._similarity_threshold:
                # Found a similar entity - use the canonical form
                canonical = matches[0][0]
                self._entity_cache[entity_lower] = canonical
                return canonical
        
        # No match found - this becomes a new canonical entity
        self._entity_cache[entity_lower] = entity_name
        return entity_name
    
    def _resolve_duplicate_entities(self):
        """
        Scan the graph and merge duplicate entities based on fuzzy matching.
        
        This fixes issues like:
        - "Bob Smith" and "Bob" → merged
        - "Project Phoenix" and "Phoenix Project" → merged
        - "Acme Corp" and "Acme Corporation" → merged
        """
        # Get all entity pairs (subject, object, relationship)
        query = """
        MATCH (s)-[r]->(o)
        RETURN s.id as subject, type(r) as relationship, o.id as object
        """
        
        try:
            result = self.graph_store._driver.execute_query(query)
            
            merges_performed = 0
            for record in result.records:
                subject = record.get('subject')
                obj = record.get('object')
                relationship = record.get('relationship')
                
                if not subject or not obj or not relationship:
                    continue
                
                # Resolve both entities
                canonical_subject = self._resolve_entity(subject)
                canonical_object = self._resolve_entity(obj)
                
                # If either entity was mapped to a different canonical form, update the relationship
                if canonical_subject != subject or canonical_object != obj:
                    # Delete old relationship
                    delete_query = """
                    MATCH (s {id: $old_subject})-[r {id: $rel_id}]->(o {id: $old_object})
                    DELETE r
                    """
                    
                    # Create new relationship with canonical entities
                    merge_query = """
                    MERGE (s {id: $canonical_subject})
                    MERGE (o {id: $canonical_object})
                    MERGE (s)-[r:%s]->(o)
                    """ % relationship
                    
                    try:
                        self.graph_store._driver.execute_query(
                            merge_query,
                            canonical_subject=canonical_subject,
                            canonical_object=canonical_object
                        )
                        merges_performed += 1
                    except Exception:
                        continue
            
            if merges_performed > 0:
                print(f"✓ Entity resolution: merged {merges_performed} duplicate entities")
        
        except Exception as e:
            # If entity resolution fails, continue without it
            pass
    
    def _create_parent_child_relationships(self, documents):
        """
        Create explicit (Parent)-[:CONTAINS]->(Child) relationships in Neo4j.
        
        This allows the agent to:
        1. Find precise facts in child nodes
        2. Traverse up to parent for full context
        """
        from llama_index.core.node_parser import SentenceSplitter
        
        parent_splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=200)
        child_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
        
        for doc in documents:
            parent_nodes = parent_splitter.get_nodes_from_documents([doc])
            
            for parent_idx, parent in enumerate(parent_nodes):
                parent_id = f"{doc.metadata.get('file_name', 'doc')}:parent:{parent_idx}"
                
                # Create parent node in Neo4j
                query = """
                MERGE (p:ChunkNode {id: $parent_id})
                SET p.text = $text,
                    p.type = 'parent',
                    p.file_name = $file_name
                """
                self.graph_store._driver.execute_query(
                    query,
                    parent_id=parent_id,
                    text=parent.text[:500],  # Store first 500 chars as preview
                    file_name=doc.metadata.get('file_name', 'unknown')
                )
                
                # Create child nodes
                child_nodes = child_splitter.get_nodes_from_documents([type(doc)(text=parent.text)])
                
                for child_idx, child in enumerate(child_nodes):
                    child_id = f"{parent_id}:child:{child_idx}"
                    
                    # Create child and CONTAINS relationship
                    query = """
                    MERGE (c:ChunkNode {id: $child_id})
                    SET c.text = $text,
                        c.type = 'child',
                        c.file_name = $file_name
                    WITH c
                    MATCH (p:ChunkNode {id: $parent_id})
                    MERGE (p)-[:CONTAINS]->(c)
                    """
                    self.graph_store._driver.execute_query(
                        query,
                        child_id=child_id,
                        parent_id=parent_id,
                        text=child.text,
                        file_name=doc.metadata.get('file_name', 'unknown')
                    )
    
    def search(self, query, top_k=5):
        """Search the knowledge graph"""
        if not self.index:
            self.index = KnowledgeGraphIndex([], storage_context=self.storage_context)
        
        retriever = self.index.as_retriever(
            retriever_mode="keyword",  # Search by keywords in the graph
            similarity_top_k=top_k
        )
        return retriever.retrieve(query)

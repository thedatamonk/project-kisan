import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
import faiss
from schemes import initialize_mock_schemes, SchemeDocument


load_dotenv()

class GovernmentSchemesRAG:
    """
    RAG system for Indian agricultural government schemes.
    Uses FAISS for vector storage and OpenAI for embeddings + generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG system
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-4o-mini"
        self.embedding_dimension = 1536  # Dimension for text-embedding-3-small
        
        # Initialize empty vector store
        self.index = None
        self.document_embeddings = []
        
        # Load mock schemes
        self.documents = initialize_mock_schemes()
        print(f"✅ Loaded {len(self.documents)} government schemes")
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def build_index(self):
        """
        Build FAISS index from documents.
        This creates vector embeddings for all schemes and stores them.
        """
        print("🔨 Building vector index...")
        
        # Generate embeddings for all documents
        embeddings = []
        for i, doc in enumerate(self.documents):
            print(f"  Embedding document {i+1}/{len(self.documents)}: {doc.title}")
            text = doc.to_text()
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        
        self.document_embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(self.document_embeddings)
        
        print(f"✅ Index built with {self.index.ntotal} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search for relevant schemes based on query
        
        Args:
            query: User's question or search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant scheme documents with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self._get_embedding(query).reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve documents
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            doc = self.documents[idx]
            results.append({
                "rank": i + 1,
                "score": float(1 / (1 + distance)),  # Convert distance to similarity score
                "document": doc
            })
        
        return results
    
    def query(
        self,
        user_query: str,
        top_k: int = 3
    ) -> str:
        """
        Answer user query using RAG approach
        
        Args:
            user_query: User's question about government schemes
            top_k: Number of schemes to retrieve
            
        Returns:
            Generated response with scheme information
        """
        # Search for relevant schemes
        search_results = self.search(user_query, top_k=top_k)
        
        # Prepare context from retrieved documents
        context = "RELEVANT GOVERNMENT SCHEMES:\n\n"
        for result in search_results:
            doc = result["document"]
            context += f"--- SCHEME {result['rank']} ---\n"
            context += doc.to_text()
            context += "\n\n"
        
        # Generate response using GPT
        system_prompt = """You are a helpful agricultural advisor helping Indian farmers understand government schemes.

INSTRUCTIONS:
1. Answer the farmer's question based ONLY on the provided scheme information
2. Explain in simple, easy-to-understand language
3. Be specific about eligibility, benefits, and application process
4. Always mention which scheme(s) you're referring to
5. Include website links and contact information
6. If the schemes don't answer the question, say so clearly

Keep your response practical and actionable."""

        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"FARMER'S QUESTION: {user_query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def get_scheme_by_id(self, scheme_id: str) -> Optional[SchemeDocument]:
        """Get specific scheme by ID"""
        for doc in self.documents:
            if doc.id == scheme_id:
                return doc
        return None
    
    def list_all_schemes(self) -> List[str]:
        """List all available schemes"""
        return [f"{doc.id}: {doc.title}" for doc in self.documents]
    
    def save_index(self, filepath: str = "schemes_index.faiss"):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        faiss.write_index(self.index, filepath)
        print(f"💾 Index saved to {filepath}")
    
    def load_index(self, filepath: str = "schemes_index.faiss"):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)
        print(f"📂 Index loaded from {filepath}")


# Example usage and testing functions
def example_basic_search():
    """Example: Basic scheme search"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Scheme Search")
    print("=" * 60)
    
    rag = GovernmentSchemesRAG()
    rag.build_index()
    
    queries = [
        "subsidy for drip irrigation",
        "crop insurance schemes",
        "help to buy tractor"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = rag.search(query, top_k=2)
        
        for result in results:
            print(f"\n  Rank {result['rank']}: {result['document'].title}")
            print(f"  Score: {result['score']:.3f}")
            print(f"  Category: {result['document'].category}")
    print()


def example_query_with_generation():
    """Example: Query with response generation"""
    print("=" * 60)
    print("EXAMPLE 2: Query with Response Generation")
    print("=" * 60)
    
    rag = GovernmentSchemesRAG()
    rag.build_index()
    
    user_query = "I want to start organic farming. What government support is available?"
    
    print(f"\n❓ Question: {user_query}\n")
    response = rag.query(user_query)
    print(f"💬 Response:\n{response}")
    print()


if __name__ == "__main__":
    print("\n🏛️ Government Schemes RAG Navigator\n")
    
    print("⚠️ Note: Set OPENAI_API_KEY environment variable before running\n")
    
    # Run examples (uncomment to test with real API)
    # example_basic_search()
    example_query_with_generation()
    # example_agent_integration()
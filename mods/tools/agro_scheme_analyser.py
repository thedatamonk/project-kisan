"""
This is where we're gonna ingest all government scheme documents
chunk them, create embeddings and store them in a vector database weaviate
"""

import os
from typing import List

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from openai import OpenAI

from mods.schemes import SchemeDocument, initialize_mock_schemes

load_dotenv()


class AgroSchemeAnalyserTool:
    def __init__(self):
        try:

            self.db_client = weaviate.connect_to_local(host="localhost", port=8080)
            self.embedding_model = "text-embedding-3-small"
            self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            if self.db_client.is_ready():
                print ("Connected to weaviate client at localhost:8080")

        except Exception as e:
            print ("Error initialising GovtSchemeDB:", e)
            raise e
        self.schema_name = "SchemeDocument"
        self._create_schema()

    def _create_schema(self):
        # Create collection (schema)
        try:
            # Check if collection already exists
            if self.db_client.collections.exists(self.schema_name):
                print(f"Collection '{self.schema_name}' already exists. Skipping creation.")
                return
            
            # Create collection only if it doesn't exist
            scheme_collection = self.db_client.collections.create(
                name=self.schema_name,
                description="Government agricultural schemes for farmers",
                vector_config=None,  # We provide embeddings
                properties=[
                    wvc.config.Property(
                        name="scheme_id",
                        data_type=wvc.config.DataType.TEXT,
                        description="Unique scheme identifier"
                    ),
                    wvc.config.Property(
                        name="title",
                        data_type=wvc.config.DataType.TEXT,
                        description="Scheme title"
                    ),
                    wvc.config.Property(
                        name="description",
                        data_type=wvc.config.DataType.TEXT,
                        description="Detailed description"
                    ),
                    wvc.config.Property(
                        name="category",
                        data_type=wvc.config.DataType.TEXT,
                        description="Scheme category"
                    ),
                    wvc.config.Property(
                        name="eligibility",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="Eligibility criteria"
                    ),
                    wvc.config.Property(
                        name="benefits",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="List of benefits"
                    ),
                    wvc.config.Property(
                        name="application_process",
                        data_type=wvc.config.DataType.TEXT,
                        description="How to apply"
                    ),
                    wvc.config.Property(
                        name="required_documents",
                        data_type=wvc.config.DataType.TEXT_ARRAY,
                        description="Required documents"
                    ),
                    wvc.config.Property(
                        name="contact_info",
                        data_type=wvc.config.DataType.TEXT,
                        description="Contact information"
                    ),
                    wvc.config.Property(
                        name="website",
                        data_type=wvc.config.DataType.TEXT,
                        description="Official website"
                    ),
                    wvc.config.Property(
                        name="state",
                        data_type=wvc.config.DataType.TEXT,
                        description="Applicable state"
                    )
                ]
            )
            print("Schema created successfully!\nSchema details:\n")
            print(scheme_collection)
        except Exception as e:
            print("Error creating schema:", e)
            raise e
    
    def get_embedding(self, text):
        """Get embedding from OpenAI"""
        response = self.llm_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
        
    def create_scheme_embedding(self, scheme_doc: SchemeDocument) -> List[float]:
        """Create embedding from all scheme info"""
        text_parts = [
            f"Title: {scheme_doc.title}",
            f"Description: {scheme_doc.description}",
            f"Category: {scheme_doc.category}",
            f"Eligibility: {', '.join(scheme_doc.eligibility)}",
            f"Benefits: {', '.join(scheme_doc.benefits)}",
            f"Application Process: {scheme_doc.application_process}",
            f"State: {scheme_doc.state}"
        ]
        combined_text = "\n".join(text_parts)
        return self.get_embedding(combined_text)
    
    def build_index(self, documents: List[SchemeDocument]):
        collection = self.db_client.collections.get(self.schema_name)
        
        # Prepare data objects
        data_objects = []
        for scheme_doc in documents:
            embedding = self.create_scheme_embedding(scheme_doc)
            
            data_objects.append({
                "properties": {
                    "scheme_id": scheme_doc.id,
                    "title": scheme_doc.title,
                    "description": scheme_doc.description,
                    "category": scheme_doc.category,
                    "eligibility": scheme_doc.eligibility,
                    "benefits": scheme_doc.benefits,
                    "application_process": scheme_doc.application_process,
                    "required_documents": scheme_doc.required_documents,
                    "contact_info": scheme_doc.contact_info,
                    "website": scheme_doc.website,
                    "state": scheme_doc.state
                },
                "vector": embedding
            })
        
        # Batch insert
        with collection.batch.dynamic() as batch:
            for obj in data_objects:
                batch.add_object(
                    properties=obj["properties"],
                    vector=obj["vector"]
                )
        
        print(f"Added {len(documents)} schemes to weaviate collection.")

    def close(self):
        self.db_client.close()

    def rewrite_query(self, query: str) -> List[str]:
        """Expand vague user query into better search queries"""
        try:
            prompt = f"""You are helping a farmer search for government agricultural schemes in India.
            
    The user asked: "{query}"

    Generate 2-3 focused search queries that would help find relevant schemes. Consider:
    - Specific scheme categories (loans, subsidies, insurance, etc.)
    - Key benefits the user is looking for
    - Related agricultural topics

    Return only the search queries, one per line."""

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            expanded_queries = response.choices[0].message.content.strip().split('\n')
            return [q.strip('- ').strip().strip('"') for q in expanded_queries if q.strip()]

        except Exception as e:
            print(f"Error rewriting query:{e}")
            print ("Falling back to original query.")
            return [query]  # Fallback to original query
        

    def query_db(self, query: str, top_k: int):
        collection = self.db_client.collections.get(self.schema_name)
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
    
        # Extract results
        results = []
        for obj in response.objects:
            results.append({
                "scheme_id": obj.properties["scheme_id"],
                "title": obj.properties["title"],
                "description": obj.properties["description"],
                "category": obj.properties["category"],
                "eligibility": obj.properties["eligibility"],
                "benefits": obj.properties["benefits"],
                "application_process": obj.properties["application_process"],
                "required_documents": obj.properties["required_documents"],
                "contact_info": obj.properties["contact_info"],
                "website": obj.properties["website"],
                "state": obj.properties["state"],
                "distance": obj.metadata.distance
            })
        
        return results


    def search_schemes(self, query, top_k=2):
        """Search schemes with vector similarity"""

        print (f"Attempting to rewrite query: {query}")
        search_queries = self.rewrite_query(query)

        # Search with all expanded queries
        all_results = []
        for subquery in search_queries:
            results = self.query_db(subquery, top_k=3)
            all_results.extend(results)
        
        # Deduplicate and return top results
        unique_results = {r['scheme_id']: r for r in all_results}.values()

        # Return results sorted by distance
        return sorted(unique_results, key=lambda x: x['distance'])[:top_k]



if __name__ == "__main__":
    db = AgroSchemeAnalyserTool()
    # mock_schemes = initialize_mock_schemes()
    # db.build_index(mock_schemes)
    # db.close()
    # print(f"✅ Loaded {len(mock_schemes)} government schemes")

    # Usage
    results = db.search_schemes("I wanna get a loan to buy agricultural equipment", top_k=5)
    for r in results:
        print(f"{r['title']} - Distance: {r['distance']:.3f} - Description: {r['description'][:100]}...")

    db.close()


    





import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name="sinhala_accounting_docs"):
        self.collection_name = collection_name
        self.vectorstore_path = "vectorstore"
        
        # Create vectorstore directory if it doesn't exist
        if not os.path.exists(self.vectorstore_path):
            os.makedirs(self.vectorstore_path)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.vectorstore_path)
        
        # Initialize sentence transformer for embeddings
        # Using multilingual model that supports Sinhala
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = None
            logger.info(f"Collection {self.collection_name} will be created when documents are added")
    
    def collection_exists(self) -> bool:
        """Check if collection exists and has documents"""
        try:
            if self.collection is None:
                return False
            count = self.collection.count()
            return count > 0
        except:
            return False
    
    def create_collection(self, documents: List[Dict]):
        """Create collection and add documents"""
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
            # Prepare data for insertion
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                texts.append(doc['text'])
                metadatas.append({
                    'source': doc['source'],
                    'chunk_id': doc['chunk_id'],
                    'document_type': doc['document_type']
                })
                ids.append(f"doc_{i}")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            logger.info(f"Successfully added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            if self.collection is None:
                logger.warning("Collection not initialized")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            logger.info(f"Found {len(formatted_results)} relevant documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def add_document(self, text: str, metadata: Dict):
        """Add a single document to the collection"""
        try:
            if self.collection is None:
                logger.error("Collection not initialized")
                return False
            
            # Generate embedding
            embedding = self.embedding_model.encode([text]).tolist()[0]
            
            # Generate unique ID
            doc_id = f"doc_{self.collection.count()}"
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            if self.collection is None:
                return {'count': 0, 'status': 'not_initialized'}
            
            count = self.collection.count()
            
            # Get sample of documents to analyze sources
            sample_results = self.collection.query(
                query_embeddings=[self.embedding_model.encode(["ගිණුම්කරණ"]).tolist()[0]],
                n_results=min(10, count),
                include=['metadatas']
            )
            
            sources = set()
            doc_types = set()
            
            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                for metadata in sample_results['metadatas'][0]:
                    sources.add(metadata.get('source', 'unknown'))
                    doc_types.add(metadata.get('document_type', 'unknown'))
            
            return {
                'count': count,
                'sources': list(sources),
                'document_types': list(doc_types),
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'count': 0, 'status': 'error', 'error': str(e)}
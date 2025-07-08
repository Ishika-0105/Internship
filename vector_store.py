import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from config import Config
import streamlit as st
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from collections import defaultdict

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.documents = []
        self.embeddings = None
        self.document_metadata = []  # Store additional metadata
        self.dimension = None
        
        # HNSW parameters
        self.hnsw_m = 16  # Number of bi-directional links created for each new element during construction
        self.hnsw_ef_construction = 200  # Size of the dynamic candidate list during construction
        self.hnsw_ef_search = 100  # Size of the dynamic candidate list during search
        
    def _initialize_hnsw_index(self, dimension: int):
        """Initialize HNSW index with optimal parameters"""
        self.dimension = dimension
        # Create HNSW index with inner product (cosine similarity when vectors are normalized)
        self.index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
        
        # Set construction parameters
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        
        # Set search parameters
        self.index.hnsw.efSearch = self.hnsw_ef_search
        
        st.info(f"🔧 Initialized HNSW index with dimension {dimension}, M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction}")
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store with HNSW indexing"""
        try:
            if not documents:
                st.warning("⚠️ No documents provided to add to vector store.")
                return False
            
            # Extract text content and metadata
            texts = []
            valid_documents = []
            metadata = []
            
            for doc in documents:
                if hasattr(doc, 'page_content') and doc.page_content:
                    texts.append(doc.page_content)
                    valid_documents.append(doc)
                    # Extract metadata
                    doc_metadata = {
                        'source': getattr(doc, 'metadata', {}).get('source', 'unknown'),
                        'page': getattr(doc, 'metadata', {}).get('page', 0),
                        'chunk_id': len(self.documents) + len(metadata),
                        'length': len(doc.page_content),
                        'keywords': self._extract_keywords(doc.page_content)
                    }
                    metadata.append(doc_metadata)
                else:
                    st.warning(f"⚠️ Skipping document with no content: {doc}")
                    continue
            
            if not texts:
                st.error("❌ No valid text content found in documents.")
                return False
            
            # Generate embeddings
            with st.spinner("🔄 Generating embeddings..."):
                embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            
            # Initialize or update indexes
            if self.index is None:
                dimension = embeddings.shape[1]
                self._initialize_hnsw_index(dimension)
                self.documents = valid_documents
                self.embeddings = embeddings
                self.document_metadata = metadata
            else:
                # Add to existing data
                self.documents.extend(valid_documents)
                self.document_metadata.extend(metadata)
                if self.embeddings is not None:
                    self.embeddings = np.vstack([self.embeddings, embeddings])
                else:
                    self.embeddings = embeddings
            
            # Add embeddings to HNSW index
            self.index.add(embeddings.astype('float32'))
            
            st.success(f"✅ Added {len(valid_documents)} documents to HNSW vector store. Total documents: {len(self.documents)}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding documents to vector store: {str(e)}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text"""
        # Simple keyword extraction - you can enhance this with NLP libraries
        words = text.lower().split()
        # Filter out common stop words and short words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        keywords = [word.strip('.,!?;:"()[]') for word in words if len(word) > 3 and word not in stop_words]
        return list(set(keywords[:10]))  # Return top 10 unique keywords
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform similarity search using HNSW"""
        try:
            if self.index is None or len(self.documents) == 0:
                st.warning("⚠️ Vector store is empty. Please add documents first.")
                return []
            
            # Generate query embedding (normalized for cosine similarity)
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search using HNSW index
            # Since we're using normalized embeddings, inner product gives us cosine similarity
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and idx != -1:  # -1 indicates no result found
                    doc = self.documents[idx]
                    # Score is already cosine similarity due to normalized embeddings
                    similarity = float(score)
                    results.append((doc, similarity))
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error during similarity search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """Hybrid search combining semantic and keyword matching"""
        try:
            if self.index is None or len(self.documents) == 0:
                st.warning("⚠️ Vector store is empty. Please add documents first.")
                return []
            
            # Semantic search using HNSW
            semantic_results = self.similarity_search(query, k * 2)
            
            # Keyword matching
            query_words = set(query.lower().split())
            keyword_scores = []
            
            for i, doc in enumerate(self.documents):
                if i < len(self.document_metadata):
                    doc_keywords = set(self.document_metadata[i].get('keywords', []))
                    keyword_overlap = len(query_words.intersection(doc_keywords))
                    keyword_score = keyword_overlap / max(len(query_words), 1)
                    keyword_scores.append((doc, keyword_score))
            
            # Sort by keyword score
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            keyword_results = keyword_scores[:k * 2]
            
            # Combine results
            combined_scores = {}
            
            # Add semantic scores
            for doc, score in semantic_results:
                doc_id = doc.page_content[:100]  # Use first 100 chars as unique identifier
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic': score,
                    'keyword': 0.0
                }
            
            # Add keyword scores
            for doc, score in keyword_results:
                doc_id = doc.page_content[:100]
                if doc_id in combined_scores:
                    combined_scores[doc_id]['keyword'] = score
                else:
                    combined_scores[doc_id] = {
                        'doc': doc,
                        'semantic': 0.0,
                        'keyword': score
                    }
            
            # Calculate final scores
            final_results = []
            for doc_id, scores in combined_scores.items():
                final_score = alpha * scores['semantic'] + (1 - alpha) * scores['keyword']
                final_results.append((scores['doc'], final_score))
            
            # Sort and return top k
            final_results.sort(key=lambda x: x[1], reverse=True)
            return final_results[:k]
            
        except Exception as e:
            st.error(f"❌ Error during hybrid search: {str(e)}")
            return self.similarity_search(query, k)  # Fallback to regular search
    
    def get_relevant_documents(self, query: str, k: int = 3, use_hybrid: bool = True) -> List[Document]:
        """Get relevant documents for a query with improved filtering"""
        try:
            if use_hybrid:
                search_results = self.hybrid_search(query, k * 2)
            else:
                search_results = self.similarity_search(query, k * 2)
            
            # Apply dynamic threshold based on score distribution
            if search_results:
                scores = [score for _, score in search_results]
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                threshold = max(0.3, mean_score - 0.5 * std_score)  # Dynamic threshold (higher for cosine similarity)
                
                relevant_docs = [doc for doc, score in search_results if score > threshold][:k]
                return relevant_docs
            
            return []
            
        except Exception as e:
            st.error(f"❌ Error getting relevant documents: {str(e)}")
            return []
    
    def update_hnsw_parameters(self, ef_search: int = None, ef_construction: int = None):
        """Update HNSW search parameters for performance tuning"""
        try:
            if self.index is None:
                st.warning("⚠️ No index initialized yet.")
                return
            
            if ef_search is not None:
                self.hnsw_ef_search = ef_search
                self.index.hnsw.efSearch = ef_search
                st.info(f"🔧 Updated HNSW efSearch to {ef_search}")
            
            if ef_construction is not None:
                self.hnsw_ef_construction = ef_construction
                # Note: efConstruction only affects newly added vectors
                st.info(f"🔧 Updated HNSW efConstruction to {ef_construction} (applies to new vectors)")
                
        except Exception as e:
            st.error(f"❌ Error updating HNSW parameters: {str(e)}")
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.documents = []
        self.embeddings = None
        self.document_metadata = []
        self.dimension = None
        st.success("🧹 Vector store cleared successfully.")
    
    def get_stats(self):
        """Get vector store statistics"""
        stats = {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index is not None else 0,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else "None",
            "dimension": self.dimension,
            "index_type": "HNSW"
        }
        
        if self.index is not None:
            stats.update({
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_construction": self.hnsw_ef_construction,
                "hnsw_ef_search": self.hnsw_ef_search,
                "max_level": getattr(self.index.hnsw, 'max_level', 'unknown')
            })
        
        return stats
    
    def get_document_sources(self) -> Dict[str, int]:
        """Get count of documents by source"""
        sources = {}
        for metadata in self.document_metadata:
            source = metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def save_index(self, filepath: str):
        """Save the HNSW index and associated data"""
        try:
            if self.index is None:
                st.warning("⚠️ No index to save.")
                return False
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save other data
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'document_metadata': self.document_metadata,
                'dimension': self.dimension,
                'hnsw_m': self.hnsw_m,
                'hnsw_ef_construction': self.hnsw_ef_construction,
                'hnsw_ef_search': self.hnsw_ef_search
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            st.success(f"✅ Index saved to {filepath}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving index: {str(e)}")
            return False
    
    def load_index(self, filepath: str):
        """Load the HNSW index and associated data"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load other data
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.document_metadata = data['document_metadata']
            self.dimension = data['dimension']
            self.hnsw_m = data.get('hnsw_m', 16)
            self.hnsw_ef_construction = data.get('hnsw_ef_construction', 200)
            self.hnsw_ef_search = data.get('hnsw_ef_search', 100)
            
            # Update search parameters
            self.index.hnsw.efSearch = self.hnsw_ef_search
            
            st.success(f"✅ Index loaded from {filepath}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading index: {str(e)}")
            return False

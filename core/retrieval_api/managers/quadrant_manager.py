import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import os
logger = logging.getLogger(__name__)

class QuadrantManager:
    """Manages vector storage and search using Quadrant-based indexing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Quadrant manager.
        
        Args:
            config: Configuration dictionary containing storage settings
        """
        self._config = config
        self._dimension = config.get("EMBEDDING_DIMENSION", 768)  # Default for most models
        self._quadrants: Dict[str, List[Dict[str, Any]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
    def add_vector(self, vector: List[float], doc_id: str, metadata: Dict[str, Any]) -> None:
        """Add a vector to the appropriate quadrant.
        
        Args:
            vector: The vector to store
            doc_id: Unique identifier for the document
            metadata: Additional metadata for the document
        """
        try:
            # Convert vector to numpy array
            vec = np.array(vector)
            
            # Determine quadrant based on vector values
            quadrant_id = self._get_quadrant_id(vec)
            
            # Initialize quadrant if it doesn't exist
            if quadrant_id not in self._quadrants:
                self._quadrants[quadrant_id] = []
                
            # Store vector and metadata
            self._quadrants[quadrant_id].append({
                "doc_id": doc_id,
                "vector": vec.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Store metadata separately
            self._metadata[doc_id] = metadata
            
            logger.info(f"Added vector to quadrant {quadrant_id} for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error adding vector to quadrant: {e}")
            raise
            
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using quadrant-based search.
        
        Args:
            query_vector: The query vector to search with
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            query_vec = np.array(query_vector)
            
            # Get relevant quadrants based on query vector
            relevant_quadrants = self._get_relevant_quadrants(query_vec)
            
            # Search within relevant quadrants
            results = []
            for quadrant_id in relevant_quadrants:
                if quadrant_id in self._quadrants:
                    for item in self._quadrants[quadrant_id]:
                        score = self._calculate_similarity(query_vec, np.array(item["vector"]))
                        results.append({
                            "doc_id": item["doc_id"],
                            "score": score,
                            "metadata": self._metadata.get(item["doc_id"], {})
                        })
            
            # Sort results by score and take top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise
            
    def save_to_disk(self, base_path: str) -> None:
        """Save quadrant data to disk.
        
        Args:
            base_path: Base directory to save data
        """
        try:
            save_path = Path(base_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save quadrants
            quadrants_file = save_path / "quadrants.json"
            with open(quadrants_file, 'w') as f:
                json.dump(self._quadrants, f)
                
            # Save metadata
            metadata_file = save_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self._metadata, f)
                
            logger.info(f"Saved quadrant data to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving quadrant data: {e}")
            raise
            
    def load_from_disk(self, base_path: str) -> None:
        """Load quadrant data from disk.
        
        Args:
            base_path: Base directory containing saved data
        """
        try:
            load_path = Path(base_path)
            
            # Load quadrants
            quadrants_file = load_path / "quadrants.json"
            if quadrants_file.exists():
                with open(quadrants_file, 'r') as f:
                    self._quadrants = json.load(f)
                    
            # Load metadata
            metadata_file = load_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                    
            logger.info(f"Loaded quadrant data from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading quadrant data: {e}")
            raise
            
    def _get_quadrant_id(self, vector: np.ndarray) -> str:
        """Determine which quadrant a vector belongs to.
        
        Args:
            vector: The vector to classify
            
        Returns:
            Quadrant identifier
        """
        # Simple quadrant classification based on vector values
        # This can be made more sophisticated based on your needs
        signs = np.sign(vector)
        return "".join(map(str, (signs > 0).astype(int)))
        
    def _get_relevant_quadrants(self, query_vector: np.ndarray) -> List[str]:
        """Get relevant quadrants for a query vector.
        
        Args:
            query_vector: The query vector
            
        Returns:
            List of relevant quadrant IDs
        """
        # Get the query vector's quadrant
        query_quadrant = self._get_quadrant_id(query_vector)
        
        # For now, return the exact quadrant and its neighbors
        # This can be made more sophisticated based on your needs
        return [query_quadrant]
        
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) 
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class TextEmbeddingService:
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the text embedding service.
        Args:
            model_name (str, optional): The SentenceTransformer model name. 
                                       Defaults to "all-MiniLM-L6-v2" or SBERT_MODEL env var.
            device (str, optional): Device to run the model on ('cpu', 'cuda'). 
                                    Defaults to 'cuda' if available, else 'cpu'.
        """
        if device:
            self.device = device
        else:
            self.device = "cuda" if os.getenv("CUDA_IS_AVAILABLE", "false").lower() == "true" and __import__('torch').cuda.is_available() else "cpu" # Simplified check for torch
            
        self.model_name = model_name or os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")
        
        try:
            logger.info(f"Loading SentenceTransformer model: {self.model_name} on device: {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"SentenceTransformer model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {self.model_name}: {e}")
            self.model = None # Ensure model is None if loading fails
            raise  # Re-raise exception to signal failure

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embeds a single string of text.
        Args:
            text (str): The text to embed.
        Returns:
            Optional[np.ndarray]: The embedding as a NumPy array, or None if embedding fails or model not loaded.
        """
        if self.model is None:
            logger.error("Text embedding model not loaded. Cannot embed text.")
            return None
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input for embedding. Please provide a non-empty string.")
            return None
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            logger.debug(f"Successfully embedded text: '{text[:50]}...'")
            return embedding
        except Exception as e:
            logger.error(f"Error during text embedding for '{text[:50]}...': {e}")
            return None

    def embed_texts(self, texts: list[str]) -> Optional[list[np.ndarray]]:
        """
        Embeds a list of text strings.
        Args:
            texts (list[str]): A list of texts to embed.
        Returns:
            Optional[list[np.ndarray]]: A list of embeddings as NumPy arrays, or None if embedding fails.
        """
        if self.model is None:
            logger.error("Text embedding model not loaded. Cannot embed texts.")
            return None
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            logger.warning("Invalid text list input for embedding. Please provide a list of non-empty strings.")
            return None
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Successfully embedded {len(texts)} texts.")
            return [emb for emb in embeddings] # Return as a list of arrays
        except Exception as e:
            logger.error(f"Error during batch text embedding: {e}")
            return None

if __name__ == '__main__':
    # Example Usage (requires torch to be installed for SentenceTransformer)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running TextEmbeddingService example...")
    
    # This example assumes CUDA might not be available or needed for a simple test
    # Forcing CPU for this example to avoid issues if torch+cuda is not set up
    os.environ["CUDA_IS_AVAILABLE"] = "false" # Mock env var for testing

    try:
        text_embedder = TextEmbeddingService(device='cpu') # Explicitly use CPU for example
        
        if text_embedder.model:
            example_text = "This is a test sentence for the text embedder."
            embedding = text_embedder.embed_text(example_text)
            if embedding is not None:
                logger.info(f"Embedding for '{example_text}': {embedding[:5]}... (shape: {embedding.shape})")

            example_texts = [
                "Another example sentence.",
                "Exploring text embeddings with SentenceTransformers."
            ]
            embeddings_list = text_embedder.embed_texts(example_texts)
            if embeddings_list is not None:
                for i, emb in enumerate(embeddings_list):
                    logger.info(f"Embedding for '{example_texts[i]}': {emb[:5]}... (shape: {emb.shape})")
        else:
            logger.error("Failed to initialize TextEmbeddingService for example.")
            
    except Exception as e:
        logger.error(f"Error in TextEmbeddingService example: {e}") 
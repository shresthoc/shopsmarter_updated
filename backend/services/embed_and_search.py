"""
Embedding and Search Service for ShopSmarter
Handles image embedding generation and similarity search using CLIP and FAISS.
"""
import os
import time
import torch
import numpy as np
from PIL import Image
import faiss
from typing import List, Dict, Optional, Tuple
import requests
from io import BytesIO
import hashlib
import logging
import json
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(
        self,
        cache_dir: str = None,
        index_path: str = None
    ):
        """
        Initialize the embedding service with CLIP model and FAISS index.
        Args:
            cache_dir: Directory to cache embeddings (ENV: EMBEDDING_CACHE_DIR or default "cache/embeddings")
            index_path: Base path for saving/loading index (ENV: FAISS_INDEX_PATH or default "cache/faiss_index")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load HuggingFace CLIP model
        self.clip_model_name = "openai/clip-vit-base-patch32"
        logger.info(
            f"Loading HF CLIP model {self.clip_model_name} on device={self.device}"
        )
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_model.to(self.device)
            self.clip_model.eval()
        except Exception as e:
            logger.error(f"Failed to load HF CLIP model: {str(e)}")
            raise

        # Load BLIP model for captioning
        self.blip_model_name = "Salesforce/blip-image-captioning-base"
        logger.info(
            f"Loading BLIP model {self.blip_model_name} on device={self.device}"
        )
        try:
            self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_name)
            self.blip_model.to(self.device)
            self.blip_model.eval()
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {str(e)}")
            # Not raising here, captioning can be optional if model load fails
            self.blip_model = None
            self.blip_processor = None

        # Determine embedding dimension dynamically from CLIP model
        try:
            # Create a dummy image input
            dummy_image = Image.new('RGB', (224, 224))
            inputs = self.clip_processor(images=dummy_image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                dim = self.clip_model.get_image_features(**inputs).shape[-1]
            self.embedding_dim = int(dim)
            logger.info(f"Determined CLIP embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {str(e)}")
            raise

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_path = index_path or os.getenv("FAISS_INDEX_PATH", "cache/faiss_index")

        # Load candidate tag list
        cfg_path = os.getenv("TAGS_JSON")
        if not cfg_path:
            logger.warning("TAGS_JSON environment variable not set. Tag generation will be disabled.")
            self.candidates: List[str] = []
        else:
            try:
                with open(cfg_path, 'r') as f:
                    self.candidates: List[str] = json.load(f)
                logger.info(f"Loaded {len(self.candidates)} candidate tags")
            except Exception as e:
                logger.error(f"Failed to load tags from {cfg_path}: {e}")
                self.candidates: List[str] = []

        # Cache for embeddings
        self.cache_dir = cache_dir or os.getenv("EMBEDDING_CACHE_DIR", "cache/embeddings")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Product mapping: index position -> product dict
        self.product_map: Dict[int, Dict] = {}

    def get_features_for_image(self, image_path_or_data) -> Optional[Tuple[Optional[Image.Image], Optional[Dict[str, any]]]]:
        """
        Public method to get all features (tags, caption) for an image from a path, URL, or data.
        Returns the loaded PIL image and the features dict.
        """
        image = self._load_image(image_path_or_data)
        if image is None:
            logger.warning("Could not load image, cannot get features.")
            return None, None
        
        tags = self.get_image_tags(image)
        caption = self.generate_caption(image)

        return image, {"tags": tags, "caption": caption}

    def generate_caption(self, image: Image.Image) -> Optional[str]:
        """Generate a caption for an image using BLIP."""
        if self.blip_model is None or self.blip_processor is None or image is None:
            logger.warning("BLIP model not loaded or image is None, skipping caption generation.")
            return None
        try:
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            start_time = time.time()
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=77) # max_length typical for CLIP text
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            logger.debug(f"Generated BLIP caption in {time.time() - start_time:.3f}s: {caption}")
            return caption
        except Exception as e:
            logger.error(f"Failed to generate BLIP caption: {str(e)}")
            return None

    def _load_image(self, image_path_or_data, retries=3, delay=1) -> Optional[Image.Image]:
        """
        Loads an image from a path, URL, or bytes data, with retries.
        This is a more robust version to handle different input types.
        """
        for attempt in range(1, retries + 1):
            try:
                image = None
                if isinstance(image_path_or_data, bytes):
                    # Handle in-memory bytes
                    logger.debug(f"Loading image from bytes (attempt {attempt})")
                    image = Image.open(io.BytesIO(image_path_or_data))
                elif isinstance(image_path_or_data, str) and image_path_or_data.startswith(('http://', 'https://')):
                    # Handle URL
                    logger.debug(f"Fetching image URL (attempt {attempt}): {image_path_or_data}")
                    response = requests.get(image_path_or_data, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                elif isinstance(image_path_or_data, str):
                    # Handle file path
                    logger.debug(f"Loading local image from path (attempt {attempt}): {image_path_or_data}")
                    image = Image.open(image_path_or_data)
                else:
                    logger.warning(f"Unsupported image data type: {type(image_path_or_data)}")
                    return None

                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return image

            except Exception as e:
                logger.error(f"Error loading image (attempt {attempt}): {e}")
                if attempt < retries:
                    time.sleep(delay * attempt)
        
        logger.error(f"Failed to load image after {retries} attempts.")
        # When logging the failed data, log its type and a snippet to avoid huge log entries
        if isinstance(image_path_or_data, bytes):
             logger.error(f"Failed data (bytes, snippet): {image_path_or_data[:100]}...")
        else:
             logger.error(f"Failed data (str): {image_path_or_data}")
        return None

    def get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for an image using HuggingFace Transformers.
        """
        if image is None:
            return None

        try:
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move inputs to device
            
            start = time.time()
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy()
            logger.debug(f"Generated HF CLIP embedding in {time.time() - start:.3f}s")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate HF CLIP embedding: {str(e)}")
            return None
    
    def get_image_tags(self, image: Image.Image, top_k: int = 5) -> List[str]:
        """
        Zero-shot CLIP tag matching against self.candidates using HuggingFace Transformers.
        """
        if not image or not self.candidates:
            logger.warning("Image is None or no candidate tags loaded, skipping tag generation.")
            return []

        try:
            # Embed all candidate texts
            # Process text inputs
            text_inputs = self.clip_processor(text=self.candidates, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                text_feats = self.clip_model.get_text_features(**text_inputs)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)

            # Embed the image
            img_emb = self.get_image_embedding(image)  # numpy array
            if img_emb is None:
                return []
            img_tensor = torch.from_numpy(img_emb).to(self.device) # Already normalized

            # Cosine-similarity
            sims = (img_tensor @ text_feats.T).cpu().numpy()[0]

            # Get top_k indices. If less than top_k candidates, take all.
            num_candidates = len(self.candidates)
            actual_top_k = min(top_k, num_candidates)
            
            if actual_top_k == 0: # Handle case with no candidates
                return []

            idxs = sims.argsort()[-actual_top_k:][::-1]
            tags = [self.candidates[i] for i in idxs]
            logger.info(f"Extracted HF CLIP image tags: {tags}")
            return tags
        except Exception as e:
            logger.error(f"Failed to generate HF CLIP tags: {str(e)}")
            return []

    def cache_embedding(self, product_id: str, embedding: np.ndarray):
        """Cache an embedding to disk."""
        if embedding is None:
            return

        try:
            path = os.path.join(self.cache_dir, f"{product_id}.npy")
            np.save(path, embedding)
            logger.debug(f"Cached embedding for {product_id} at {path}")
        except Exception as e:
            logger.error(f"Failed to cache embedding: {str(e)}")

    def load_cached_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Load a cached embedding from disk."""
        try:
            path = os.path.join(self.cache_dir, f"{product_id}.npy")
            if os.path.exists(path):
                logger.debug(f"Loading cached embedding for {product_id}")
                return np.load(path)
        except Exception as e:
            logger.error(f"Failed to load cached embedding: {str(e)}")
        return None

    def add_to_index(self, product: Dict):
        """
        Add a product's image embedding to the FAISS index, caching if needed.
        Generates embedding if not already provided in the product dict.
        Also generates and stores tags and a caption for the product.
        """
        product_id = product.get("id")
        if not product_id:
            logger.warning("Skipping product with no ID.")
            return

        # Check for cached embedding first
        embedding = self.load_cached_embedding(product_id)

        # If not cached, generate from image URL
        if embedding is None:
            image_url = product.get("photo_url")
            if not image_url:
                logger.warning(f"Skipping product {product_id} with no image URL.")
                return
            
            # Load the image
            image = self._load_image(image_url)
            if image is None:
                logger.warning(f"Failed to load image for product {product_id} from {image_url}")
                return
            
            # Generate and cache embedding
            embedding = self.get_image_embedding(image)
            if embedding is not None:
                self.cache_embedding(product_id, embedding)
            else:
                logger.warning(f"Failed to generate embedding for product {product_id}")
                return
        
        # This part ensures we have the image features (tags, caption)
        # even if the embedding was cached previously without them.
        if 'tags' not in product or 'caption' not in product:
            logger.debug(f"Tags/caption missing for {product_id}, generating now...")
            image = self._load_image(product.get("photo_url"))
            if image:
                product['tags'] = self.get_image_tags(image)
                product['caption'] = self.generate_caption(image)
                logger.info(f"Generated features for {product_id}: Tags={product['tags']}, Caption='{product['caption']}'")
            else:
                logger.warning(f"Could not load image to generate features for {product_id}")


        # Add to FAISS index
        current_index = self.index.ntotal
        self.index.add(embedding.astype('float32'))
        self.product_map[current_index] = product
        logger.debug(f"Added product {product_id} to FAISS index at position {current_index}")

    def build_index(self, products: List[Dict]):
        """Build the FAISS index from a list of product data."""
        logger.info(f"Building index for {len(products)} products...")
        start_time = time.time()
        # Clear existing index and map
        self.index.reset()
        self.product_map.clear()

        for product in products:
            self.add_to_index(product)
        
        logger.info(f"Index build complete with {self.index.ntotal} vectors in {time.time() - start_time:.2f} seconds.")

    def find_similar(
        self,
        query_image: Image.Image,
        k: int = 10,
        price_range: Optional[Tuple[float, float]] = None,
        text_query: Optional[str] = None
    ) -> List[Dict]:
        """
        Finds products with images similar to the query image.
        Optionally refines the search with a text query.
        """
        if not query_image:
            logger.error("A query image must be provided to find_similar.")
            return []

        # Get embedding for the query image
        query_embedding = self.get_image_embedding(query_image)
        if query_embedding is None:
            logger.error("Failed to generate embedding for the query image.")
            return []

        # Perform the search on the entire index first
        # We search for more than k initially to have a larger pool for filtering
        search_k = min(self.index.ntotal, max(k * 5, 50)) # Search for more items to filter from
        if search_k == 0:
            logger.warning("FAISS index is empty. Cannot perform search.")
            return []
            
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # --- Filtering Logic ---
        # This section can be expanded with more complex filtering
        
        results = []
        seen_product_ids = set()

        for i in range(indices.shape[1]):
            idx = indices[0][i]
            dist = distances[0][i]
            
            # Retrieve product info
            product = self.product_map.get(idx)
            if not product:
                logger.warning(f"No product found for index {idx}")
                continue

            product_id = product.get('id')
            if product_id in seen_product_ids:
                continue
            
            # Price filtering
            if price_range:
                price_str = product.get("product_price")
                if price_str:
                    try:
                        # Clean price string (e.g., "$19.99" -> 19.99)
                        price = float(str(price_str).replace('$', '').replace(',', '').strip())
                        if not (price_range[0] <= price <= price_range[1]):
                            continue # Skip if outside price range
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse price for product {product_id}: {price_str}")
                        pass # or treat as not matching
            
            # Add similarity score to the product dict
            product['similarity_score'] = 1 - (dist / self.embedding_dim) # Example normalization
            results.append(product)
            if product_id:
                seen_product_ids.add(product_id)

            # Stop if we have enough results
            if len(results) >= k:
                break
        
        logger.info(f"Found {len(results)} similar products after filtering.")
        return results

    def save_index(self):
        """Save FAISS index and product mapping to disk."""
        try:
            idx_path = f"{self.index_path}.index"
            json_path = f"{self.index_path}.json"
            faiss.write_index(self.index, idx_path)
            logger.info(f"FAISS index saved to {idx_path}")
            
            with open(json_path, 'w') as f:
                json.dump(self.product_map, f)
            logger.info(f"Product mapping saved to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")

    def load_index(self):
        """Load FAISS index and product mapping from disk."""
        try:
            idx_path = f"{self.index_path}.index"
            json_path = f"{self.index_path}.json"
            if os.path.exists(idx_path) and os.path.exists(json_path):
                self.index = faiss.read_index(idx_path)
                with open(json_path, 'r') as f:
                    self.product_map = json.load(f)
                logger.info(f"Loaded index and product map from disk: {len(self.product_map)} items")
            else:
                logger.warning(f"Index files not found at {self.index_path}, skipping load")
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.product_map.clear()

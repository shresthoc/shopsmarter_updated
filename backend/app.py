"""
ShopSmarter Flask Backend
Main API server that integrates search, embedding, and refinement services.
"""
import os
import json
import logging
import traceback
from typing import List, Dict, Optional
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

# For BLIP captioning (ensure transformers is installed: pip install transformers)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Get the backend directory path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Vercel Cache Configuration ---
VERCEL_ENV = os.environ.get("VERCEL") == "1"
MODEL_CACHE_ROOT_DIR = "/tmp/model_cache" # Root for all model caches on Vercel

if VERCEL_ENV:
    os.makedirs(MODEL_CACHE_ROOT_DIR, exist_ok=True)
    # Set Hugging Face cache environment variables
    os.environ['HF_HOME'] = os.path.join(MODEL_CACHE_ROOT_DIR, 'huggingface')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(MODEL_CACHE_ROOT_DIR, 'huggingface')
    # Set PyTorch Hub cache environment variable (used by open_clip and others)
    os.environ['TORCH_HOME'] = os.path.join(MODEL_CACHE_ROOT_DIR, 'pytorch')
    # Ensure subdirectories exist
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
    print(f"[Vercel Init] HF_HOME set to: {os.environ['HF_HOME']}")
    print(f"[Vercel Init] TORCH_HOME set to: {os.environ['TORCH_HOME']}")
# --- End Vercel Cache Configuration ---

# Load environment variables before anything else
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logging
    format="%(asctime)s %(levelname)s %(name)s – %(message)s"
)
logger = logging.getLogger(__name__)
logger.info('Environment variables loaded')

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Simplified CORS for API routes
logger.info('Flask app initialized with CORS')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Import services with correct names
from services.embed_and_search import EmbeddingService
from services.refinement_service import RefinementService
# Replace single RapidApiEcomService with specific Amazon and Flipkart RapidAPI services
# from services.rapidapi_ecom import RapidApiEcomService # REMOVED
from services.base_rapidapi_service import BaseRapidApiService
# from services.rapidapi_amazon_service import RapidApiAmazonService # DELETED
# from services.rapidapi_flipkart_service import RapidApiFlipkartService # DELETED

# Initialize services and models
embedding_service: Optional[EmbeddingService] = None
refinement_service: Optional[RefinementService] = None
# rapidapi_service: Optional[RapidApiEcomService] = None # REMOVED
# rapidapi_amazon_service: Optional[RapidApiAmazonService] = None # DELETED
# rapidapi_flipkart_service: Optional[RapidApiFlipkartService] = None # DELETED
product_search_service: Optional[BaseRapidApiService] = None # ADDED: Single service for product search
rapidapi_amazon_service: Optional[BaseRapidApiService] = None # RE-ADDED for Amazon
rapidapi_flipkart_service: Optional[BaseRapidApiService] = None # RE-ADDED for Flipkart
blip_processor: Optional[BlipProcessor] = None
blip_model: Optional[BlipForConditionalGeneration] = None

# --- Global State & Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This will hold the context of the last successful search (for follow-up questions)
last_search_context = {}

# Known common object types for simple parsing
KNOWN_OBJECT_TYPES = {"shirt", "t-shirt", "tshirt", "jeans", "pants", "shoes", "jacket", "dress", "skirt", "shorts"}
COMMON_STOP_WORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'on', 'in', 'at', 'to', 'for', 'with', 'of', 'by', 'and', 'or', 'but', 'it', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'men', "men's", 'womens', "women's", 'lady', "lady's"} # Added some common gendered words that are often not core attributes
COMMON_COLORS = {"red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray", "grey", "beige", "navy", "teal", "maroon", "olive", "silver", "gold"}

# Keywords that indicate the user is referring to the last search result
CONTEXTUAL_REFERENCE_KEYWORDS = {
    # Pronouns and general references
    "it", "them", "they", "those", "these", 
    "another", "other", "others", "one", "ones",
    # Phrases for changing attributes or asking for alternatives
    "in another color", "in a different color", "in another colour",
    "how about", "what about"
    # "show me" and "find me" were removed as they are too ambiguous
    # and cause false positives for new searches.
}

# Keywords for outfit building
COORDINATION_KEYWORDS = {
    # Core phrases
    "go with", "goes with", "match", "matches", "matching", 
    "style with", "pair with", "wear with", "look good with",
    # Common variations
    "go well with", "goes well with", "pairs well with",
    "look great with", "works with", "work with",
    # More explicit commands
    "complete the look", "find a match for", "what to wear with"
}

# Prompts that are too generic to be useful for API search when an image is provided
GENERIC_PROMPTS = {
    "show similar products",
    "find similar items",
    "show me more like this",
    "anything like this",
    "show such similar products",
    "search",
    "find",
    "get products"
}

def parse_new_item_type_from_prompt(prompt_text: str, known_types: set) -> Optional[str]:
    """Tries to find a new explicit item type in a contextual prompt."""
    words = prompt_text.lower().split()
    # Search for known types, prioritizing multi-word types if they exist in known_types
    # For simplicity, just iterates. A more robust parser would be better.
    
    # Check for two-word types first
    for i in range(len(words) - 1):
        two_word_type = f"{words[i]} {words[i+1]}"
        logger.debug(f"Contextual parse (new type): Checking two-word: '{two_word_type}'")
        if two_word_type in known_types:
            if f"last {two_word_type}" not in prompt_text.lower() and \
               f"previous {two_word_type}" not in prompt_text.lower():
                logger.debug(f"Contextual parse: Found new two-word item type: '{two_word_type}'")
                return two_word_type

    # Check for single-word types
    for word in words:
        logger.debug(f"Contextual parse (new type): Checking single-word: '{word}'")
        if word in known_types:
            if f"last {word}" not in prompt_text.lower() and \
               f"previous {word}" not in prompt_text.lower():
                logger.debug(f"Contextual parse: Found new single-word item type: '{word}'")
                return word
    logger.debug(f"Contextual parse: No new item type found in prompt: '{prompt_text}'")
    return None

def extract_key_elements_from_query(query: str) -> Dict[str, any]:
    """
    Extracts primary object type and attributes from a search query string.
    This is a heuristic-based approach.
    """
    if not query or not isinstance(query, str):
        return {"primary_object_type": None, "primary_attributes": []}

    words = query.lower().split()
    # Filter out stop words early, but keep original words for object type matching if they are part of a multi-word type
    # For attribute extraction, we'll use the filtered list.
    potential_attribute_words = [word for word in words if word not in COMMON_STOP_WORDS]
    
    # Attempt to find a known object type by checking from the end of the query (using original words for type matching)
    primary_object_type = None
    # Explicitly check for "t-shirt" variations
    if "t-shirt" in KNOWN_OBJECT_TYPES: # Ensure "t-shirt" is the canonical form we want
        if "t-shirt" in query.lower() or "t - shirt" in query.lower(): # Check both forms
            primary_object_type = "t-shirt" # Standardize to "t-shirt"
            logger.debug(f"extract_key_elements: Found 't-shirt' (or variant). primary_object_type set to: {primary_object_type}")
    
    if not primary_object_type:
        logger.debug(f"extract_key_elements: 't-shirt' not found or not primary (after specific check). Proceeding to general object type search.")
        # Loop backwards through original words to give precedence to terms at the end (often the main noun)
        for i in range(len(words) -1, -1, -1):
            # Check two words phrase first in this loop to catch things like "summer dress" before just "dress"
            if i > 0:
                two_word_candidate = f"{words[i-1]} {words[i]}"
                if two_word_candidate in KNOWN_OBJECT_TYPES:
                    primary_object_type = two_word_candidate
                    logger.debug(f"extract_key_elements: Found two-word type: '{primary_object_type}'")
                    break 
            
            # Check single word
            single_word_candidate = words[i]
            if single_word_candidate in KNOWN_OBJECT_TYPES:
                primary_object_type = single_word_candidate
                logger.debug(f"extract_key_elements: Found single-word type: '{primary_object_type}'")
                break # Found an object type, break from loop
    else:
        logger.debug(f"extract_key_elements: primary_object_type already set to '{primary_object_type}'. Skipping general search.")

    attributes = []
    if primary_object_type:
        object_type_words = set(primary_object_type.split())
        # Use potential_attribute_words for attribute extraction now
        # potential_attribute_words are already filtered from COMMON_STOP_WORDS
        attributes = [word for word in potential_attribute_words if word not in object_type_words]
    else:
        # If no specific object type found, consider all non-stop words as potential attributes/keywords
        attributes = potential_attribute_words

    # Basic cleanup of attributes (e.g. remove very short words, duplicates)
    attributes = sorted(list(set(attr for attr in attributes if len(attr) > 1))) # len > 1 might be too aggressive, consider len > 2 for some cases or specific removal
    
    logger.debug(f"Extracted from query '{query}': Type='{primary_object_type}', Attributes={attributes}")
    return {"primary_object_type": primary_object_type, "primary_attributes": attributes}

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"

try:
    # Set environment variable for tags.json path
    os.environ['TAGS_JSON'] = os.path.join(BACKEND_DIR, 'tags.json')
    logger.debug(f"TAGS_JSON path set to: {os.environ['TAGS_JSON']}")
    
    embedding_service = EmbeddingService()
    logger.debug("EmbeddingService initialized")
    
    # REMOVE OLD RapidApiAmazonService and RapidApiFlipkartService initializations
    # logger.debug("RapidApiAmazonService initialized") # REMOVED
    # logger.debug("RapidApiFlipkartService initialized") # REMOVED

    # ADDED: Initialize separate Amazon and Flipkart services using BaseRapidApiService
    rapidapi_amazon_service = BaseRapidApiService(
        api_key_env_var="RAPIDAPI_KEY",
        host_env_var="RAPIDAPI_AMAZON_HOST",
        endpoint_path_env_var="RAPIDAPI_AMAZON_ENDPOINT_PATH",
        default_endpoint_path="/search", # From your Amazon curl output
        text_query_param_env_var="RAPIDAPI_AMAZON_TEXT_QUERY_PARAM",
        default_text_query_param="query", # From your Amazon curl output
        category_param_env_var="RAPIDAPI_AMAZON_CATEGORY_PARAM", # May be blank/unused for Amazon /search
        category_value_format_env_var="RAPIDAPI_AMAZON_CATEGORY_VALUE_FORMAT",
        service_name="RapidAPI-Amazon"
    )
    logger.debug("RapidApiAmazonService (via Base) initialized")

    rapidapi_flipkart_service = BaseRapidApiService(
        api_key_env_var="RAPIDAPI_KEY",
        host_env_var="RAPIDAPI_FLIPKART_HOST",
        endpoint_path_env_var="RAPIDAPI_FLIPKART_ENDPOINT_PATH",
        default_endpoint_path="/products-by-category", # From your Flipkart curl output
        text_query_param_env_var="RAPIDAPI_FLIPKART_TEXT_QUERY_PARAM", # Configurable, but likely not primary for this endpoint
        default_text_query_param=None, # Explicitly no default text query param for this category-focused endpoint
        category_param_env_var="RAPIDAPI_FLIPKART_CATEGORY_PARAM",
        # default_category_param_name="categoryId", # This should be set by RAPIDAPI_FLIPKART_CATEGORY_PARAM in .env
        category_value_format_env_var="RAPIDAPI_FLIPKART_CATEGORY_VALUE_FORMAT",
        service_name="RapidAPI-Flipkart"
    )
    logger.debug("RapidApiFlipkartService (via Base) initialized")

    refinement_service = RefinementService()
    logger.debug("RefinementService initialized")
    if os.getenv("OPENAI_API_KEY"):
        logger.info("OpenAI API key found. RefinementService may use OpenAI for LLM tasks if implemented to do so.")
    else:
        logger.info("OpenAI API key NOT found. RefinementService will rely on non-OpenAI methods for refinement (e.g., basic heuristics or local models if implemented). Ensure RefinementService handles this gracefully.")

    # Load BLIP model for captioning
    logger.info(f"Loading BLIP model: {BLIP_MODEL_NAME}")
    blip_cache_subdir = None
    if VERCEL_ENV:
        blip_cache_subdir = os.path.join(os.environ['HF_HOME'], 'blip') # Specific subdir for BLIP
        os.makedirs(blip_cache_subdir, exist_ok=True)
        print(f"[Vercel BLIP Load] BLIP model cache directory: {blip_cache_subdir}")

    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, cache_dir=blip_cache_subdir)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME, cache_dir=blip_cache_subdir)
    logger.info("BLIP model loaded successfully.")
    
    logger.info('All services and models initialized successfully')
except ImportError as e:
    logger.error(f'Failed to import required modules: {str(e)}')
    logger.error(traceback.format_exc())
    raise
except Exception as e:
    logger.error(f'Failed to initialize services or models: {str(e)}')
    logger.error(traceback.format_exc())
    raise

# Configure upload settings
UPLOAD_FOLDER = os.path.abspath(os.getenv('UPLOAD_FOLDER', os.path.join(BACKEND_DIR, 'uploads')))
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
logger.debug(f"Upload folder set to: {UPLOAD_FOLDER}")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def merge_and_dedupe_products(products_list1: List[Dict], products_list2: List[Dict]) -> List[Dict]:
    """Merges two lists of product dictionaries and removes duplicates based on product 'id' or (if unavailable) 'photo_url'."""
    if not products_list1:
        return products_list2 or []
    if not products_list2:
        return products_list1 or []

    # Use a dictionary for efficient deduplication
    merged_products: Dict[str, Dict] = {}

    # Combine lists, giving priority to the first list
    combined = products_list1 + products_list2

    for product in combined:
        # Use a tuple of ('id', value) or ('photo', value) as key
        key = None
        if product.get('id'):
            key = ('id', product.get('id'))
        elif product.get('photo_url'):
            key = ('photo', product.get('photo_url'))
        
        # If a key was determined and it's not already in our dict, add it.
        if key and key not in merged_products:
            merged_products[key] = product

    logger.info(f"Merged {len(products_list1)} and {len(products_list2)} product lists into {len(merged_products)} unique products.")
    return list(merged_products.values())

@app.route('/api/query', methods=['POST'])
def query():
    """Handles user queries, both with and without images."""
    global last_search_context # Required to modify the global variable within this function

    try:
        if 'image' not in request.files and 'prompt' not in request.form:
            return jsonify({"error": "No prompt or image provided."}), 400
        
        prompt = request.form.get('prompt', '').strip()
        image_file = request.files.get('image')
        final_products = [] # Initialize for all paths to prevent UnboundLocalError
        is_coordination_query = any(keyword in prompt.lower() for keyword in COORDINATION_KEYWORDS)
        
        # --- 1. Image Processing (if an image is provided) ---
        if image_file:
            if not allowed_file(image_file.filename):
                return jsonify({"error": "Invalid file type. Allowed types are png, jpg, jpeg, gif."}), 400
            
            # Reset context if a new image is uploaded, as it's a new search.
            last_search_context.clear()
            embedding_service.clear_index() # Also clear the visual search index
            
            # Read image data into memory once
            image_data = image_file.read()

            # --- 1a. Feature Extraction from Image ---
            image_object = embedding_service._load_image(image_data)
            if image_object:
                generated_tags = embedding_service.get_image_tags(image_object)
                generated_caption = embedding_service.generate_caption(image_object)
                
                # --- 1b. Determine the initial search term or trigger style builder ---
                initial_search_term = "" 
                
                # Scenario: Image + Coordination Prompt (e.g., "show me jeans that go with this")
                if prompt and is_coordination_query:
                    new_item_type = parse_new_item_type_from_prompt(prompt, KNOWN_OBJECT_TYPES)
                    if new_item_type and refinement_service:
                        # Build a description from the image tags
                        base_item_attributes = [tag for tag in generated_tags if tag not in KNOWN_OBJECT_TYPES]
                        base_item_type = next((tag for tag in generated_tags if tag in KNOWN_OBJECT_TYPES), "item")
                        base_item_desc = f"{' '.join(base_item_attributes)} {base_item_type}".strip()
                        
                        logger.info(f"Triggering outfit builder with image. Base: '{base_item_desc}', New: '{new_item_type}'")

                        suggested_queries = refinement_service.suggest_style_queries(base_item_desc, new_item_type)
                        
                        if suggested_queries and rapidapi_amazon_service:
                            logger.info(f"LLM suggested search queries for outfit: {suggested_queries}")
                            for query_term in suggested_queries:
                                api_products = rapidapi_amazon_service.search(query=query_term, country="IN")
                                final_products.extend(api_products)
                                logger.info(f"Found {len(api_products)} products for style query: '{query_term}'")
                        else:
                            logger.warning("Outfit builder failed to get suggestions, falling back to simple search.")
                            initial_search_term = f"{' '.join(base_item_attributes)} {new_item_type}"
                    else:
                        logger.warning("Coordination prompt with image, but couldn't parse new item type. Falling back.")
                        initial_search_term = ' '.join(generated_tags[:3]) if generated_tags else generated_caption

                # Scenario: Image + Standard Prompt (e.g., "make it a brighter red")
                elif prompt:
                    top_tags = ' '.join(generated_tags[:3])
                    initial_search_term = f"{prompt} {top_tags}"

                # Scenario: Image only
                else:
                    initial_search_term = ' '.join(generated_tags[:3]) if generated_tags else generated_caption

                logger.info(f"Image features extracted. Caption: '{generated_caption}'. Tags: {generated_tags}. Using '{initial_search_term}' for API search.")

            else:
                return jsonify({"error": f"Failed to process image: {image_file.filename}. It may be corrupt."}), 500
        
        # --- 1b. Contextual Search Logic (No Image) ---
        else: 
            if not prompt:
                return jsonify({"error": "Please provide a prompt."}), 400
            
            initial_search_term = ""
            final_products = [] 
            is_coordination_query = any(keyword in prompt.lower() for keyword in COORDINATION_KEYWORDS)

            # Check for coordination intent first
            if is_coordination_query and last_search_context.get('primary_object_type'):
                logger.info(f"Outfit coordination query detected. Last context: {last_search_context}")
                
                new_item_type = parse_new_item_type_from_prompt(prompt, KNOWN_OBJECT_TYPES)
                
                if new_item_type and refinement_service:
                    base_item_desc = f"{' '.join(last_search_context.get('primary_attributes', []))} {last_search_context.get('primary_object_type')}".strip()
                    logger.info(f"Base item for style matching: '{base_item_desc}'. New item: '{new_item_type}'.")

                    suggested_queries = refinement_service.suggest_style_queries(base_item_desc, new_item_type)
                    
                    if suggested_queries:
                        logger.info(f"LLM suggested search queries: {suggested_queries}")
                        all_api_products = []
                        if rapidapi_amazon_service:
                            for query in suggested_queries:
                                api_products = rapidapi_amazon_service.search(query=query, country="IN")
                                all_api_products.extend(api_products)
                        
                        initial_search_term = f"Finding items that go with a {base_item_desc}..." # This is more for logging now
                        final_products = all_api_products # These products will bypass the initial search stage
                    else:
                        logger.warning("LLM style suggestion failed. Falling back to a simple search for the new item.")
                        initial_search_term = new_item_type
                else:
                    logger.info("Could not determine new item for coordination, or no refinement service. Treating as standard search.")
                    initial_search_term = prompt
            
            # If not coordination, check for standard attribute change
            else:
                is_contextual_query = any(keyword in prompt.lower() for keyword in CONTEXTUAL_REFERENCE_KEYWORDS)
                
                if is_contextual_query and last_search_context.get('primary_object_type'):
                    logger.info(f"Standard contextual query detected. Last context: {last_search_context}")
                    
                    new_elements = extract_key_elements_from_query(prompt)
                    new_item_type = new_elements.get('primary_object_type')
                    new_attributes = new_elements.get('primary_attributes', [])

                    # If user asks for the same item type again (e.g., "more t-shirts"), reuse last attributes.
                    if new_item_type and new_item_type == last_search_context.get('primary_object_type') and not new_attributes:
                        logger.info(f"Query for same item type ('{new_item_type}') with no new attributes. Reusing last attributes.")
                        preserved_attributes = last_search_context.get('primary_attributes', [])
                        initial_search_term = f"{' '.join(preserved_attributes)} {new_item_type}"
                    
                    # If user asks for a new item type (e.g., "how about a jacket")
                    elif new_item_type:
                        final_attributes = list(set(last_search_context.get('primary_attributes', []) + new_attributes))
                        initial_search_term = f"{' '.join(final_attributes)} {new_item_type}"

                    # If user just specifies new attributes (e.g., "in blue")
                    else:
                        final_attributes = list(set(new_attributes))
                        if not final_attributes: # Should not happen with contextual query, but as a fallback
                            final_attributes = last_search_context.get('primary_attributes', [])
                        initial_search_term = f"{' '.join(final_attributes)} {last_search_context['primary_object_type']}"
                    
                    logger.info(f"Constructed new search term from context: '{initial_search_term}'")
                
                # If nothing contextual matches, it's a new search
                else:
                    last_search_context.clear()
                    logger.info("No image and not a contextual query. Treating as a new search.")
                    initial_search_term = prompt

        # --- 2. Initial Product Retrieval ---
        # This stage is skipped if the outfit builder (with or without image) already populated `final_products`
        if not final_products and initial_search_term:
            logger.info(f"Performing initial product search on APIs with term: '{initial_search_term}'")
            if rapidapi_amazon_service:
                api_products = rapidapi_amazon_service.search(query=initial_search_term, country="IN")
                logger.info(f"Found {len(api_products)} products from API search.")
                final_products.extend(api_products)
            else:
                logger.warning("No product search service available. Unable to perform initial search.")

        # --- 3. Vector Index Management ---
        # The vector index is only relevant for image-based searches, which are handled above.
        # For text-based outfit building, we rely on the API results.
        
        # --- 4. Product Merging ---
        # The `final_products` list already contains the merged results from the multi-search or single search.
        # We just need to deduplicate them.
        unique_products = {p['id']: p for p in final_products}.values()
        logger.info(f"Final list contains {len(unique_products)} unique products.")

        # --- 5. Context Update ---
        is_coordination_query = any(keyword in prompt.lower() for keyword in COORDINATION_KEYWORDS)

        # Update context based on the *new* search that was just performed.
        if initial_search_term:
            if not is_coordination_query:
                context_elements = extract_key_elements_from_query(initial_search_term)
                # Persist the full context from the search that was just run
                last_search_context['primary_object_type'] = context_elements.get('primary_object_type')
                last_search_context['primary_attributes'] = context_elements.get('primary_attributes', [])
                logger.debug(f"Updated primary object/attributes in context: {last_search_context}")

        # --- 6. LLM Refinement ---
        # The LLM has already been used for suggestions. We can probably skip re-ranking for now.
        # Or, we could re-rank the merged list against the original user prompt.
        # For simplicity, we will skip refinement for style queries for now.

        # --- 7. Final Response ---
        response_text = f"Here are {len(unique_products)} products I found based on your request."
        is_coordination_query = any(keyword in prompt.lower() for keyword in COORDINATION_KEYWORDS)
        if is_coordination_query and unique_products:
            # Re-calculate base_item_desc if it was a no-image query
            if not image_file and last_search_context.get('primary_object_type'):
                 base_item_desc = f"{' '.join(last_search_context.get('primary_attributes', []))} {last_search_context.get('primary_object_type')}".strip()
            elif image_file: # Re-calculate for image-based coordination query
                 base_item_attributes = [tag for tag in generated_tags if tag not in KNOWN_OBJECT_TYPES]
                 base_item_type = next((tag for tag in generated_tags if tag in KNOWN_OBJECT_TYPES), "item")
                 base_item_desc = f"{' '.join(base_item_attributes)} {base_item_type}".strip()
            else:
                base_item_desc = "your previous item"

            response_text = f"Based on your request, here are some items that might go well with the {base_item_desc}:"

        return jsonify({
            "text": response_text,
            "products": list(unique_products),
            "isBot": True
        })

    except Exception as e:
        logger.critical(f"An unexpected error occurred in /api/query: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    # Basic check
    health_status = {"status": "ok", "services": {}}
    
    # Check services
    health_status["services"]["embedding"] = "ok" if embedding_service else "unavailable"
    health_status["services"]["refinement_llm"] = "ok" if refinement_service and refinement_service.model else "unavailable"
    health_status["services"]["product_search"] = "ok" if product_search_service else "unavailable"
    
    is_healthy = all(status == "ok" for status in health_status["services"].values())
    
    return jsonify(health_status), 200 if is_healthy else 503

if __name__ == '__main__':
    # Use a development server. For production, use a WSGI server like Gunicorn.
    # Example: gunicorn --bind 0.0.0.0:5001 app:app
    # The port should match what your frontend expects.
    app.run(debug=True, host='0.0.0.0', port=5001)

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
# Enable CORS for all routes with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "supports_credentials": True
    }
})
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
from services.refine_with_llm import RefinementService
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

# Stores details from the last successful search to provide conversational context
last_search_context: Dict = {}

# Known common object types for simple parsing
KNOWN_OBJECT_TYPES = {
    'shirt', 't-shirt', 'jacket', 'dress', 'pants', 'shoes', 'top', 'blouse', 'skirt', 'jeans', 'sweater', 'coat',
    'shirts', 't-shirts', 'jackets', 'dresses', 'shoe', 'tops', 'blouses', 'skirts', 'sweaters', 'coats' # Added plurals & shoe singular
}
COMMON_STOP_WORDS = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'on', 'in', 'at', 'to', 'for', 'with', 'of', 'by', 'and', 'or', 'but', 'it', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'men', "men's", 'womens', "women's", 'lady', "lady's"} # Added some common gendered words that are often not core attributes
COMMON_COLORS = {"red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown", "gray", "grey", "beige", "navy", "teal", "maroon", "olive", "silver", "gold"}

# Keywords indicating a reference to the last search context
CONTEXTUAL_REFERENCE_KEYWORDS = {
    # General references to previous item
    "last item", "previous item", "last one", "that one", 
    "similar to last", "like the last", "like that",
    
    # References to attributes of the last item
    "that color", "that colour", "its color", "its colour",
    "same color", "same colour", "in that color", "in that colour",
    "in the color of", "in the colour of", # More flexible
    "color of last", "colour of last",
    
    # References to type of last item (will also be covered by loop below but good to have common ones)
    "last t-shirt", "last shirt", "last jacket", "last dress",
    
    # General similarity
    "similar to that",
}

# Add more specific item references and phrases like "... of the last [item_type]"
for item_type in KNOWN_OBJECT_TYPES:
    CONTEXTUAL_REFERENCE_KEYWORDS.add(f"last {item_type}")
    CONTEXTUAL_REFERENCE_KEYWORDS.add(f"the last {item_type}") # Handles "...of the last tshirt"
    CONTEXTUAL_REFERENCE_KEYWORDS.add(f"previous {item_type}")
    CONTEXTUAL_REFERENCE_KEYWORDS.add(f"color of the last {item_type}")
    CONTEXTUAL_REFERENCE_KEYWORDS.add(f"colour of the last {item_type}")

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

def merge_and_dedupe_products(products_list1: List[Dict], products_list2: List[Dict]) -> List[Dict]: # UPDATED: Takes two lists again
    """
    Merges two lists of products and dedupulicates them based on title and image_url.
    """
    combined_products = products_list1 + products_list2 # REINSTATED: Combining two lists here
    logger.debug(f"Starting dedupe for {len(combined_products)} products (from {len(products_list1)} + {len(products_list2)})." )
    
    seen_products = set()
    deduped_list = []
    
    for product in combined_products: # Iterate over the combined list
        # Create a unique key for deduplication, prefer non-None titles and image_urls
        title = product.get("title", "") or ""
        image_url = product.get("image_url", "") or ""
        
        # Normalize by lowercasing and removing simple punctuation/whitespace for better matching
        # This is a basic normalization. More sophisticated methods could be used.
        normalized_title = "".join(filter(str.isalnum, title.lower()))
        
        # Use a tuple of (normalized_title_prefix, image_url) as a key.
        # Using a prefix of title can help catch near-duplicates if full titles vary slightly.
        # Image URL is a strong signal for uniqueness if available.
        # If image_url is not available, rely more on title.
        key_title_part = normalized_title[:50] # Use first 50 chars of normalized title

        if image_url:
            product_key = (key_title_part, image_url)
        else: # Fallback if no image_url, rely on title more heavily (less reliable for dedupe)
            product_key = (key_title_part,)

        if product_key not in seen_products:
            seen_products.add(product_key)
            deduped_list.append(product)
        else:
            logger.debug(f"Duplicate product skipped: Title '{title}', Image '{image_url}'")
            
    logger.debug(f"Finished dedupe. Original combined: {len(combined_products)}, Deduped: {len(deduped_list)}")
    return deduped_list

@app.route('/api/query', methods=['POST'])
def query_route():
    start_time_total = time.time()
    logger.info("Received API query request")
    
    image_file = request.files.get('image')
    prompt_text = request.form.get('prompt', '')
    use_context = request.form.get('use_context', 'false').lower() == 'true'
    
    image_path = None
    image_tags_hint = None # For category search via RapidAPI
    blip_caption_for_query = None # For text search via RapidAPI if no prompt

    if image_file and allowed_file(image_file.filename):
        start_time_save = time.time()
        filename = secure_filename(image_file.filename)
        # Use /tmp directory for Vercel compatibility
        temp_dir = "/tmp" 
        if not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir) # Should not be strictly necessary on Vercel /tmp but good practice
            except OSError as e:
                logger.error(f"Error creating /tmp directory (should be unnecessary on Vercel): {e}")
                # Fallback or error, though /tmp should exist and be writable
        
        image_path = os.path.join(temp_dir, filename)
        try:
            image_file.save(image_path)
            logger.info(f"File saved to: {image_path} in {time.time() - start_time_save:.2f}s")

            # Process image for tags (category hint) and BLIP caption (potential query)
            start_time_img_proc = time.time()
            try:
                with Image.open(image_path) as img:
                    img_rgb = img.convert('RGB') # Ensure image is in RGB for models
                    
                    # Get image tags for category hint
                    if embedding_service:
                        tags = embedding_service.get_image_tags(img_rgb, top_n=2) # Get top 2 tags
                        if tags:
                            image_tags_hint = " ".join(tags) 
                            logger.info(f"Generated category hint from image tags: '{image_tags_hint}'")
                    
                    # Generate BLIP caption if no user prompt
                    if not prompt_text and blip_model and blip_processor:
                        inputs = blip_processor(images=img_rgb, return_tensors="pt")
                        out = blip_model.generate(**inputs, max_length=75) # Increased max_length for more descriptive captions
                        blip_caption_for_query = blip_processor.decode(out[0], skip_special_tokens=True)
                        logger.info(f"BLIP caption: '{blip_caption_for_query}' generated")

                logger.info(f"Image opened and processed (tags/BLIP) in {time.time() - start_time_img_proc:.2f}s")
            except Exception as e:
                logger.error(f"Error processing image {image_path} for tags/BLIP: {e}\n{traceback.format_exc()}")
        except Exception as e:
            logger.error(f"Error saving file {filename} to {temp_dir}: {e}\n{traceback.format_exc()}")
            image_path = None # Ensure image_path is None if save fails

    # --- Contextual Query Handling ---
    user_prompt_for_context_check = prompt_text.lower()
    original_user_prompt = prompt_text # Keep original case for later use
    is_contextual_query = False
    contextual_search_query_for_apis = None

    if last_search_context and user_prompt_for_context_check:
        logger.debug(f"Checking for contextual reference in prompt: '{user_prompt_for_context_check}'")
        logger.debug(f"Current last_search_context: {last_search_context}")
        
        found_ref_keyword = None
        # Check if any keyword is a substring of the prompt, to catch phrases more flexibly
        for ref_keyword in sorted(list(CONTEXTUAL_REFERENCE_KEYWORDS), key=len, reverse=True): # Check longer keywords first
            if ref_keyword in user_prompt_for_context_check:
                found_ref_keyword = ref_keyword
                logger.debug(f"Found contextual keyword: '{found_ref_keyword}' in prompt: '{user_prompt_for_context_check}'")
                break
        
        if found_ref_keyword:
            logger.info(f"Contextual reference detected ('{found_ref_keyword}'). Trying to apply last search context.")
            is_contextual_query = True
            
            # Try to parse a new item type from the current prompt
            new_item_type = parse_new_item_type_from_prompt(user_prompt_for_context_check, KNOWN_OBJECT_TYPES)
            
            # Get attributes from last search (e.g., color)
            last_primary_attributes = last_search_context.get("primary_attributes", [])
            last_object_type = last_search_context.get("primary_object_type")
            
            query_parts = []
            # Check if the reference is specifically about color
            is_color_reference = any(c_ref in found_ref_keyword for c_ref in ["color", "colour"])
            if not is_color_reference: # Also check the broader prompt text if keyword itself wasn't specific enough
                is_color_reference = any(c_ref in user_prompt_for_context_check for c_ref in ["color", "colour"])

            if is_color_reference:
                logger.info("Contextual query is color-focused. Prioritizing color attributes from last search.")
                # Extract only common color words from the last primary attributes
                extracted_colors = [attr for attr in last_primary_attributes if attr in COMMON_COLORS]
                if extracted_colors:
                    query_parts.extend(extracted_colors)
                    logger.info(f"Using specific colors from last context: {extracted_colors}")
                else:
                    # If no common colors found in attributes, but it was a color reference,
                    # perhaps don't add any attributes or add a generic one if that makes sense.
                    # For now, we'll be conservative and not add non-specific attributes if specific colors were asked for but not found.
                    logger.info("Color reference made, but no common colors found in last_primary_attributes. Not adding color attributes to query.")
            elif last_primary_attributes: # Not a specific color reference, but there are previous attributes
                # Use non-stop-word attributes from last search, could be style, material etc.
                # We might want to be even more selective here in future (e.g. exclude a wider range of generic nouns/verbs)
                attributes_to_carry = [attr for attr in last_primary_attributes if attr not in COMMON_STOP_WORDS and attr not in COMMON_COLORS]
                # If there are too many attributes, prioritize (e.g. first N, or based on POS tagging - advanced)
                # For now, take a few if many are present
                if len(attributes_to_carry) > 3: # Heuristic: limit carried-over non-color attributes
                    logger.info(f"Too many non-color attributes from last context ({attributes_to_carry}), consider refining selection. Using first 3 for now.")
                    query_parts.extend(attributes_to_carry[:3]) 
                else:
                    query_parts.extend(attributes_to_carry)
                logger.info(f"Using general attributes from last context: {query_parts}")
            
            if new_item_type:
                query_parts.append(new_item_type)
                logger.info(f"Identified new item type for contextual query: '{new_item_type}'")
            elif last_object_type: # If no new type but there was an old one, re-use old one with new attributes implicitly
                query_parts.append(last_object_type)
                logger.info(f"No new item type, re-using last object type: '{last_object_type}'")
            
            if query_parts:
                contextual_search_query_for_apis = " ".join(query_parts)
                logger.info(f"Constructed contextual search query: '{contextual_search_query_for_apis}'")
            else:
                logger.warning("Could not construct a meaningful contextual query. Falling back.")
                is_contextual_query = False # Reset flag
        else:
            logger.debug("No contextual reference keyword found in prompt.")
    # --- End Contextual Query Handling ---

    search_query_for_apis: str
    query_generation_start_time = time.time()

    # Use original_user_prompt for processing now, not the lowercased version
    prompt = original_user_prompt 

    if is_contextual_query and contextual_search_query_for_apis:
        logger.info(f"Using contextual query for APIs: '{contextual_search_query_for_apis}'")
        search_query_for_apis = contextual_search_query_for_apis
        # The user's original prompt (which was contextual) is still passed to refinement service later.
    elif prompt:
        logger.info(f"User provided prompt: '{prompt}'. Using it to generate search query for APIs.")
        search_query_for_apis = refinement_service.generate_shopping_query(prompt)
        logger.info(f"LLM generated search_query_for_apis from user prompt: {search_query_for_apis} in {time.time() - query_generation_start_time:.2f}s")
        # Note: blip_caption might still be generated if an image is present, for context storage, but not for API query if prompt is given.
        # We need to decide if BLIP caption should be generated even when prompt is primary.
        # For now, let's assume if prompt is given, search_query_for_apis is SOLELY from prompt.
        # We still need blip_caption for last_search_context if no prompt was given for THAT turn.
        # If an image is present with a prompt, we can still generate blip_caption for potential future reference or just to have it.
        if blip_model and blip_processor: # Generate blip_caption if model available, even if prompt is used for main query
            blip_start_time_prompt = time.time()
            inputs_prompt = blip_processor(images=img_rgb, return_tensors="pt")
            out_prompt = blip_model.generate(**inputs_prompt, max_new_tokens=50)
            caption_temp = blip_processor.decode(out_prompt[0], skip_special_tokens=True)
            blip_caption_for_query = caption_temp # Store it in the broader scope variable
            logger.info(f"BLIP caption (generated alongside prompt-based query): '{blip_caption_for_query}' in {time.time() - blip_start_time_prompt:.2f}s")

    elif blip_model and blip_processor: # This case: No contextual query, NO user prompt, so rely on BLIP
        logger.info("No user prompt and not contextual. Generating image caption with BLIP for API query.")
        blip_start_time = time.time()
        inputs = blip_processor(images=img_rgb, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        blip_caption_for_query = caption # Assign to the broader scope variable
        logger.info(f"BLIP caption: '{caption}' generated in {time.time() - blip_start_time:.2f}s")
        
        refine_blip_start_time = time.time()
        search_query_for_apis = refinement_service.generate_shopping_query(caption) # Query from BLIP
        logger.info(f"LLM generated search_query_for_apis from BLIP caption in {time.time() - refine_blip_start_time:.2f}s. Total query gen: {time.time() - query_generation_start_time:.2f}s")
    else:
        logger.warning("No contextual query, no user prompt, and BLIP model not available. Using category hint or fallback for API query.")
        # Fallback: Use category hint if available, otherwise a generic term
        search_query_for_apis = image_tags_hint if image_tags_hint else 'product'
        if not image_tags_hint and embedding_service:
             # If no category hint from tags but embedding service is there, try to get at least one tag as a last resort for query
            try:
                tags_fallback = embedding_service.get_image_tags(img_rgb, top_k=1)
                if tags_fallback: search_query_for_apis = tags_fallback[0]
            except Exception: pass # Ignore if this fails, use 'product'
        logger.info(f"Fallback search query: {search_query_for_apis} in {time.time() - query_generation_start_time:.2f}s")

    api_call_start_time = time.time()
    # all_raw_products: List[Dict] = [] # REMOVED: We will have separate lists
    amazon_products: List[Dict] = []
    flipkart_products: List[Dict] = []

    # REMOVE Single call to the unified product search service
    # if product_search_service: 
    #    ...
    # else:
    #    logger.warning("UnifiedProductSearch service (BaseRapidApiService) not initialized. Skipping product search.")

    # ADDED: Calls to separate Amazon and Flipkart services
    if rapidapi_amazon_service:
        logger.info(f"Calling RapidAPI-Amazon Service with query: '{search_query_for_apis}', category hint: '{image_tags_hint if image_tags_hint else 'N/A'}'")
        # Amazon /search uses text query primarily; category_search_term might be ignored if not configured for this specific API endpoint.
        amazon_products = rapidapi_amazon_service.search_products(
            query=search_query_for_apis,
            limit=15, 
            category_search_term=image_tags_hint
        )
        logger.info(f"RapidAPI-Amazon Service returned {len(amazon_products)} products.")
    else:
        logger.warning("RapidAPI-Amazon service not initialized. Skipping Amazon product search.")

    if rapidapi_flipkart_service:
        logger.info(f"Calling RapidAPI-Flipkart Service with category: '{image_tags_hint if image_tags_hint else 'N/A'}', text query hint: '{search_query_for_apis}'")
        # Flipkart /products-by-category uses category_search_term primarily.
        # search_query_for_apis will only be used if RAPIDAPI_FLIPKART_TEXT_QUERY_PARAM is set in .env.
        flipkart_products = rapidapi_flipkart_service.search_products(
            query=search_query_for_apis, 
            limit=15,
            category_search_term=image_tags_hint 
        )
        logger.info(f"RapidAPI-Flipkart Service returned {len(flipkart_products)} products.")
    else:
        logger.warning("RapidAPI-Flipkart service not initialized. Skipping Flipkart product search.")

    logger.info(f"API calls completed in {time.time() - api_call_start_time:.2f}s")

    merging_start_time = time.time()
    # all_products = merge_and_dedupe_products(all_raw_products) # REMOVED
    all_products = merge_and_dedupe_products(amazon_products, flipkart_products) # UPDATED: Pass the two lists
    logger.info(f"Merged and deduped products in {time.time() - merging_start_time:.2f}s. Total products: {len(all_products)}")

    # Optional: Rerank based on CLIP similarity (if embedding_service is available and products have image_url)
    # This requires products to have 'image_url' and embedding_service to be functional
    # For now, we'll just use the order from the API after deduplication.

    final_refinement_start_time = time.time()
    MAX_PRODUCTS_TO_REFINE = int(os.getenv('MAX_PRODUCTS_TO_REFINE', 10)) # Default to 10
    refined_products = []

    if refinement_service:
        # Only refine if there are products and a refinement service
        if all_products:
            logger.info(f"Refining up to {MAX_PRODUCTS_TO_REFINE} products using RefinementService.")
            # Construct a context string for refinement if needed by the LLM
            context_for_refinement = f"Original user prompt (if any): {prompt}. Image caption/tags generated: {search_query_for_apis}. Category hint used: {image_tags_hint if image_tags_hint else 'N/A'}."
            if is_contextual_query: # Add more context if it was a contextual query
                context_for_refinement = f"Contextual User Prompt: {prompt}. Search derived from context: '{search_query_for_apis}'. Image (if any) provided with this prompt. Last context was: {last_search_context}. Category hint: {image_tags_hint if image_tags_hint else 'N/A'}."
            
            refined_products = refinement_service.refine_results(
                context_for_refinement,
                all_products[:MAX_PRODUCTS_TO_REFINE]
            )
            logger.info(f"RefinementService processed {len(refined_products)} products in {time.time() - final_refinement_start_time:.2f}s")
        else:
            logger.info("No products to refine.")
            refined_products = []
    else:
        logger.warning("RefinementService not available. Using products as is (up to MAX_PRODUCTS_TO_REFINE).")
        refined_products = all_products[:MAX_PRODUCTS_TO_REFINE] # Fallback: just take top N

    # If refinement somehow failed or returned empty, but we had products, fallback to unrefined (deduped) list
    if not refined_products and all_products:
        logger.warning("Refinement returned empty list, but original products existed. Falling back to non-refined list (up to MAX_PRODUCTS_TO_REFINE).")
        refined_products = all_products[:MAX_PRODUCTS_TO_REFINE]
    
    processing_time = time.time() - start_time_total
    logger.info(f"Total request processing time: {processing_time:.2f}s. Returning {len(refined_products)} products.")
    
    # --- Update last_search_context (AFTER successful processing) ---
    # Ensure blip_caption_value is defined correctly based on whether prompt was used
    blip_caption_value = None
    if not prompt and blip_caption_for_query: # Use the blip_caption variable defined in the broader scope
        blip_caption_value = blip_caption_for_query
        
    if search_query_for_apis: # Ensure we have a query that was used for APIs
        extracted_elements = extract_key_elements_from_query(search_query_for_apis)
        last_search_context = {
            "search_query_for_apis": search_query_for_apis,
            "primary_object_type": extracted_elements.get("primary_object_type"),
            "primary_attributes": extracted_elements.get("primary_attributes", []),
            "image_tags_hint": image_tags_hint, # The one used for category search if any
            "user_prompt_at_time": prompt, # Original user prompt for this search
            "blip_caption_at_time": blip_caption_value # BLIP caption if used
        }
        logger.info(f"Updated last_search_context: {last_search_context}")
    else:
        logger.warning("Skipping update to last_search_context as search_query_for_apis was empty.")
    # --- End Update last_search_context ---
    
    return jsonify({
        'products': refined_products,
        'search_query_used': search_query_for_apis,
        'category_hint_used': image_tags_hint,
        'processing_time_seconds': round(processing_time, 2),
        'message': 'Query processed successfully.'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'ShopSmarter backend is running.'}), 200

if __name__ == '__main__':
    # Ensure UPLOAD_FOLDER exists (it's also checked during app init, but good to double-check)
    if not os.path.exists(UPLOAD_FOLDER):
        try:
            os.makedirs(UPLOAD_FOLDER)
            app.logger.info(f"UPLOAD_FOLDER created at {UPLOAD_FOLDER} before app run.")
        except Exception as e_mkdir:
            app.logger.error(f"Could not create UPLOAD_FOLDER at {UPLOAD_FOLDER}: {e_mkdir}")
            # Decide if this is fatal or if the app should attempt to run anyway
            # For now, we'll let it try, as init also checks.

    app.run(debug=True, host='0.0.0.0', port=5001)

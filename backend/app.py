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
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
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
    start_time = time.time()
    app.logger.info("Received API query request")
    if 'image' not in request.files:
        app.logger.error("No image file in request")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    prompt = request.form.get('prompt', '')

    if file.filename == '':
        app.logger.error("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400

    time_before_save = time.time()
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    app.logger.info(f"File saved to: {filepath} in {time.time() - time_before_save:.2f}s")
    
    time_before_img_open = time.time()
    img_pil = Image.open(filepath).convert("RGB")
    app.logger.info(f"Image opened and converted in {time.time() - time_before_img_open:.2f}s")

    # --- Category Hint Generation (Simple Heuristic) ---
    category_search_term: Optional[str] = None
    if embedding_service:
        tag_gen_start_time = time.time()
        try:
            # Get top 2 tags for category hint
            image_tags = embedding_service.get_image_tags(img_pil, top_k=2) 
            if image_tags:
                category_search_term = " ".join(image_tags)
                app.logger.info(f"Generated category hint from image tags: '{category_search_term}' in {time.time() - tag_gen_start_time:.2f}s")
            else:
                app.logger.info(f"No image tags returned for category hint. Time: {time.time() - tag_gen_start_time:.2f}s")
        except Exception as e_tag:
            app.logger.error(f"Error getting image tags for category hint: {e_tag}. Proceeding without category hint.")
            app.logger.error(traceback.format_exc())
    else:
        app.logger.warning("EmbeddingService not available, cannot generate category hint from image tags.")

    search_query_for_apis: str
    query_generation_start_time = time.time()

    if prompt:
        app.logger.info(f"User provided prompt: '{prompt}'. Using it to generate search query.")
        search_query_for_apis = refinement_service.generate_shopping_query(prompt)
        app.logger.info(f"LLM generated search query from prompt: {search_query_for_apis} in {time.time() - query_generation_start_time:.2f}s")
    elif blip_model and blip_processor:
        app.logger.info("No user prompt. Generating image caption with BLIP.")
        blip_start_time = time.time()
        inputs = blip_processor(images=img_pil, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        app.logger.info(f"BLIP caption: '{caption}' generated in {time.time() - blip_start_time:.2f}s")
        
        refine_blip_start_time = time.time()
        search_query_for_apis = refinement_service.generate_shopping_query(caption)
        app.logger.info(f"LLM generated search query from BLIP caption in {time.time() - refine_blip_start_time:.2f}s. Total query gen: {time.time() - query_generation_start_time:.2f}s")
    else:
        app.logger.warning("BLIP model not available and no prompt. Using category hint as search query if available, else 'product'.")
        # Fallback: Use category hint if available, otherwise a generic term
        search_query_for_apis = category_search_term if category_search_term else 'product'
        if not category_search_term and embedding_service:
             # If no category hint from tags but embedding service is there, try to get at least one tag as a last resort for query
            try:
                tags_fallback = embedding_service.get_image_tags(img_pil, top_k=1)
                if tags_fallback: search_query_for_apis = tags_fallback[0]
            except Exception: pass # Ignore if this fails, use 'product'
        app.logger.info(f"Fallback search query: {search_query_for_apis} in {time.time() - query_generation_start_time:.2f}s")

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
        app.logger.info(f"Calling RapidAPI-Amazon Service with query: '{search_query_for_apis}', category hint: '{category_search_term if category_search_term else 'N/A'}'")
        # Amazon /search uses text query primarily; category_search_term might be ignored if not configured for this specific API endpoint.
        amazon_products = rapidapi_amazon_service.search_products(
            query=search_query_for_apis,
            limit=15, 
            category_search_term=category_search_term
        )
        app.logger.info(f"RapidAPI-Amazon Service returned {len(amazon_products)} products.")
    else:
        app.logger.warning("RapidAPI-Amazon service not initialized. Skipping Amazon product search.")

    if rapidapi_flipkart_service:
        app.logger.info(f"Calling RapidAPI-Flipkart Service with category: '{category_search_term if category_search_term else 'N/A'}', text query hint: '{search_query_for_apis}'")
        # Flipkart /products-by-category uses category_search_term primarily.
        # search_query_for_apis will only be used if RAPIDAPI_FLIPKART_TEXT_QUERY_PARAM is set in .env.
        flipkart_products = rapidapi_flipkart_service.search_products(
            query=search_query_for_apis, 
            limit=15,
            category_search_term=category_search_term 
        )
        app.logger.info(f"RapidAPI-Flipkart Service returned {len(flipkart_products)} products.")
    else:
        app.logger.warning("RapidAPI-Flipkart service not initialized. Skipping Flipkart product search.")

    app.logger.info(f"API calls completed in {time.time() - api_call_start_time:.2f}s")

    merging_start_time = time.time()
    # all_products = merge_and_dedupe_products(all_raw_products) # REMOVED
    all_products = merge_and_dedupe_products(amazon_products, flipkart_products) # UPDATED: Pass the two lists
    app.logger.info(f"Merged and deduped products in {time.time() - merging_start_time:.2f}s. Total products: {len(all_products)}")

    # Optional: Rerank based on CLIP similarity (if embedding_service is available and products have image_url)
    # This requires products to have 'image_url' and embedding_service to be functional
    # For now, we'll just use the order from the API after deduplication.

    final_refinement_start_time = time.time()
    MAX_PRODUCTS_TO_REFINE = int(os.getenv('MAX_PRODUCTS_TO_REFINE', 10)) # Default to 10
    refined_products = []

    if refinement_service:
        # Only refine if there are products and a refinement service
        if all_products:
            app.logger.info(f"Refining up to {MAX_PRODUCTS_TO_REFINE} products using RefinementService.")
            # Construct a context string for refinement if needed by the LLM
            context_for_refinement = f"Original user prompt (if any): {prompt}. Image caption/tags generated: {search_query_for_apis}. Category hint used: {category_search_term if category_search_term else 'N/A'}."
            
            refined_products = refinement_service.refine_results(
                context_for_refinement,
                all_products[:MAX_PRODUCTS_TO_REFINE]
            )
            app.logger.info(f"RefinementService processed {len(refined_products)} products in {time.time() - final_refinement_start_time:.2f}s")
        else:
            app.logger.info("No products to refine.")
            refined_products = []
    else:
        app.logger.warning("RefinementService not available. Using products as is (up to MAX_PRODUCTS_TO_REFINE).")
        refined_products = all_products[:MAX_PRODUCTS_TO_REFINE] # Fallback: just take top N

    # If refinement somehow failed or returned empty, but we had products, fallback to unrefined (deduped) list
    if not refined_products and all_products:
        app.logger.warning("Refinement returned empty list, but original products existed. Falling back to non-refined list (up to MAX_PRODUCTS_TO_REFINE).")
        refined_products = all_products[:MAX_PRODUCTS_TO_REFINE]
    
    processing_time = time.time() - start_time
    app.logger.info(f"Total request processing time: {processing_time:.2f}s. Returning {len(refined_products)} products.")
    
    # Ensure consistent response structure
    return jsonify({
        'products': refined_products,
        'search_query_used': search_query_for_apis,
        'category_hint_used': category_search_term,
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

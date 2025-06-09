import logging
# import torch # No longer needed directly for transformers model
import json
import os
# from transformers import AutoModelForCausalLM, AutoTokenizer # Replaced by llama_cpp
from llama_cpp import Llama, LlamaGrammar # Added Llama and LlamaGrammar
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

# Define a simple grammar to encourage JSON list output
# This is a basic example; more robust JSON grammars can be constructed.
JSON_LIST_GRAMMAR = r'''
root   ::= "[" ( (value ("," value)*)? ) "]" space
value  ::= string
string ::= "\"" (([^"]))* "\""
space  ::= ([ \t\n])*
'''
# Note: LlamaGrammar might require `pip install lark` if not already a sub-dependency of llama-cpp-python
# For simplicity, we'll try without grammar first, as modern instruct models are good with JSON.
# If output is inconsistent, grammar can be enabled.

class RefinementService:
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048, n_threads: Optional[int] = None):
        """
        Initialize the refinement service with a GGUF LLM loaded via llama-cpp-python.
        Args:
            model_path (Optional[str]): Path to the GGUF model file.
                                        Defaults to LLM_GGUF_MODEL_PATH env var.
            n_ctx (int): Context size for the model.
            n_threads (Optional[int]): Number of threads to use. Defaults to llama.cpp's default.
        """
        logger.info(f"RefinementService (llama-cpp-python) initializing...")

        _model_path = model_path or os.getenv("LLM_GGUF_MODEL_PATH")
        
        if not _model_path:
            logger.error("LLM_GGUF_MODEL_PATH environment variable not set, or no model_path provided.")
            logger.error("RefinementService will not be functional. Falling back to no-op refinement.")
            self.model = None
            return

        if not os.path.exists(_model_path):
            logger.error(f"GGUF model file not found at path: {_model_path}")
            logger.error("RefinementService will not be functional. Falling back to no-op refinement.")
            self.model = None
            return
            
        logger.info(f"Loading GGUF LLM model from: {_model_path}. Context size: {n_ctx}")
        
        try:
            # For CPU, n_gpu_layers=0.
            # n_batch can be adjusted, 512 is a common default.
            self.model = Llama(
                model_path=_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=0,  # Explicitly set to 0 for CPU only
                n_threads=n_threads, # None will let llama.cpp decide
                verbose=os.getenv("LLAMA_VERBOSE", "true").lower() == "true" # Control llama.cpp verbosity
            )
            # Try to load grammar - optional, can proceed if it fails or lark is not installed
            try:
                self.json_grammar = LlamaGrammar.from_string(JSON_LIST_GRAMMAR)
                logger.info("Successfully compiled JSON list grammar for LLM output.")
            except Exception as e:
                self.json_grammar = None
                logger.warning(f"Could not compile JSON grammar: {e}. Proceeding without grammar. Ensure model follows JSON instructions.")

            logger.info(f"GGUF LLM model loaded successfully from {_model_path}.")
        except Exception as e:
            logger.error(f"Failed to load GGUF LLM model from {_model_path}: {str(e)}")
            logger.error("RefinementService will not be functional. Falling back to no-op refinement.")
            self.model = None

    def _build_prompt(self, products: List[Dict], user_prompt: str) -> str:
        """
        Builds a detailed prompt for the LLM to refine product search results.
        (This prompt structure is for instruct-tuned models like Mistral Instruct)
        """
        if not products:
            return ""

        # System prompt can be added here if desired, but usually incorporated into the first user turn for llama.cpp
        # For Mistral Instruct, the format is typically <s>[INST] User Prompt [/INST] Model Response </s>
        # We will construct the "User Prompt" part.
        
        product_details_list = []
        for i, p in enumerate(products[:10]): # Limit to top 10 products to keep prompt manageable
            product_info_lines = [f"Product {i+1} (URL: {p.get('product_url', 'N/A')}, Title: {p.get('title', 'N/A')})"]
            price = p.get('price')
            if price is not None:
                try:
                    price_str = f"${float(price):.2f}"
                    product_info_lines.append(f"  Price: {price_str}")
                except (ValueError, TypeError):
                    product_info_lines.append(f"  Price: {price}")
            
            if p.get('tags'):
                product_info_lines.append(f"  Tags: {', '.join(p['tags'])}")
            if p.get('caption'):
                product_info_lines.append(f"  Description: {p['caption']}")
            product_details_list.append("\n".join(product_info_lines))
        
        product_section = "\n\n".join(product_details_list)

        # Create a more realistic example for the LLM
        example_urls = [p.get('product_url') for p in products[:2] if p.get('product_url')]
        example_json = json.dumps(example_urls) if example_urls else '["https://www.amazon.com/dp/B08N5G5TFG", "https://www.amazon.com/dp/B07M61KPFB"]'


        full_user_prompt = f"""You are a helpful shopping assistant. Your task is to refine and re-rank a list of products based on a user's specific request.

User's request: "{user_prompt}"

Products:
{product_section}

Based *only* on the user's request and the provided product information (title, price, tags, description), return a JSON list containing ONLY the URLs of the products that best match the user's request, in the order of relevance (most relevant first).

If no products are relevant, return an empty JSON list: [].
If all products are equally relevant, you can return them in their original order.
Ensure your output is ONLY the JSON list and nothing else. For example: {example_json}.
"""
        # For Mistral Instruct and similar models, wrap with [INST] tags
        return f"[INST] {full_user_prompt.strip()} [/INST]"


    def _parse_llm_output(self, llm_output: str, original_products: List[Dict]) -> List[Dict]:
        """
        Parses the LLM's JSON output (a list of URLs) and re-ranks the original products.
        """
        try:
            # Attempt to find JSON list within the output
            # llama.cpp output might be cleaner, but good to be robust
            json_str = llm_output.strip()
            if not (json_str.startswith('[') and json_str.endswith(']')):
                # Try to extract if it's embedded
                json_start = llm_output.find('[')
                json_end = llm_output.rfind(']')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    json_str = llm_output[json_start : json_end+1]
                else:
                    logger.warning(f"LLM output does not appear to be a JSON list: '{llm_output[:200]}...'. Falling back.")
                    return original_products
            
            ranked_urls = json.loads(json_str)
            
            if not isinstance(ranked_urls, list):
                logger.warning(f"LLM output was valid JSON but not a list: {ranked_urls}. Falling back.")
                return original_products

            product_map_by_url = {p.get("product_url"): p for p in original_products if p.get("product_url")}
            
            refined_products = []
            for url in ranked_urls:
                if isinstance(url, str) and url in product_map_by_url: # Ensure URL is a string
                    refined_products.append(product_map_by_url[url])
                else:
                    logger.warning(f"LLM returned URL not in original products or invalid type: {url}")
            
            logger.info(f"LLM refined products. Original count: {len(original_products)}, Refined count: {len(refined_products)}")
            return refined_products if refined_products or ranked_urls == [] else original_products

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed for LLM output: {e}. Output: '{llm_output[:200]}...'. Falling back.")
            return original_products
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}. Output: '{llm_output[:200]}...'. Falling back.")
            return original_products

    def refine_results(self, products: List[Dict], prompt: str) -> List[Dict]:
        """Refine product results based on the given prompt using the GGUF LLM."""
        if self.model is None:
            logger.warning("GGUF LLM model not loaded. Skipping refinement.")
            return products
        
        if not products:
            logger.info("No products to refine.")
            return []
            
        if not prompt:
            logger.info("No refinement prompt provided by user. Returning original products.")
            return products

        # Check if the prompt is too generic to be useful for refinement
        if prompt.lower().strip() in {"search", "find", "get", "products", "show products"}:
             logger.info(f"Prompt '{prompt}' is too generic. Skipping LLM refinement and returning original list.")
             return products

        full_prompt = self._build_prompt(products, prompt)
        if not full_prompt:
            return products

        logger.info(f"Sending prompt to GGUF LLM for refinement (first 200 chars of user content): {full_prompt[6:206]}...") # Skip [INST]
        
        try:
            generation_params = {
                "max_tokens": 1024,      # Increased tokens for longer lists
                "temperature": 0.1,    # Low temperature for factual JSON output
                "top_p": 0.9,
                "stop": ["]"],         # Stop when the JSON list is closed. The stop token itself is not included in output.
                "echo": False,          # Don't echo the prompt
            }
            # Add grammar if available and seems to work well with the model
            if self.json_grammar:
                generation_params["grammar"] = self.json_grammar
                logger.info("Using JSON grammar for LLM generation.")
            
            logger.info("Generating LLM response for refinement (llama-cpp-python)...")
            start_time = time.time()
            
            output = self.model(
                full_prompt,
                **generation_params
            )
            
            end_time = time.time()
            logger.info(f"LLM (llama-cpp-python) generation took {end_time - start_time:.2f} seconds.")

            llm_response_text = output["choices"][0]["text"].strip()
            
            # The stop token ']' is not included in the output. We append it to complete the JSON.
            # Handle cases where the model response is empty or doesn't start with a list.
            if not llm_response_text:
                # Model produced no output before stop token, assume empty list.
                llm_response_text = "[]"
            elif not llm_response_text.startswith('['):
                # Model started generating content without the opening bracket.
                # This is a guess, but it's better than failing.
                llm_response_text = f"[{llm_response_text}]"
            else:
                # This is the expected case, just add the closing bracket.
                llm_response_text = llm_response_text + "]"

            logger.info(f"LLM (llama-cpp-python) raw response: {llm_response_text}")
            
            return self._parse_llm_output(llm_response_text, products)

        except Exception as e:
            logger.error(f"Error during GGUF LLM refinement: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return products

    def suggest_style_queries(self, base_item_description: str, new_item_type: str, num_suggestions: int = 3) -> List[str]:
        """
        Uses the LLM to suggest search queries for an item that would stylistically match a base item.
        
        Args:
            base_item_description: A description of the item the user already has (e.g., "black t-shirt").
            new_item_type: The type of new item the user wants (e.g., "jacket").
            num_suggestions: The number of different search queries to generate.
            
        Returns:
            A list of suggested search query strings.
        """
        if not self.model:
            logger.warning("LLM model not loaded, cannot suggest style queries.")
            return []

        prompt = f"""You are a fashion stylist's assistant. A user has a '{base_item_description}' and wants to find a '{new_item_type}' that would go well with it. 
Your task is to generate {num_suggestions} diverse and specific search query ideas for an online shopping website. 
The queries should be practical and represent different styles (e.g., casual, formal, trendy).

IMPORTANT: The search queries must ONLY describe the '{new_item_type}'. They must NOT mention the original item ('{base_item_description}') or any other type of product.

Do not add any explanation, just return a single JSON list of strings.

Example:
Base Item: 'blue jeans'
New Item: 'shoes'
Output: ["white leather sneakers", "brown suede chelsea boots", "black canvas high-tops"]

Base Item: '{base_item_description}'
New Item: '{new_item_type}'
Output:"""

        logger.info(f"Generating style suggestions for base '{base_item_description}' and new item '{new_item_type}'.")
        
        try:
            # Use similar generation params as the refinement task, but maybe allow more creativity
            generation_params = {
                "max_tokens": 256,
                "temperature": 0.5, # Higher temperature for more creative/diverse suggestions
                "top_p": 0.9,
                "stop": ["]"], # Stop when the JSON list is closed
                "echo": False,
            }
            if self.json_grammar:
                generation_params["grammar"] = self.json_grammar

            output = self.model(prompt, **generation_params)
            
            raw_response = output['choices'][0]['text'] + "]" # Append the closing bracket
            logger.info(f"LLM raw response for style suggestion: {raw_response}")

            # Robust JSON parsing
            try:
                # Find the start of the JSON array
                json_start = raw_response.find('[')
                if json_start == -1:
                    logger.error("LLM did not return a JSON list for style suggestions.")
                    return []
                
                # Parse the JSON from the identified start
                suggestions = json.loads(raw_response[json_start:])
                
                if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                    logger.info(f"Successfully parsed style suggestions: {suggestions}")
                    return suggestions[:num_suggestions] # Ensure we don't return more than requested
                else:
                    logger.error(f"LLM returned a JSON object that was not a list of strings: {suggestions}")
                    return []

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from LLM style suggestion response: {raw_response}")
                return []

        except Exception as e:
            logger.error(f"An unexpected error occurred during style suggestion generation: {e}")
            return []

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # --- IMPORTANT ---
    # To run this example:
    # 1. Download a GGUF model (e.g., a small quantized Mistral or Llama model).
    #    Search "TheBloke GGUF" on Hugging Face. For example, Mistral-7B-Instruct-v0.2.Q4_K_M.gguf
    # 2. Set the environment variable LLM_GGUF_MODEL_PATH to the local path of your GGUF file.
    #    e.g., export LLM_GGUF_MODEL_PATH=/path/to/your/model.gguf
    # ---

    if not os.getenv("LLM_GGUF_MODEL_PATH"):
        logger.error(" Skipping RefinementService example: LLM_GGUF_MODEL_PATH not set.")
    else:
        logger.info("Running RefinementService (llama-cpp-python) example...")
        refiner = RefinementService()

        if refiner.model:
            sample_products = [
                {"url": "http://example.com/product1", "title": "Red Running Shoes", "price": {"current_price": "59.99", "currency": "$"}, "tags": ["footwear", "sports", "red"], "caption": "Comfortable red running shoes for athletes."},
                {"url": "http://example.com/product2", "title": "Blue Denim Jeans", "price": {"current_price": "79.50", "currency": "$"}, "tags": ["apparel", "denim", "blue"], "caption": "Stylish blue denim jeans for casual wear."},
                {"url": "http://example.com/product3", "title": "Red Cotton T-Shirt", "price": {"current_price": "19.99", "currency": "$"}, "tags": ["apparel", "cotton", "red"], "caption": "A bright red cotton t-shirt, very comfortable."},
                {"url": "http://example.com/product4", "title": "Fancy Red Dress", "price": {"current_price": "120.00", "currency": "$"}, "tags": ["apparel", "formal", "red", "dress"], "caption": "Elegant red dress for special occasions."}
            ]
            user_query = "I'm looking for affordable red clothing items, but not shoes."
            
            logger.info(f"Refining products with query: '{user_query}'")
            refined = refiner.refine_results(sample_products, user_query)
            
            logger.info("Refined Products:")
            for prod in refined:
                logger.info(f"  - {prod.get('title')} (URL: {prod.get('url')})")
        else:
            logger.error("Failed to initialize RefinementService for example.") 
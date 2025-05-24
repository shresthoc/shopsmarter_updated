"""
Refinement Service for ShopSmarter
Analyzes user queries and current results to extract filter criteria via LLM,
applies those filters, then re-ranks using a second LLM call for accuracy.
"""
import os
import logging
import json
from typing import List, Dict, Optional, Tuple
import uuid
import re
import traceback

import spacy
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from transformers import pipeline as hf_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Model configuration (CPU only)
MODEL_NAME = os.getenv("REFINE_MODEL", "facebook/opt-350m")

class FilterCriteria(BaseModel):
    """Structured filter criteria for product refinement."""
    min_price: Optional[float] = Field(None, description="Minimum price to include")
    max_price: Optional[float] = Field(None, description="Maximum price to include")
    style_keywords: List[str] = Field(default_factory=list, description="List of style descriptors to include")
    color_preferences: List[str] = Field(default_factory=list, description="List of preferred colors")
    excluded_terms: List[str] = Field(default_factory=list, description="List of terms to exclude from titles")
    sort_by: str = Field("relevance", description="Sorting criteria: relevance, price_low, price_high")

class RefinementService:
    def __init__(self):
        # Load spaCy model for text analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.error("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            raise
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            raise

        # Initialize HF text-generation pipelines on CPU
        # MODEL_NAME is fetched from os.getenv("REFINE_MODEL", ...) inside the try block
        model_name_for_llm = os.getenv("REFINE_MODEL", "facebook/opt-350m") # Default if not set
        logger.info(f"Attempting to load REFINE_MODEL: {model_name_for_llm}")

        try:
            # General purpose generator for tasks like filter extraction, reranking
            self.general_generator = hf_pipeline(
                task="text-generation",
                model=model_name_for_llm, # Use the fetched model name
                max_new_tokens=150,
                device=-1,  # CPU only
                trust_remote_code=True # Added for models like Phi
            )
            self.llm = HuggingFacePipeline(pipeline=self.general_generator)
            logger.info(f"Initialized general HuggingFace pipeline with model {model_name_for_llm}.")

            # Specialized generator for concise Google query generation
            self.query_gen_generator = hf_pipeline(
                task="text-generation",
                model=model_name_for_llm, # Use the fetched model name
                max_new_tokens=40,
                device=-1,  # CPU only
                trust_remote_code=True # Added for models like Phi
            )
            self.query_gen_llm = HuggingFacePipeline(pipeline=self.query_gen_generator)
            logger.info(f"Initialized query-focused HuggingFace pipeline with model {model_name_for_llm}.")

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace pipelines with model {model_name_for_llm}: {str(e)}")
            logger.error(traceback.format_exc()) # Log full traceback
            raise

        # Prompt template for query analysis
        self.parser = PydanticOutputParser(pydantic_object=FilterCriteria)
        self.query_analysis_prompt = PromptTemplate(
            input_variables=["query", "current_results", "format_instructions"],
            template=(
                "You are a shopping assistant.\n"
                "Extract filter criteria from the user query and sample results.\n"
                "Return JSON with keys: min_price, max_price, style_keywords, color_preferences, excluded_terms, sort_by.\n"
                "User Query: {query}\n"
                "Sample Results:\n{current_results}\n"
                "{format_instructions}"
            )
        )
        self.filter_extraction_chain = LLMChain(
            llm=self.llm, # Uses general llm
            prompt=self.query_analysis_prompt,
            output_parser=self.parser
        )

        # Prompt template for re-ranking
        self.product_reranking_prompt = PromptTemplate(
            input_variables=["products", "criteria"],
            template=(
                "Given the filtered products and criteria, rank them by relevance.\n"
                "Products (id and title list): {products}\n"
                "Criteria: {criteria}\n"
                "Return a JSON list of product IDs in ranked order."
            )
        )
        self.product_reranking_chain = LLMChain(
            llm=self.llm, # Uses general llm
            prompt=self.product_reranking_prompt
        )

        # Prompt template for Google query generation
        self.google_query_generation_prompt = PromptTemplate(
            input_variables=["text_input"],
            template=(
                "You are an expert at creating concise Google search queries for e-commerce product discovery. "
                "Convert the following input text into a short, effective Google search query. "
                "The query should focus on key visual attributes, product type, and essential keywords. "
                "It must be suitable for a search engine. Output only the query string itself, without any preamble or extra examples.\n\n"
                "Example 1:\n"
                "Input Text: A beautiful long flowing blue summer dress with floral patterns, perfect for weddings and outdoor parties. It is made of cotton.\n"
                "Google Search Query: blue floral cotton summer dress wedding party\n\n"
                "Example 2:\n"
                "Input Text: High-tech noise cancelling headphones, black, wireless, for travel and office use. Excellent battery life.\n"
                "Google Search Query: black wireless noise cancelling travel office headphones\n\n"
                "Example 3:\n"
                "Input Text: a vibrant red t-shirt for men, pure cotton, with a small white embroidered logo on the chest, crew neck, suitable for casual wear.\n"
                "Google Search Query: mens red cotton t-shirt white logo crew neck casual\n\n"
                "Input Text: \"{text_input}\"\n"
                "Google Search Query:"
            )
        )
        self.google_query_generation_chain = LLMChain(
            llm=self.query_gen_llm, # Uses query_gen_llm with shorter max_new_tokens
            prompt=self.google_query_generation_prompt
        )

        logger.info("RefinementService initialized with LLM chains (including few-shot for query gen) and spaCy.")

    def extract_price_range(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract price range from text using spaCy."""
        if not text or not isinstance(text, str): # Added type check
            logger.warning(f"extract_price_range: input text is not a valid string: {text}")
            return None, None

        try:
            doc = self.nlp(text.lower())
            prices = []
            for token in doc:
                if token.like_num:
                    prev = doc[token.i-1].text if token.i > 0 else ""
                    if prev in ['$', '£', '€', '₹']:
                        try:
                            prices.append(float(token.text))
                        except ValueError:
                            continue
            if len(prices) >= 2:
                return min(prices), max(prices)
            if len(prices) == 1:
                txt = text.lower()
                if any(w in txt for w in ['under', 'less than', 'cheaper than']):
                    return None, prices[0]
                if any(w in txt for w in ['over', 'more than', 'above']):
                    return prices[0], None
        except Exception as e:
            logger.error(f"Error extracting price range: {str(e)}")
        return None, None

    def extract_style_keywords(self, text: str) -> List[str]:
        """Extract style keywords from text using spaCy."""
        if not text or not isinstance(text, str): # Added type check
            logger.warning(f"extract_style_keywords: input text is not a valid string: {text}")
            return []

        try:
            doc = self.nlp(text.lower())
            styles = set()
            for token in doc:
                if token.pos_ in ['ADJ', 'NOUN']:
                    lemma = token.lemma_
                    if lemma in ['casual','formal','elegant','modern','vintage','classic','trendy','bohemian','minimalist','luxury']:
                        styles.add(lemma)
            return list(styles)
        except Exception as e:
            logger.error(f"Error extracting style keywords: {str(e)}")
            return []

    def analyze_query(self, query: str, current_results: List[Dict]) -> FilterCriteria:
        """Analyze user query and current results to extract filter criteria."""
        logger.info("analyze_query: Temporarily bypassing LLM filter extraction and returning default criteria.")
        return FilterCriteria() # <-- TEMPORARY SIMPLIFICATION

        # Original logic commented out for now:
        # if not query or not current_results:
        #     return FilterCriteria()

        # try:
        #     # Add unique IDs to products if not present
        #     for p in current_results:
        #         if 'id' not in p:
        #             p['id'] = str(uuid.uuid4())

        #     summary = "\\n".join(f"- {p['title']} ({p.get('price','N/A')})" for p in current_results[:3])
        #     instructions = self.parser.get_format_instructions()
        #     logger.debug("Analyzing query with LLMChain for filter criteria")
            
        #     # This is where the 'str' object does not support item assignment error was happening
        #     raw = self.filter_extraction_chain.run({
        #         'query': query,
        #         'current_results': summary,
        #         'format_instructions': instructions
        #     })
            
        #     logger.debug(f"Raw LLM output for filter criteria: {raw}")

        #     # If the chain already returns a FilterCriteria object due to PydanticOutputParser
        #     if isinstance(raw, FilterCriteria):
        #         criteria = raw
        #     # Else, if it's a dict (e.g., from a different parser setup or if LLM returns dict-like string)
        #     elif isinstance(raw, dict):
        #         criteria = FilterCriteria.parse_obj(raw)
        #     # Else, if it's a string (expected to be JSON)
        #     elif isinstance(raw, str):
        #         try:
        #             parsed_json = json.loads(raw)
        #             criteria = FilterCriteria.parse_obj(parsed_json)
        #         except json.JSONDecodeError:
        #             logger.error(f"Failed to decode JSON from LLM for criteria: {raw}")
        #             raise OutputParserException(f"LLM output for criteria is not valid JSON: {raw}")
        #     else:
        #         logger.error(f"Unexpected output type from LLM chain for criteria: {type(raw)}")
        #         raise OutputParserException(f"Cannot parse criteria from type: {type(raw)}")

        #     logger.info(f"Parsed criteria: {criteria}")
        #     return criteria
        # except OutputParserException as ope: # Catch specific parser errors
        #     logger.error(f"OutputParserException in analyze_query: {str(ope)}")
        # except Exception as e:
        #     logger.error(f"Error in analyze_query LLM section: {str(e)} - Query: '{query}'")
        #     logger.error(traceback.format_exc())


        # # Fallback to basic extraction
        # logger.info("Falling back to basic spaCy keyword extraction for filters.")
        # if not isinstance(query, str): # Defensive check
        #     logger.error(f"analyze_query fallback: query is not a string! Got: {type(query)}. Using empty query.")
        #     query_for_fallback = ""
        # else:
        #     query_for_fallback = query

        # min_p, max_p = self.extract_price_range(query_for_fallback)
        # styles = self.extract_style_keywords(query_for_fallback)
        # return FilterCriteria(
        #     min_price=min_p,
        #     max_price=max_p,
        #     style_keywords=styles,
        #     color_preferences=[],
        #     excluded_terms=[],
        #     sort_by="relevance"
        # )

    def apply_filters(self, products: List[Dict], criteria: FilterCriteria) -> List[Dict]:
        """Apply filter criteria to product list."""
        if not products:
            return []

        try:
            filtered = products[:]
            # Price filters
            if criteria.min_price is not None:
                filtered = [p for p in filtered if p.get('price') is not None and p['price'] >= criteria.min_price]
            if criteria.max_price is not None:
                filtered = [p for p in filtered if p.get('price') is not None and p['price'] <= criteria.max_price]
            # Style/color filters
            if criteria.style_keywords or criteria.color_preferences:
                keys = [k.lower() for k in criteria.style_keywords + criteria.color_preferences]
                filtered = [p for p in filtered if any(k in p['title'].lower() for k in keys)]
            # Exclusions
            if criteria.excluded_terms:
                exclude = [t.lower() for t in criteria.excluded_terms]
                filtered = [p for p in filtered if not any(t in p['title'].lower() for t in exclude)]
            return filtered
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return products

    def rerank(self, filtered: List[Dict], criteria: FilterCriteria) -> List[Dict]:
        """Re-rank filtered products using LLM."""
        if not filtered:
            return []

        # Simple reranking: No LLM if no criteria specified
        if not any([criteria.min_price, criteria.max_price, criteria.style_keywords, criteria.color_preferences, criteria.excluded_terms]):
            logger.info("No specific criteria for re-ranking, returning as is.")
            return filtered
        
        # Prepare data for LLM
        product_info = [{"id": p.get("id", str(uuid.uuid4())), "title": p["title"]} for p in filtered]
        criteria_summary = criteria.json()

        try:
            logger.debug(f"Reranking {len(product_info)} products with criteria: {criteria_summary}")
            ranked_ids_json = self.product_reranking_chain.run({
                "products": json.dumps(product_info),
                "criteria": criteria_summary
            })
            ranked_ids = json.loads(ranked_ids_json) # Expects a list of IDs
            
            # Create a mapping of id to product for efficient reordering
            product_map = {p.get("id"): p for p in filtered}
            
            # Reorder products based on ranked_ids
            re_ranked_products = [product_map[pid] for pid in ranked_ids if pid in product_map]
            
            # Add any products not in ranked_ids (e.g., if LLM missed some) to the end
            # This ensures we don't lose products if LLM output is incomplete
            missing_products = [p for p in filtered if p.get("id") not in ranked_ids]
            re_ranked_products.extend(missing_products)

            logger.info(f"Re-ranked products. Original count: {len(filtered)}, New count: {len(re_ranked_products)}")
            return re_ranked_products
        except Exception as e:
            logger.error(f"Error during LLM re-ranking: {str(e)}. Returning products with original filtering only.")
            return filtered # Fallback to filtered but not re-ranked by LLM

    def generate_shopping_query(self, text_input: str, max_length: int = 25) -> str:
        """Generate a concise Google shopping query from text using LLM."""
        if not text_input:
            return ""
        
        original_text_input_for_fallback = text_input # Keep original for spacy fallback

        try:
            logger.info(f"Generating Google shopping query from text: '{text_input[:100]}...' using LLM.")
            raw_llm_output = self.google_query_generation_chain.run({"text_input": text_input})
            logger.debug(f"Raw LLM output for query generation: '{raw_llm_output}'")

            # Attempt to extract the query after "Google Search Query:"
            # This handles the LLM returning the whole prompt.
            generated_query = ""
            # Split the output by lines. The LLM might return the prompt including examples.
            # We are looking for the line that directly follows "Google Search Query:"
            # which itself follows the *actual* input text.
            lines = raw_llm_output.splitlines()
            
            # Find the block related to the actual text_input
            input_text_marker = f'Input Text: "{text_input}"' # Exact match for our input
            # A more robust but complex way if text_input might be slightly altered by LLM:
            # Find the last "Input Text:" that is NOT one of the hardcoded examples.
            
            found_actual_input_block = False
            for i, line in enumerate(lines):
                if input_text_marker in line:
                    found_actual_input_block = True
                    # Now look for "Google Search Query:" in the subsequent lines of this block
                    for j in range(i + 1, len(lines)):
                        if "Google Search Query:" in lines[j]:
                            parts = lines[j].split("Google Search Query:", 1)
                            if len(parts) > 1 and parts[1].strip():
                                generated_query = parts[1].strip()
                                break # Found query for our specific input
                            # Check if the query is on the very next line
                            elif j + 1 < len(lines) and lines[j+1].strip() and "Input Text:" not in lines[j+1] and "Example " not in lines[j+1]:
                                generated_query = lines[j+1].strip()
                                break # Found query for our specific input
                    break # Stop searching after processing the block for our input
            
            if not generated_query and found_actual_input_block:
                # If we found the input block but not the query immediately after,
                # it might be that the LLM just appended the query.
                # This is a weaker heuristic.
                # Example:
                # Input Text: "actual user input"
                # Google Search Query: 
                # actual generated query for user input
                # We look for the first non-empty line after the "Google Search Query:" associated with our input
                for i, line in enumerate(lines):
                    if input_text_marker in line:
                        for j in range(i + 1, len(lines)):
                            if "Google Search Query:" in lines[j]:
                                if j + 1 < len(lines) and lines[j+1].strip() and "Input Text:" not in lines[j+1] and "Example " not in lines[j+1]:
                                     generated_query = lines[j+1].strip()
                                     break
                        break
            
            if not generated_query: # Fallback if the specific input block parsing didn't work
                # Try the previous broader fallback: last query-like line not part of an example.
                # This is less reliable but better than nothing.
                # Iterate reversed to find the last "Google Search Query: X" pattern
                for i in range(len(lines) - 1, -1, -1):
                    if "Google Search Query:" in lines[i]:
                        parts = lines[i].split("Google Search Query:", 1)
                        if len(parts) > 1 and parts[1].strip() and "Input Text:" not in parts[1] and "Example " not in parts[1]:
                            # Ensure it's not an example query that's part of an "Input Text:" line itself
                            is_part_of_example_input = False
                            if i > 0 and "Input Text:" in lines[i-1] and any(ex_marker in lines[i-1] for ex_marker in ["Example 1:", "Example 2:", "Example 3:"]):
                                is_part_of_example_input = True
                            
                            if not is_part_of_example_input:
                                generated_query = parts[1].strip()
                                break
                        # Check if query is on next line and that next line is not an example's Input Text
                        elif i + 1 < len(lines) and lines[i+1].strip() and "Input Text:" not in lines[i+1] and "Example " not in lines[i+1]:
                            # And also ensure that this "Google Search Query:" line wasn't part of an example's input
                            is_part_of_example_input = False
                            if "Input Text:" in lines[i] and any(ex_marker in lines[i] for ex_marker in ["Example 1:", "Example 2:", "Example 3:"]):
                                is_part_of_example_input = True

                            if not is_part_of_example_input:
                                generated_query = lines[i+1].strip()
                                break
            
            if not generated_query: # Ultimate fallback to previous "last non-empty line"
                non_empty_lines = [line.strip() for line in lines if line.strip()]
                for line in reversed(non_empty_lines):
                    if not line.startswith("Input Text:") and not line.startswith("Google Search Query:") and not line.startswith("Example"):
                        generated_query = line
                        break
                if not generated_query and non_empty_lines:
                     generated_query = non_empty_lines[-1]

            # Clean up common LLM artifacts like quotes around the query
            generated_query = generated_query.strip('\'"')

            if not generated_query:
                logger.warning("LLM generated an empty query.")
            elif len(generated_query.split()) > max_length: # Check word count
                logger.warning(f"LLM generated query is too long ({len(generated_query.split())} words, max {max_length}): '{generated_query}'.")
                generated_query = "" # Invalidate if too long
            else:
                logger.info(f"LLM generated shopping query: '{generated_query}'")
                return generated_query

        except Exception as e:
            logger.error(f"LLM (google_query_generation_chain) failed or produced unusable output for input '{text_input[:100]}': {str(e)}")
        
        # Fallback to basic spaCy keyword extraction if LLM fails or output is unusable
        logger.warning("Falling back to basic spaCy keyword extraction for Google query.")
        try:
            doc = self.nlp(original_text_input_for_fallback) # Use original text input
            keywords = [token.lemma_ for token in doc if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop]
            fallback_query = " ".join(keywords[:max_length // 2]) # Take fewer keywords for fallback
            logger.info(f"Generated Google shopping query with spaCy fallback: '{fallback_query}'")
            return fallback_query.strip()
        except Exception as spacy_e:
            logger.error(f"spaCy fallback also failed: {str(spacy_e)}")
            return original_text_input_for_fallback.strip() # Ultimate fallback

    def refine_results(self, query: str, products: List[Dict]) -> List[Dict]:
        """Main refinement pipeline: analyze query, filter, and rerank products."""
        logger.info(f"refine_results: Received {len(products)} products initially. Query: '{str(query)[:100]}'") # Log length and type of input

        # Temporarily return products immediately to isolate issues
        logger.info(f"refine_results: Temporarily returning {len(products)} products directly without further processing.")
        return products

        # Original logic commented out:
        # if not query or not products:
        #     logger.warning("Refine_results called with empty query or products. Returning products as is.")
        #     return products

        # try:
        #     criteria = self.analyze_query(query, products) 
        #     logger.info(f"refine_results: Criteria after analyze_query: {criteria.dict() if criteria else 'None'}")
            
        #     # Defensive copy before filtering
        #     products_to_filter = [p.copy() for p in products] if products else []
            
        #     filtered = self.apply_filters(products_to_filter, criteria) 
        #     logger.info(f"refine_results: Filtered to {len(filtered)} products.")
            
        #     # Defensive copy before reranking
        #     products_to_rerank = [p.copy() for p in filtered] if filtered else []

        #     final = self.rerank(products_to_rerank, criteria) 
        #     logger.info(f"Final results: {len(final)} products")
        #     return final
        # except Exception as e:
        #     logger.error(f"Error in refine_results: {str(e)}")
        #     logger.error(traceback.format_exc()) 
        #     # Fallback to a fresh copy of the original products list if errors occur
        #     return [p.copy() for p in products] if products else []

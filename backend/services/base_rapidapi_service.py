import os
import requests
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BaseRapidApiService:
    def __init__(self, api_key_env_var: str, host_env_var: str, 
                 endpoint_path_env_var: str, default_endpoint_path: str,
                 text_query_param_env_var: Optional[str] = None,
                 default_text_query_param: str = "q",
                 category_param_env_var: Optional[str] = None, 
                 category_value_format_env_var: Optional[str] = None,
                 service_name: str = "RapidAPI"):
        
        self.api_key = os.getenv(api_key_env_var)
        self.host = os.getenv(host_env_var)
        self.service_name = service_name

        if not self.api_key:
            logger.error(f"{self.service_name}: {api_key_env_var} not found in environment variables.")
            raise EnvironmentError(f"{self.service_name}: {api_key_env_var} not set. Please set it in your .env file.")
        if not self.host:
            logger.error(f"{self.service_name}: {host_env_var} not found in environment variables.")
            raise EnvironmentError(f"{self.service_name}: {host_env_var} not set. Please set it in your .env file.")

        env_endpoint_path = os.getenv(endpoint_path_env_var)
        self.endpoint_path = env_endpoint_path if env_endpoint_path else default_endpoint_path
        
        env_text_query_param = os.getenv(text_query_param_env_var) if text_query_param_env_var else None
        self.text_query_param = env_text_query_param if env_text_query_param else default_text_query_param
        if not self.text_query_param and default_text_query_param:
             logger.info(f"{self.service_name}: Text query parameter name not found via {text_query_param_env_var}, using default: '{default_text_query_param}'")
             self.text_query_param = default_text_query_param
        elif not env_text_query_param and not default_text_query_param :
            logger.info(f"{self.service_name}: No text query parameter configured (env var {text_query_param_env_var} not set, no default). Assumes service may not use text query.")
            self.text_query_param = None

        env_category_param_name = os.getenv(category_param_env_var) if category_param_env_var else None
        self.category_param = env_category_param_name if env_category_param_name else None
        
        if category_value_format_env_var:
            env_category_value_format = os.getenv(category_value_format_env_var)
            self.category_value_format = env_category_value_format if env_category_value_format else "{}"
        else:
            self.category_value_format = "{}"

        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }
        logger.info(
            f"{self.service_name} initialized: Host='{self.host}', "
            f"Endpoint='{self.endpoint_path}', "
            f"TextQueryParam='{self.text_query_param or 'NOT_CONFIGURED'}', "
            f"CategoryParam='{self.category_param or 'NOT_CONFIGURED'}'"
            f"{f' (Format: {self.category_value_format})' if self.category_param else ''}"
        )

    def _parse_item(self, item: Dict) -> Optional[Dict]:
        """
        Parses a single raw product item from an API response into a standardized format.
        This is a flexible parser that tries to find common keys.
        """
        if not isinstance(item, dict):
            return None

        # --- Flexible ID Extraction ---
        id_keys = ['asin', 'product_id', 'item_id', 'id']
        product_id = None
        for key in id_keys:
            if item.get(key):
                product_id = str(item.get(key)) # Ensure ID is a string
                break
        
        # --- Flexible Title Extraction ---
        title_keys = ['product_title', 'title', 'name']
        title = "No Title"
        for key in title_keys:
            if item.get(key):
                title = item.get(key)
                break

        # --- Flexible Photo Extraction ---
        photo_keys = ['product_photo', 'image', 'thumbnail', 'main_image_url']
        photo = None
        for key in photo_keys:
            if item.get(key):
                photo = item.get(key)
                break
        
        # --- Flexible Product URL Extraction ---
        url_keys = ['product_url', 'url', 'link']
        product_url = None
        for key in url_keys:
            if item.get(key):
                product_url = item.get(key)
                break
        
        # --- Flexible Price & Currency Extraction ---
        price = None
        currency = None
        price_str = None

        # A list of possible keys for price, ordered by preference
        price_keys = [
            'product_price', 'price_string', 'product_sale_price', 'sale_price', 
            'current_price', 'price'
        ]

        for key in price_keys:
            if item.get(key):
                price_candidate = item.get(key)
                if isinstance(price_candidate, dict):
                    # Handle cases like "price": {"value": 12.99, "currency": "USD"}
                    value_keys = ['value', 'amount', 'current_price']
                    for v_key in value_keys:
                        if price_candidate.get(v_key):
                            price = price_candidate.get(v_key)
                            break
                    if price and 'currency' in price_candidate:
                        currency = price_candidate.get('currency')
                    break # Found a price dict, stop searching keys
                elif isinstance(price_candidate, (str, int, float)):
                    price_str = str(price_candidate)
                    break # Found a price string/number, stop searching keys
        
        # If we found a price string, parse it
        if price_str:
            # Detect currency symbol first
            currency_symbols = {'$': 'USD', '₹': 'INR', '€': 'EUR', '£': 'GBP', 'Rs': 'INR', 'INR': 'INR'}
            for symbol, code in currency_symbols.items():
                if symbol in price_str:
                    currency = code
                    break
            
            # Remove symbols and characters to get a clean number
            cleaned_price_str = price_str
            for symbol in currency_symbols.keys():
                cleaned_price_str = cleaned_price_str.replace(symbol, '')
            cleaned_price_str = cleaned_price_str.replace(',', '').strip()

            try:
                price = float(cleaned_price_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse price string '{price_str}' to float for item '{title}'.")
                price = None
        
        # Final check for numeric but non-string price from the item root
        if price is None and 'price' in item and isinstance(item['price'], (int, float)):
            price = item['price']

        # Only return a product if it has the essentials
        if not product_id or not photo:
            logger.debug(f"Skipping item due to missing ID or Photo. Title: '{title}'")
            return None

        return {
            "id": product_id,
            "title": title,
            "photo_url": photo,
            "price": price,
            "currency": currency,
            "product_url": product_url,
            "source": self.service_name
        }

    def search_products(self, query: Optional[str] = None, limit: int = 20, category_search_term: Optional[str] = None) -> List[Dict]:
        url = f"https://{self.host}{self.endpoint_path}"
        headers = {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": self.host}
        params = {
            "country": "in",
            "limit": str(limit)
        }

        if query and self.text_query_param:
            params[self.text_query_param] = query
            logger.info(f"{self.service_name}: Using text query param '{self.text_query_param}' with query '{query}'.")
        elif query:
            logger.warning(f"{self.service_name}: Text query '{query}' provided, but no text query parameter is configured for this service. Text query will be ignored.")

        if category_search_term and self.category_param and not self.category_param.startswith("# BLANK"):
            formatted_category_search_term = self.category_value_format.format(category_search_term)
            params[self.category_param] = formatted_category_search_term
            logger.info(f"{self.service_name}: Adding category filter: {self.category_param}='{formatted_category_search_term}'")
        elif category_search_term:
            logger.info(f"{self.service_name}: Category hint '{category_search_term}' provided, but category search is not configured or param name is a placeholder ('{self.category_param}'). Category hint will be ignored for param sending.")

        if not params:
            logger.warning(f"{self.service_name}: No query parameters formed (neither text query nor category term applicable/configured). Aborting API call to {url}.")
            return []
            
        logger.debug(f"{self.service_name}: Calling API: URL={url}, Params={params}, HeaderKeys={[k for k in headers.keys()]}")
        
        response_text = "No response text available"
        response_url = url 

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response_url = response.url 
            response_text = response.text 
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"{self.service_name} HTTP error: {http_err} - URL: {response_url} - Response: {response_text}")
            return []
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"{self.service_name} Connection error: {conn_err} - Attempted URL: {url}")
            return []
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"{self.service_name} Timeout error: {timeout_err} - Attempted URL: {url}")
            return []
        except requests.exceptions.RequestException as req_err:
            logger.error(f"{self.service_name} Request error: {req_err} - Attempted URL: {url}")
            return []
        except ValueError as json_err: 
            logger.error(f"{self.service_name} JSON decoding error: {json_err} - URL: {response_url} - Response: {response_text}")
            return []

        results = []
        raw_products_list_keys = ["products", "results", "items", "data", "search_results", "searchProductDTOResponse"]
        raw_products = None

        if isinstance(data, list):
            raw_products = data
        elif isinstance(data, dict):
            for key in raw_products_list_keys:
                if key in data and isinstance(data[key], list):
                    raw_products = data[key]
                    break
            if raw_products is None and 'data' in data and isinstance(data['data'], dict):
                 for key in raw_products_list_keys:
                    if key in data['data'] and isinstance(data['data'][key], list):
                        raw_products = data['data'][key]
                        logger.info(f"{self.service_name}: Found products list under data['{key}']")
                        break
        
        if raw_products is None:
            logger.warning(f"{self.service_name}: Products list not found or not a list (tried common keys or direct list). Data: {str(data)[:500]}...")
            return []
            
        for item in raw_products:
            parsed_item = self._parse_item(item)
            if parsed_item:
                results.append(parsed_item)
                
        logger.info(f"{self.service_name}: Processed {len(results)} products for query='{query if query else 'N/A'}', category='{category_search_term if category_search_term else 'N/A'}'")
        return results 

    def search(self, query: str, **kwargs) -> List[Dict]:
        """
        A standardized search method that uses make_request.
        'query' is mapped to the service's specific text query parameter.
        Other kwargs are passed through.
        """
        if not self.text_query_param:
            logger.warning(f"{self.service_name}: 'search' called but no text query parameter is configured. Returning empty list.")
            return []
            
        params = {self.text_query_param: query}
        params.update(kwargs) # Add any other params like 'country', 'sort_by', etc.
        
        return self.make_request(params=params)

    def make_request(self, params: Dict) -> List[Dict]:
        """
        Makes a request to the configured RapidAPI endpoint.
        """
        url = f"https://{self.host}{self.endpoint_path}"
        headers = self.headers
        logger.debug(f"{self.service_name}: Calling API: URL={url}, Params={params}, HeaderKeys={[k for k in headers.keys()]}")
        
        response_text = "No response text available"
        response_url = url 

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response_url = response.url 
            response_text = response.text 
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"{self.service_name} HTTP error: {http_err} - URL: {response_url} - Response: {response_text}")
            return []
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"{self.service_name} Connection error: {conn_err} - Attempted URL: {url}")
            return []
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"{self.service_name} Timeout error: {timeout_err} - Attempted URL: {url}")
            return []
        except requests.exceptions.RequestException as req_err:
            logger.error(f"{self.service_name} Request error: {req_err} - Attempted URL: {url}")
            return []
        except ValueError as json_err: 
            logger.error(f"{self.service_name} JSON decoding error: {json_err} - URL: {response_url} - Response: {response_text}")
            return []

        results = []
        raw_products_list_keys = ["products", "results", "items", "data", "search_results", "searchProductDTOResponse"]
        raw_products = None

        if isinstance(data, list):
            raw_products = data
        elif isinstance(data, dict):
            for key in raw_products_list_keys:
                if key in data and isinstance(data[key], list):
                    raw_products = data[key]
                    break
            if raw_products is None and 'data' in data and isinstance(data['data'], dict):
                 for key in raw_products_list_keys:
                    if key in data['data'] and isinstance(data['data'][key], list):
                        raw_products = data['data'][key]
                        logger.info(f"{self.service_name}: Found products list under data['{key}']")
                        break
        
        if raw_products is None:
            logger.warning(f"{self.service_name}: Products list not found or not a list (tried common keys or direct list). Data: {str(data)[:500]}...")
            return []
            
        for item in raw_products:
            parsed_item = self._parse_item(item)
            if parsed_item:
                results.append(parsed_item)
                
        query_val = params.get(self.text_query_param, 'N/A')
        logger.info(f"{self.service_name}: Processed {len(results)} products for query='{query_val}'")
        return results 
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
        self.text_query_param_name = env_text_query_param if env_text_query_param else default_text_query_param
        if not self.text_query_param_name and default_text_query_param:
             logger.info(f"{self.service_name}: Text query parameter name not found via {text_query_param_env_var}, using default: '{default_text_query_param}'")
             self.text_query_param_name = default_text_query_param
        elif not env_text_query_param and not default_text_query_param :
            logger.info(f"{self.service_name}: No text query parameter configured (env var {text_query_param_env_var} not set, no default). Assumes service may not use text query.")
            self.text_query_param_name = None

        env_category_param_name = os.getenv(category_param_env_var) if category_param_env_var else None
        self.category_param_name = env_category_param_name if env_category_param_name else None
        
        if category_value_format_env_var:
            env_category_value_format = os.getenv(category_value_format_env_var)
            self.category_value_format = env_category_value_format if env_category_value_format else "{}"
        else:
            self.category_value_format = "{}"

        log_parts = [f"{self.service_name} initialized: Host='{self.host}', Endpoint='{self.endpoint_path}'"]
        if self.text_query_param_name:
            log_parts.append(f"TextQueryParam='{self.text_query_param_name}'")
        if self.category_param_name:
            log_parts.append(f"CategoryParam='{self.category_param_name}' (Format: '{self.category_value_format}')")
        else:
            log_parts.append("CategoryFiltering=NOT_CONFIGURED")
        logger.info(", ".join(log_parts))

    def _parse_item(self, item: Dict) -> Optional[Dict]:
        """
        Parses a single item from the API response into the standardized product schema.
        This method SHOULD be overridden by subclasses if the API response structure is unique.
        """
        # Ensure basic structure for a product item
        if not isinstance(item, dict):
            logger.debug(f"{self.service_name}: Item is not a dictionary, skipping: {item}")
            return None

        title = None
        product_url = None
        image_url = None
        price_str = None
        currency = None
        rating_str = None

        # Prioritize specific Amazon fields if this is an Amazon service
        is_amazon_service = "amazon" in self.service_name.lower()

        if is_amazon_service:
            title = item.get("product_title")
            product_url = item.get("product_url")
            image_url = item.get("product_photo")
            price_str = item.get("product_price")
            currency = item.get("currency")
            rating_str = item.get("product_star_rating")
        
        # Fallback to common field names if not found or not an Amazon service
        if title is None:
            title = item.get("name") or item.get("title") or item.get("product_name") or item.get("product_title") # Keep product_title as general fallback too
        if product_url is None:
            product_url = item.get("url") or item.get("link") or item.get("product_url") or item.get("item_url") or item.get("product_link") # Keep product_url
        if image_url is None:
            image_url = item.get("image") or item.get("image_url") or item.get("product_photo") or item.get("image_link") or item.get("picture_url")
        if price_str is None:
            price_str = item.get("price") or item.get("product_price") or item.get("current_price")
        if currency is None:
            currency = item.get("currency") or item.get("price_currency")
        if rating_str is None:
            rating_str = item.get("rating") or item.get("product_star_rating") or item.get("star_rating") or item.get("average_rating")

        price_info = price_str
        current_price = None

        if isinstance(price_info, dict):
            current_price = price_info.get("value") or price_info.get("amount") or price_info.get("current")
            currency = price_info.get("currency", currency)
        elif isinstance(price_info, (int, float)):
            current_price = price_info
        elif isinstance(price_info, str):
            try:
                current_price = float("".join(filter(lambda c: c.isdigit() or c == '.', price_info)))
            except ValueError:
                pass
        
        rating_value = None
        if isinstance(rating_str, dict):
            rating_value = rating_str.get("value") or rating_str.get("average") or rating_str.get("score") or rating_str.get("star_rating")
            if rating_value is None and "rate" in rating_str: rating_value = rating_str.get("rate")
        elif isinstance(rating_str, (int, float, str)):
            try:
                rating_value = float(str(rating_str).split()[0])
            except ValueError:
                pass

        if not (title and product_url):
            logger.debug(f"{self.service_name}: Skipping item due to missing title or product_url: {str(item)[:200]}...")
            return None

        return {
            "title": title,
            "price": current_price,
            "currency": currency,
            "image_url": image_url,
            "product_url": product_url,
            "rating": rating_value,
            "source": item.get("source") or item.get("store") or item.get("seller_name") or self.host
        }

    def search_products(self, query: Optional[str] = None, limit: int = 20, category_search_term: Optional[str] = None) -> List[Dict]:
        url = f"https://{self.host}{self.endpoint_path}"
        headers = {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": self.host}
        params = {
            "country": "in",
            "limit": str(limit)
        }

        if query and self.text_query_param_name:
            params[self.text_query_param_name] = query
            logger.info(f"{self.service_name}: Using text query param '{self.text_query_param_name}' with query '{query}'.")
        elif query:
            logger.warning(f"{self.service_name}: Text query '{query}' provided, but no text query parameter is configured for this service. Text query will be ignored.")

        if category_search_term and self.category_param_name and not self.category_param_name.startswith("# BLANK"):
            formatted_category_search_term = self.category_value_format.format(category_search_term)
            params[self.category_param_name] = formatted_category_search_term
            logger.info(f"{self.service_name}: Adding category filter: {self.category_param_name}='{formatted_category_search_term}'")
        elif category_search_term:
            logger.info(f"{self.service_name}: Category hint '{category_search_term}' provided, but category search is not configured or param name is a placeholder ('{self.category_param_name}'). Category hint will be ignored for param sending.")

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
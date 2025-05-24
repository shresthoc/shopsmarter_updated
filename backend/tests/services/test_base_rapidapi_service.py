import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import requests
import logging

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, backend_dir)

from services.base_rapidapi_service import BaseRapidApiService

class TestBaseRapidApiService(unittest.TestCase):

    def get_default_env(self):
        return {
            "TEST_API_KEY": "test_key",
            "TEST_HOST": "test.api.host",
            "TEST_ENDPOINT_PATH": "/test/search",
            "TEST_CATEGORY_PARAM": "cat_param",
            "TEST_CATEGORY_FORMAT": "fmt:{}"
        }

    @patch.dict(os.environ, get_default_env({})) # Load default env
    def setUp(self):
        self.service = BaseRapidApiService(
            api_key_env_var="TEST_API_KEY",
            host_env_var="TEST_HOST",
            endpoint_path_env_var="TEST_ENDPOINT_PATH",
            default_endpoint_path="/default/path",
            category_param_env_var="TEST_CATEGORY_PARAM",
            category_value_format_env_var="TEST_CATEGORY_FORMAT",
            service_name="TestBaseService"
        )
        self.test_url = "https://test.api.host/test/search"

    def test_init_success(self):
        self.assertEqual(self.service.api_key, "test_key")
        self.assertEqual(self.service.host, "test.api.host")
        self.assertEqual(self.service.endpoint_path, "/test/search")
        self.assertEqual(self.service.category_param_name, "cat_param")
        self.assertEqual(self.service.category_value_format, "fmt:{}")
        self.assertEqual(self.service.service_name, "TestBaseService")

    def test_init_default_endpoint_and_category_format(self):
        # Test when specific env vars for endpoint path and cat format are not set
        with patch.dict(os.environ, {"TEST_API_KEY": "k", "TEST_HOST": "h", "TEST_ENDPOINT_PATH": "", "TEST_CATEGORY_FORMAT": ""}):
            service = BaseRapidApiService(
                api_key_env_var="TEST_API_KEY", host_env_var="TEST_HOST",
                endpoint_path_env_var="TEST_ENDPOINT_PATH", default_endpoint_path="/default/ep",
                category_param_env_var="TEST_CATEGORY_PARAM", # cat_param still set
                category_value_format_env_var="TEST_CATEGORY_FORMAT",
                service_name="TestDefault"
            )
            self.assertEqual(service.endpoint_path, "/default/ep")
            self.assertEqual(service.category_value_format, "{}") # Default format
    
    def test_init_no_category_param_env_var_defined(self):
         with patch.dict(os.environ, {"TEST_API_KEY": "k", "TEST_HOST": "h"}): # TEST_CATEGORY_PARAM not in env
            service = BaseRapidApiService(
                api_key_env_var="TEST_API_KEY", host_env_var="TEST_HOST",
                endpoint_path_env_var="ANY_EP_PATH_VAR", default_endpoint_path="/d_ep",
                category_param_env_var="NON_EXISTENT_CAT_PARAM_VAR", # This env var does not exist
                service_name="TestNoCat"
            )
            self.assertIsNone(service.category_param_name)
            self.assertEqual(service.category_value_format, "{}")

    def test_init_missing_api_key(self):
        with patch.dict(os.environ, {"TEST_API_KEY": "", "TEST_HOST": "h"}):
            with self.assertRaisesRegex(EnvironmentError, "TEST_API_KEY not set"):
                BaseRapidApiService("TEST_API_KEY", "TEST_HOST", "EP", "/dep")

    def test_init_missing_host(self):
        with patch.dict(os.environ, {"TEST_API_KEY": "k", "TEST_HOST": ""}):
            with self.assertRaisesRegex(EnvironmentError, "TEST_HOST not set"):
                BaseRapidApiService("TEST_API_KEY", "TEST_HOST", "EP", "/dep")

    @patch('services.base_rapidapi_service.requests.get')
    def test_search_products_success_with_category(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [ # Using 'results' as a common key
            {"name": "Product 1", "link": "http://p1.com", "price": 100},
            {"title": "Product 2", "product_url": "http://p2.com", "price": {"value": 200}} 
        ]}
        mock_get.return_value = mock_response
        
        query = "test search"
        cat_term = "electronics"
        products = self.service.search_products(query, limit=5, category_search_term=cat_term)
        
        self.assertEqual(len(products), 2)
        self.assertEqual(products[0]["title"], "Product 1")
        self.assertEqual(products[1]["price"], 200)
        
        mock_get.assert_called_once_with(
            self.test_url,
            headers={"X-RapidAPI-Key": "test_key", "X-RapidAPI-Host": "test.api.host"},
            params={"q": query, "country": "in", "limit": "5", "cat_param": "fmt:electronics"},
            timeout=15
        )

    @patch('services.base_rapidapi_service.requests.get')
    def test_search_products_no_category_param_configured(self, mock_get):
        # Reconfigure service instance to have no category_param_name
        with patch.dict(os.environ, {"TEST_API_KEY": "k", "TEST_HOST": "h", "TEST_CATEGORY_PARAM": ""}): # Param is empty
            service_no_cat = BaseRapidApiService(
                "TEST_API_KEY", "TEST_HOST", "EP", "/dep", "TEST_CATEGORY_PARAM"
            )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"products": [{"title": "Prod X", "url": "http://x.com"}]}
        mock_get.return_value = mock_response
        
        query = "query no cat"
        cat_term = "ignored_cat"
        products = service_no_cat.search_products(query, category_search_term=cat_term)
        
        self.assertEqual(len(products), 1)
        # Verify that cat_param was NOT in the call
        mock_get.assert_called_once_with(
            f"https://h/dep", # Uses default_endpoint_path as EP env var is not "set" to anything here for this service
            headers={"X-RapidAPI-Key": "k", "X-RapidAPI-Host": "h"},
            params={"q": query, "country": "in", "limit": "20"}, # No cat_param
            timeout=15
        )

    @patch('services.base_rapidapi_service.requests.get')
    @patch('services.base_rapidapi_service.logger') # Patch the module-level logger
    def test_search_products_api_error_handling(self, mock_module_logger, mock_get): # mock_module_logger is now an arg

        error_map = {
            requests.exceptions.HTTPError("http err"): MagicMock(status_code=500, text="server error", raise_for_status=MagicMock(side_effect=requests.exceptions.HTTPError("http err"))),
            requests.exceptions.ConnectionError("conn err"): MagicMock(side_effect=requests.exceptions.ConnectionError("conn err")),
            requests.exceptions.Timeout("timeout err"): MagicMock(side_effect=requests.exceptions.Timeout("timeout err")),
            requests.exceptions.RequestException("req err"): MagicMock(side_effect=requests.exceptions.RequestException("req err")),
            ValueError("json err"): MagicMock(status_code=200, json=MagicMock(side_effect=ValueError("json err")), text="bad json"),
        }
        for error_type, mock_behavior in error_map.items():
            mock_get.reset_mock()
            # Reset the call state of the error method on our mock_module_logger for each iteration
            mock_module_logger.error.reset_mock() 
            
            if isinstance(error_type, ValueError): # For JSON error
                mock_get.return_value = mock_behavior
            else: # For request exceptions
                mock_get.side_effect = error_type
            
            products = self.service.search_products("test")
            self.assertEqual(products, [])
            
            # Check that mock_module_logger.error was called
            self.assertTrue(mock_module_logger.error.called, 
                            f"Expected logger.error to be called for {type(error_type).__name__}. It was not.")
            # Optionally, check the content of the log message if needed by inspecting mock_module_logger.error.call_args
            # Example: self.assertIn(str(error_type).lower(), mock_module_logger.error.call_args[0][0].lower())
        # The finally block for restoring logger is removed. Patching handles cleanup.

    def test_parse_item_standard_structure(self):
        item = {
            "name": "Awesome Gadget",
            "price": {"value": 199.99, "currency": "USD"},
            "image": "http://img.com/gadget.png",
            "link": "http://store.com/gadget",
            "rating": {"average": 4.5, "count": 150},
            "store": "GadgetPlace"
        }
        parsed = self.service._parse_item(item)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "Awesome Gadget")
        self.assertEqual(parsed["price"], 199.99)
        self.assertEqual(parsed["currency"], "USD")
        self.assertEqual(parsed["image_url"], "http://img.com/gadget.png")
        self.assertEqual(parsed["product_url"], "http://store.com/gadget")
        self.assertEqual(parsed["rating"], 4.5)
        self.assertEqual(parsed["source"], "GadgetPlace") # Prefers item source over host

    def test_parse_item_missing_title_or_url(self):
        self.assertIsNone(self.service._parse_item({"price": 10, "link": "http://url.com"})) # Missing title
        self.assertIsNone(self.service._parse_item({"name": "Title", "price": 10})) # Missing URL

    @patch('services.base_rapidapi_service.requests.get')
    def test_search_products_direct_list_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        # API returns a list directly
        mock_response.json.return_value = [
            {"title": "Item 1", "product_url": "http://i1.com"},
            {"name": "Item 2", "link": "http://i2.com"}
        ]
        mock_get.return_value = mock_response
        products = self.service.search_products("direct list query")
        self.assertEqual(len(products), 2)
        self.assertEqual(products[0]["title"], "Item 1")
        self.assertEqual(products[1]["title"], "Item 2")

    @patch('services.base_rapidapi_service.requests.get')
    def test_search_products_various_product_list_keys(self, mock_get):
        list_keys_to_test = ["products", "results", "items", "data", "search_results"]
        for key in list_keys_to_test:
            mock_get.reset_mock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {key: [{"title": f"Prod via {key}", "link": "http://some.url"}]}
            mock_get.return_value = mock_response
            
            products = self.service.search_products(f"query for {key}")
            self.assertEqual(len(products), 1)
            self.assertEqual(products[0]["title"], f"Prod via {key}")

if __name__ == '__main__':
    unittest.main() 
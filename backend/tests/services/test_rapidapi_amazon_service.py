import unittest
from unittest.mock import patch
import os
import sys
import logging

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, backend_dir)

from services.rapidapi_amazon_service import RapidApiAmazonService

logging.disable(logging.CRITICAL)

class TestRapidApiAmazonService(unittest.TestCase):

    @patch.dict(os.environ, {
        "RAPIDAPI_KEY": "shared_key_for_amazon",
        "RAPIDAPI_AMAZON_HOST": "amazon.api.host",
        "RAPIDAPI_AMAZON_ENDPOINT_PATH": "/amazon/products",
        "RAPIDAPI_AMAZON_CATEGORY_PARAM": "amazon_cat",
        "RAPIDAPI_AMAZON_CATEGORY_VALUE_FORMAT": "amz:{}"
    })
    def test_init_amazon_service(self):
        service = RapidApiAmazonService()
        self.assertEqual(service.api_key, "shared_key_for_amazon")
        self.assertEqual(service.host, "amazon.api.host")
        self.assertEqual(service.endpoint_path, "/amazon/products")
        self.assertEqual(service.category_param_name, "amazon_cat")
        self.assertEqual(service.category_value_format, "amz:{}")
        self.assertEqual(service.service_name, "RapidAPI-Amazon")

    @patch.dict(os.environ, {
        "RAPIDAPI_KEY": "key",
        "RAPIDAPI_AMAZON_HOST": "host",
        "RAPIDAPI_AMAZON_ENDPOINT_PATH": "", # Test default endpoint path
        "RAPIDAPI_AMAZON_CATEGORY_PARAM": "", # Test category param being disabled (empty)
        "RAPIDAPI_AMAZON_CATEGORY_VALUE_FORMAT": "" # Test default format
    })
    def test_init_amazon_service_minimal_env(self):
        service = RapidApiAmazonService()
        self.assertEqual(service.api_key, "key")
        self.assertEqual(service.host, "host")
        self.assertEqual(service.endpoint_path, "/product/search") # Default endpoint
        self.assertIsNone(service.category_param_name) # Should be None if env var is empty
        self.assertEqual(service.category_value_format, "{}") # Default

if __name__ == '__main__':
    unittest.main() 
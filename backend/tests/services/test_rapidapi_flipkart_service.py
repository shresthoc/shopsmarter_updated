import unittest
from unittest.mock import patch
import os
import sys
import logging

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, backend_dir)

from services.rapidapi_flipkart_service import RapidApiFlipkartService

logging.disable(logging.CRITICAL)

class TestRapidApiFlipkartService(unittest.TestCase):

    @patch.dict(os.environ, {
        "RAPIDAPI_KEY": "shared_key_for_flipkart",
        "RAPIDAPI_FLIPKART_HOST": "flipkart.api.host",
        "RAPIDAPI_FLIPKART_ENDPOINT_PATH": "/flipkart/items",
        "RAPIDAPI_FLIPKART_CATEGORY_PARAM": "flipkart_cat_name",
        "RAPIDAPI_FLIPKART_CATEGORY_VALUE_FORMAT": "fk:{}"
    })
    def test_init_flipkart_service(self):
        service = RapidApiFlipkartService()
        self.assertEqual(service.api_key, "shared_key_for_flipkart")
        self.assertEqual(service.host, "flipkart.api.host")
        self.assertEqual(service.endpoint_path, "/flipkart/items")
        self.assertEqual(service.category_param_name, "flipkart_cat_name")
        self.assertEqual(service.category_value_format, "fk:{}")
        self.assertEqual(service.service_name, "RapidAPI-Flipkart")

    @patch.dict(os.environ, {
        "RAPIDAPI_KEY": "key_fk",
        "RAPIDAPI_FLIPKART_HOST": "host_fk",
        "RAPIDAPI_FLIPKART_ENDPOINT_PATH": "", # Test default endpoint path
        "RAPIDAPI_FLIPKART_CATEGORY_PARAM": "", # Test category param being disabled (empty)
        "RAPIDAPI_FLIPKART_CATEGORY_VALUE_FORMAT": "" # Test default format
    })
    def test_init_flipkart_service_minimal_env(self):
        service = RapidApiFlipkartService()
        self.assertEqual(service.api_key, "key_fk")
        self.assertEqual(service.host, "host_fk")
        self.assertEqual(service.endpoint_path, "/product/search") # Default endpoint
        self.assertIsNone(service.category_param_name) # Should be None if env var is empty
        self.assertEqual(service.category_value_format, "{}") # Default

if __name__ == '__main__':
    unittest.main() 
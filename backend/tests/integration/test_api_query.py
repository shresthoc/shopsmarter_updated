import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
from io import BytesIO
import base64 # For creating a minimal PNG

# Add the parent directory (backend) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, '..', '..')) 
sys.path.insert(0, backend_dir)

# Import app after sys.path is modified
from app import app 

def _create_dummy_image_bytes() -> BytesIO:
    """Creates a BytesIO object containing a minimal 1x1 transparent PNG."""
    # 1x1 transparent PNG, base64 encoded
    # (Original: \\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01\\x08\\x06\\x00\\x00\\x00\\x1f\\x15\\xc4\\x89\\x00\\x00\\x00\\nIDAT\\x08\\xd7c`\\x00\\x00\\x00\\x02\\x00\\x01\\xe2!\\xb3X\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82)
    # For safety and to avoid issues with characters, using base64 representation
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    return BytesIO(base64.b64decode(png_b64))

@patch.dict(os.environ, {
    "RAPIDAPI_KEY": "fake_rapidapi_key",
    # Amazon specific mock env vars
    "RAPIDAPI_AMAZON_HOST": "mock.amazon.rapidapi.host",
    "RAPIDAPI_AMAZON_ENDPOINT_PATH": "/mock/amazon/search",
    "RAPIDAPI_AMAZON_TEXT_QUERY_PARAM": "query_amz",
    "RAPIDAPI_AMAZON_CATEGORY_PARAM": "cat_amz", 
    "RAPIDAPI_AMAZON_CATEGORY_VALUE_FORMAT": "{}",
    # Flipkart specific mock env vars
    "RAPIDAPI_FLIPKART_HOST": "mock.flipkart.rapidapi.host",
    "RAPIDAPI_FLIPKART_ENDPOINT_PATH": "/mock/flipkart/category_search",
    "RAPIDAPI_FLIPKART_TEXT_QUERY_PARAM": "query_fk", # For potential secondary text search on Flipkart
    "RAPIDAPI_FLIPKART_CATEGORY_PARAM": "categoryId_fk",
    "RAPIDAPI_FLIPKART_CATEGORY_VALUE_FORMAT": "{}",

    "FLASK_ENV": "testing",
    "TAGS_JSON": os.path.join(backend_dir, 'tests', 'mock_tags.json')
})
class TestApiQueryIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mock_tags_path = os.path.join(backend_dir, 'tests', 'mock_tags.json')
        mock_tags_content = ["mock_tag_1", "mock_tag_2", "mock_tag_3"] # Simple list of strings
        if not os.path.exists(os.path.dirname(mock_tags_path)):
             os.makedirs(os.path.dirname(mock_tags_path), exist_ok=True)
        with open(mock_tags_path, 'w') as f:
            json.dump(mock_tags_content, f)

    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        
        self.rapidapi_amazon_patch = patch('app.rapidapi_amazon_service', spec=True)
        self.rapidapi_flipkart_patch = patch('app.rapidapi_flipkart_service', spec=True)
        self.refinement_service_patch = patch('app.refinement_service', spec=True)
        self.embedding_service_patch = patch('app.embedding_service', spec=True)
        self.blip_model_patch = patch('app.blip_model')
        self.blip_processor_patch = patch('app.blip_processor')

        self.mock_rapidapi_amazon_service = self.rapidapi_amazon_patch.start()
        self.mock_rapidapi_flipkart_service = self.rapidapi_flipkart_patch.start()
        self.mock_refinement_service = self.refinement_service_patch.start()
        self.mock_embedding_service = self.embedding_service_patch.start()
        self.mock_blip_model = self.blip_model_patch.start()
        self.mock_blip_processor = self.blip_processor_patch.start()

        # Default mock behaviors
        self.mock_blip_processor.decode.return_value = "blip image caption"
        self.mock_blip_model.generate.return_value = MagicMock()
        self.mock_embedding_service.get_image_tags.return_value = ["tagA", "tagB"]
        self.mock_refinement_service.generate_shopping_query.side_effect = lambda query: f"refined_{query}"
        self.mock_refinement_service.refine_results.side_effect = lambda prods, context_str: prods[:min(len(prods), 10)]

    def tearDown(self):
        self.rapidapi_amazon_patch.stop()
        self.rapidapi_flipkart_patch.stop()
        self.refinement_service_patch.stop()
        self.embedding_service_patch.stop()
        self.blip_model_patch.stop()
        self.blip_processor_patch.stop()

    def _get_mock_product(self, i, source_api_name, title_prefix="Mock Product"):
        # source_api_name will be like "RapidAPI-Amazon" or "RapidAPI-Flipkart"
        return {
            "title": f"{title_prefix} {i} from {source_api_name}",
            "price": 100.00 + float(i),
            "currency": "INR",
            "image_url": f"http://example.com/{source_api_name.lower().replace('-', '')}_img{i}.jpg",
            "product_url": f"http://example.com/{source_api_name.lower().replace('-', '')}_prod{i}",
            "rating": 4.0 + (float(i) / 10.0),
            "source": f"mock.{source_api_name.split('-')[-1].lower()}.rapidapi.host" 
        }

    def test_query_with_prompt_both_services_called(self):
        """Test /api/query with prompt, ensuring both Amazon and Flipkart services are called correctly."""
        mock_amazon_prods = [self._get_mock_product(1, "RapidAPI-Amazon")]
        mock_flipkart_prods = [self._get_mock_product(1, "RapidAPI-Flipkart"), self._get_mock_product(2, "RapidAPI-Flipkart")]
        
        self.mock_rapidapi_amazon_service.search_products.return_value = mock_amazon_prods
        self.mock_rapidapi_flipkart_service.search_products.return_value = mock_flipkart_prods
        
        expected_category_tags = ["tagA", "tagB"]
        expected_category_hint_for_api = "tagA tagB"
        self.mock_embedding_service.get_image_tags.return_value = expected_category_tags

        user_prompt = 'stylish blue jeans'
        refined_user_query = f"refined_{user_prompt}"
        img_byte_arr = _create_dummy_image_bytes()
        data = {'image': (img_byte_arr, 'test_jeans.jpg'), 'prompt': user_prompt}
        
        response = self.client.post('/api/query', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data.decode('utf-8'))

        self.assertIn('products', response_data)
        # Total products after merge and dedupe (assuming no duplicates in this simple case for count check)
        self.assertEqual(len(response_data['products']), len(mock_amazon_prods) + len(mock_flipkart_prods)) 
        self.assertEqual(response_data['search_query_used'], refined_user_query)
        self.assertEqual(response_data['category_hint_used'], expected_category_hint_for_api)
        
        self.mock_embedding_service.get_image_tags.assert_called_once_with(unittest.mock.ANY, top_k=2)
        self.mock_refinement_service.generate_shopping_query.assert_called_with(user_prompt)
        
        # Amazon called with text query and category hint
        self.mock_rapidapi_amazon_service.search_products.assert_called_once_with(
            query=refined_user_query, 
            limit=15, 
            category_search_term=expected_category_hint_for_api
        )
        # Flipkart called with category hint and text query (as per app.py logic)
        self.mock_rapidapi_flipkart_service.search_products.assert_called_once_with(
            query=refined_user_query, 
            limit=15, 
            category_search_term=expected_category_hint_for_api
        )
        self.mock_refinement_service.refine_results.assert_called_once()

    def test_query_with_blip_caption_both_services_called(self):
        mock_amazon_prods = [self._get_mock_product(10, "RapidAPI-Amazon")]
        mock_flipkart_prods = [self._get_mock_product(20, "RapidAPI-Flipkart")]
        self.mock_rapidapi_amazon_service.search_products.return_value = mock_amazon_prods
        self.mock_rapidapi_flipkart_service.search_products.return_value = mock_flipkart_prods
        
        blip_caption = "blip image caption"
        refined_blip_query = f"refined_{blip_caption}"
        expected_category_tags = ["tagA", "tagB"]
        expected_category_hint_for_api = "tagA tagB"
        self.mock_embedding_service.get_image_tags.return_value = expected_category_tags

        img_byte_arr = _create_dummy_image_bytes()
        data = {'image': (img_byte_arr, 'test_blip.jpg')} # No prompt
        
        response = self.client.post('/api/query', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response_data['search_query_used'], refined_blip_query)
        self.assertEqual(response_data['category_hint_used'], expected_category_hint_for_api)
        
        self.mock_rapidapi_amazon_service.search_products.assert_called_once_with(
            query=refined_blip_query, limit=15, category_search_term=expected_category_hint_for_api
        )
        self.mock_rapidapi_flipkart_service.search_products.assert_called_once_with(
            query=refined_blip_query, limit=15, category_search_term=expected_category_hint_for_api
        )

    def test_query_deduplication_between_amazon_and_flipkart(self):
        """Test deduplication works for items fetched from Amazon and Flipkart services."""
        true_shared_title_for_dedupe = "Exact Same Product Title for Deduplication Test"
        shared_image_for_dedupe = "http://example.com/common_image_for_dedupe.jpg"

        amazon_product_1 = self._get_mock_product(1, "RapidAPI-Amazon", title_prefix="Shared")
        amazon_product_1['title'] = true_shared_title_for_dedupe
        amazon_product_1['image_url'] = shared_image_for_dedupe
        amazon_product_2 = self._get_mock_product(2, "RapidAPI-Amazon", title_prefix="Distinct Amazon")

        flipkart_product_duplicate = self._get_mock_product(3, "RapidAPI-Flipkart", title_prefix="Shared")
        flipkart_product_duplicate['title'] = true_shared_title_for_dedupe 
        flipkart_product_duplicate['image_url'] = shared_image_for_dedupe
        flipkart_product_distinct = self._get_mock_product(4, "RapidAPI-Flipkart", title_prefix="Distinct Flipkart")

        self.mock_rapidapi_amazon_service.search_products.return_value = [amazon_product_1, amazon_product_2]
        self.mock_rapidapi_flipkart_service.search_products.return_value = [flipkart_product_duplicate, flipkart_product_distinct]

        user_prompt = 'common product for dedupe'
        img_byte_arr = _create_dummy_image_bytes()
        data = {'image': (img_byte_arr, 'test_common.jpg'), 'prompt': user_prompt}

        response = self.client.post('/api/query', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data.decode('utf-8'))

        # Expected: amazon_product_1 (or its duplicate), amazon_product_2, flipkart_product_distinct
        self.assertEqual(len(response_data['products']), 3) # One duplicate should be removed by merge_and_dedupe

        returned_titles = {p['title'] for p in response_data['products']}
        self.assertIn(true_shared_title_for_dedupe, returned_titles)
        self.assertIn(amazon_product_2['title'], returned_titles)
        self.assertIn(flipkart_product_distinct['title'], returned_titles)

if __name__ == '__main__':
    unittest.main() 
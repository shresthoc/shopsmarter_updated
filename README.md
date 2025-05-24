# ShopSmarter 🛍️

An AI-powered e-commerce assistant that helps users find visually similar or complementary products using image and text queries, primarily focusing on Indian e-commerce platforms.

## Core Features

- 📸 **Image-based Search**: Upload an image to find similar products.
- ✍️ **Text-based Search**: Describe the product you're looking for.
- 🖼️ **Image + Text Search**: Combine an image with a textual description for refined results.
- 🧠 **AI-Powered Product Understanding**:
    - Uses **CLIP** embeddings for visual similarity.
    - Employs **BLIP** for image captioning to generate textual descriptions from images.
    - Extracts key tags/keywords from images for category hints.
- 💬 **Query Refinement (Optional LLM)**:
    - Uses `RefinementService` to generate better e-commerce search queries.
    - Can leverage a Hugging Face-based LLM (e.g., from `NousResearch`) if configured.
    - Falls back to spaCy-based keyword extraction if an LLM is not available or fails.
- 🛍️ **Dual E-commerce Platform Integration via RapidAPI**:
    - Fetches product data from **Amazon (India)** and **Flipkart** through configurable RapidAPI endpoints.
    - Centralized API key management for RapidAPI.
    - Flexible configuration for different API endpoints and parameters per platform.
- ✨ **Product Merging & Deduplication**: Combines results from different sources and removes duplicates.
- ⚙️ **Configurable API Behavior**: API endpoints, query parameters, and category search behavior can be customized via environment variables.

## Software & Technologies Used

- **Backend**:
    - Python 3.8+
    - Flask (for the web server)
    - `requests` (for making API calls)
    - `Pillow` (for image processing)
    - `sentence-transformers` (for CLIP embeddings)
    - `transformers` (for BLIP image captioning and Hugging Face LLMs)
    - `faiss-cpu` (for similarity search)
    - `spacy` (for fallback NLP tasks)
    - `python-dotenv` (for managing environment variables)
    - `langchain` (for structuring LLM interactions)
- **Frontend**:
    - React
    - JavaScript
    - CSS
- **APIs**:
    - **RapidAPI**: Used as a gateway to access e-commerce product data. The application is configured to use two instances of a generic RapidAPI service, one for an Amazon data provider and one for a Flipkart data provider.
        - Specific RapidAPI providers used:
            - `real-time-amazon-data.p.rapidapi.com` (configurable)
            - `real-time-flipkart-data2.p.rapidapi.com` (configurable)
- **Machine Learning Models**:
    - `clip-ViT-B-32` (or similar from sentence-transformers for image/text embeddings)
    - `Salesforce/blip-image-captioning-large` (for image captioning)
    - `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` (or similar, if `RefinementService` uses an LLM)

## Project Logic Overview

1.  **User Input**: The user provides an image, text, or both via the React frontend.
2.  **Backend Processing (`app.py`)**:
    *   **Image Handling**:
        *   If an image is uploaded, it's saved.
        *   **BLIP Captioning**: A descriptive caption is generated from the image (e.g., "a red t-shirt with a white logo").
        *   **Embedding Service**: Image tags are extracted (e.g., "red", "t-shirt") to be used as a `category_search_term`.
    *   **Query Generation**:
        *   If only an image is provided, the BLIP caption becomes the primary search query.
        *   If text is provided, it's used as the primary search query.
        *   If both are provided, the user's text query is prioritized.
        *   **`RefinementService`**: This service takes the initial query (either from user text or BLIP caption) and attempts to generate a more optimized e-commerce search query.
            *   It tries to use an LLM (if `OPENAI_API_KEY` is *not* set, it defaults to a free Hugging Face model specified in `RefinementService`).
            *   If LLM fails or is not configured, it falls back to spaCy for keyword extraction.
    *   **API Calls via `RapidApiEcomService`**:
        *   `app.py` initializes two instances of `RapidApiEcomService`: one configured for Amazon and one for Flipkart, using distinct environment variable prefixes (e.g., `RAPIDAPI_HOST_AMAZON`, `RAPIDAPI_HOST_FLIPKART`).
        *   For each service:
            *   The refined search query (from `RefinementService`) is passed as the main text query (e.g., to the `q` or `query` parameter of the respective RapidAPI endpoint).
            *   The `category_search_term` (derived from image tags) is passed. The `RapidApiEcomService` will use this for category-specific searching if `RAPIDAPI_CATEGORY_PARAM_<SERVICE_NAME>` and `RAPIDAPI_ENDPOINT_PATH_<SERVICE_NAME>` (for a category search endpoint) are configured in the `.env` file.
        *   Each `RapidApiEcomService` instance handles:
            *   Constructing the API request URL and parameters based on its specific `.env` configuration (`RAPIDAPI_HOST`, `RAPIDAPI_ENDPOINT_PATH`, `RAPIDAPI_TEXT_QUERY_PARAM`, `RAPIDAPI_CATEGORY_PARAM`, `RAPIDAPI_CATEGORY_VALUE_FORMAT`).
            *   Making the HTTP request to the configured RapidAPI endpoint.
            *   Parsing the JSON response and extracting product details (title, price, image URL, product URL, rating), with robust fallback mechanisms for various JSON structures.
    *   **Result Aggregation**:
        *   Products from both Amazon and Flipkart services are collected.
        *   `merge_and_dedupe_products`: Results are merged, and duplicates (based on title and image URL) are removed.
    *   **Final Refinement (Optional)**: The merged list can be further processed by `RefinementService`'s `refine_results` method (though current integration primarily uses it for top-N selection if LLM-based filtering is not active).
3.  **Response**: The final list of products is sent to the frontend for display.

## Setup and Run Locally

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd shopsmarter
```

**2. Set Up Backend (Python)**

*   Navigate to the backend directory:
    ```bash
    cd backend
    ```
*   Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
*   Install Python dependencies:
    ```bash
    pip install -r ../requirements.txt
    ```
    *(Note: `requirements.txt` is in the root directory)*

**3. Configure Environment Variables for Backend**

*   In the `backend/` directory, create a file named `.env`.
*   Copy the contents from `backend/.env.sample` into your new `.env` file.
*   **Fill in the required values in `backend/.env`**:

    ```env
    # Flask Environment
    FLASK_ENV=development
    FLASK_APP=app.py # Or the specific file that runs your Flask app

    # Directory for uploaded images (relative to backend/ or absolute)
    UPLOAD_FOLDER=uploads

    # --- Generic RapidAPI Credentials (used by both Amazon & Flipkart services) ---
    RAPIDAPI_KEY="your_actual_rapidapi_key_here" # Your single RapidAPI subscription key

    # --- Configuration for Amazon Product Source (via a RapidAPI provider) ---
    RAPIDAPI_HOST_AMAZON="real-time-amazon-data.p.rapidapi.com" # Host for the Amazon data API
    RAPIDAPI_ENDPOINT_PATH_AMAZON="/search"                     # Endpoint for searching Amazon products
    RAPIDAPI_TEXT_QUERY_PARAM_AMAZON="query"                    # URL parameter for the text search query (e.g., 'q', 'query', 'keyword')
    # Optional: For category-specific search on Amazon
    # RAPIDAPI_CATEGORY_PARAM_AMAZON="category_id"              # URL parameter for category filtering
    # RAPIDAPI_CATEGORY_VALUE_FORMAT_AMAZON="{category}"        # Format string if category value needs specific formatting
    # RAPIDAPI_ENDPOINT_PATH_CATEGORY_AMAZON="/search_by_category" # Separate endpoint if Amazon API uses one for categories

    # --- Configuration for Flipkart Product Source (via a RapidAPI provider) ---
    RAPIDAPI_HOST_FLIPKART="real-time-flipkart-data2.p.rapidapi.com" # Host for the Flipkart data API
    RAPIDAPI_ENDPOINT_PATH_FLIPKART="/search"                        # Default endpoint for searching Flipkart (adjust if using category specific one primarily)
    RAPIDAPI_TEXT_QUERY_PARAM_FLIPKART="query"                       # URL parameter for the text search query for Flipkart
    # Mandatory for current Flipkart setup if relying on category search:
    RAPIDAPI_CATEGORY_PARAM_FLIPKART="categoryId"                    # URL parameter for category filtering for Flipkart
    RAPIDAPI_ENDPOINT_PATH_CATEGORY_FLIPKART="/products-by-category" # Endpoint for category search for Flipkart

    # OpenAI API Key (Optional - for LLM query refinement)
    # If you have an OpenAI API key, uncomment and add it.
    # If not, the RefinementService will attempt to use a free Hugging Face model.
    # OPENAI_API_KEY="your_openai_api_key_here"

    # Hugging Face API Token (Optional)
    # Needed if using private/gated models from Hugging Face Hub.
    # Usually not required for public models like BLIP or default LLMs.
    # HF_TOKEN="your_huggingface_token_here"
    ```

    **Important Notes for `.env`:**
    *   You need to subscribe to the relevant APIs (e.g., an Amazon data API and a Flipkart data API) on the RapidAPI Marketplace to get `RAPIDAPI_KEY` and to ensure the `RAPIDAPI_HOST_...` values are correct for your subscriptions.
    *   The `RAPIDAPI_ENDPOINT_PATH_...`, `RAPIDAPI_TEXT_QUERY_PARAM_...`, and `RAPIDAPI_CATEGORY_PARAM_...` values **must match** the actual API provider's documentation on RapidAPI. Use the "Test Endpoint" feature on RapidAPI to verify these.
    *   For the Flipkart service, if you primarily want to use its category search, ensure `RAPIDAPI_ENDPOINT_PATH_FLIPKART` is set to the category search path (like `/products-by-category`) or that `RAPIDAPI_ENDPOINT_PATH_CATEGORY_FLIPKART` is set and used.

**4. Set Up Frontend (React)**

*   Navigate to the frontend directory (from the root of the project):
    ```bash
    cd ../frontend  # Or cd frontend from root
    ```
*   Install Node.js dependencies:
    ```bash
    npm install
    ```
*   (Optional) If your frontend requires environment variables (e.g., for a different Stripe key or analytics), create a `.env` or `.env.local` file in the `frontend/` directory as per your React app's convention (e.g., `REACT_APP_YOUR_VAR=value`).

**5. Run the Application**

*   **Terminal 1: Start Backend Server**
    Navigate to the `backend/` directory and run:
    ```bash
    source venv/bin/activate # If not still active
    flask run
    ```
    The backend will typically start on `http://127.0.0.1:5000`.

*   **Terminal 2: Start Frontend Development Server**
    Navigate to the `frontend/` directory and run:
    ```bash
    npm start
    ```
    The frontend will typically open automatically in your browser, often at `http://localhost:3000`.

Now you should be able to use the ShopSmarter application locally.

## Development Notes

*   **LLM for Refinement**: The `RefinementService` is designed to work without an OpenAI key by falling back to open-source models from Hugging Face. If `OPENAI_API_KEY` is not provided, it will attempt to use a pre-configured model like "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO". Ensure you have sufficient resources (RAM/VRAM) if running large local LLMs.
*   **API Rate Limits**: Be mindful of API rate limits on RapidAPI, especially on free tiers.
*   **Error Handling**: The backend includes logging. Check the Flask console output for errors or information about API calls.

## Future Enhancements (Ideas)

-   Implement user accounts and saved searches.
-   Add more advanced filtering options on the frontend (price range, ratings).
-   Support more e-commerce platforms by configuring additional `RapidApiEcomService` instances.
-   Improve the visual similarity model or fine-tune it on e-commerce data.
-   Implement more sophisticated deduplication strategies.

## Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/your-amazing-feature`).
3.  Commit your changes (`git commit -m 'Add your amazing feature'`).
4.  Push to the branch (`git push origin feature/your-amazing-feature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License. 
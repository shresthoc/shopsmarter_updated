# ShopSmarter 🛍️

An AI-powered e-commerce assistant that helps users find visually similar or complementary products using image and text queries.

## Features

- 📸 Image + text-based product search
- 🔍 AI-powered visual similarity matching using CLIP
- 🤖 LLM-driven query refinement
- 🛒 Test-mode checkout with Stripe
- 🎭 Automated demo with Puppeteer

## Prerequisites

- Python 3.8+
- Node.js 16+
- ~~Google Custom Search API key~~ (No longer primary, direct APIs used)
- Stripe test mode API keys
- **Unwrangle API Key** (for Amazon product data)
- **Flipkart Affiliate ID & Token** (for Flipkart product data)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ShopSmarter.git
   cd ShopSmarter
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Node.js dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure environment variables**
   Create a `.env` file in the `backend/` directory (note the change from root to `backend/` for server-side keys):
   ```env
   # Backend .env file (backend/.env)

   # Flask Environment (development, production)
   FLASK_ENV=development
   FLASK_APP=app.py # Should be app.py if FLASK_APP is run from backend dir

   # Directory for uploaded images (relative to backend/ or absolute)
   UPLOAD_FOLDER=uploads

   # Unwrangle Amazon API Key
   # Get your key from Unwrangle services
   UNWRANGLE_API_KEY="your_unwrangle_api_key_here"

   # Flipkart Affiliate API Credentials
   # Register as a Flipkart Affiliate to get these
   FK_AFFILIATE_ID="your_flipkart_affiliate_id_here"
   FK_AFFILIATE_TOKEN="your_flipkart_affiliate_token_here"
   
   # OpenAI API Key (for LLM refinement)
   # Optional: if you are using OpenAI for the RefinementService
   # OPENAI_API_KEY="your_openai_api_key_here"

   # Hugging Face API Token (if using private/gated models from Hugging Face Hub)
   # Optional: usually not needed for public models like BLIP
   # HF_TOKEN="your_huggingface_token_here"

   # --- Frontend .env file (frontend/.env.local or similar) ---
   # If your Stripe keys are used client-side, they should be in the frontend's .env
   # Example for React (typically .env.local, prefixed with REACT_APP_):
   # REACT_APP_STRIPE_PUBLISHABLE_KEY=your_stripe_test_publishable_key
   ```
   A `.env.sample` is provided in the `backend/` directory as a template.
   **Note on Stripe Keys**: The `STRIPE_SECRET_KEY` should remain in the backend's `.env` file. The `STRIPE_PUBLISHABLE_KEY` is often used by the frontend, so it might be in a `.env` file within the `frontend` directory (e.g., `frontend/.env.local` for Create React App, prefixed appropriately like `REACT_APP_STRIPE_PUBLISHABLE_KEY`). Adjust according to your frontend setup.

5. **~~Set up Google Custom Search API~~** (This step is no longer primary)
   - ~~Go to [Google Cloud Console](https://console.cloud.google.com)~~
   - ~~Create a new project~~
   - ~~Enable Custom Search API~~
   - ~~Create credentials and copy your API key~~
   - ~~Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com/)~~
   - ~~Create a new search engine~~
   - ~~Add the sites you want to search (e.g., amazon.com, ebay.com)~~
   - ~~Copy your Search Engine ID~~

   **New: Set up Product APIs**

   **a. Unwrangle Amazon API**
   - Visit [Unwrangle](https://www.unwrangle.com/) (or their specific API portal).
   - Sign up or log in to your account.
   - Navigate to your API dashboard or settings to find/generate your API key.
   - Copy this key into your `backend/.env` file as `UNWRANGLE_API_KEY`.

   **b. Flipkart Affiliate API**
   - Go to the [Flipkart Affiliate Program website](https://affiliate.flipkart.com/).
   - Register or log in to your affiliate account.
   - Once approved, find your Affiliate ID (`FK_AFFILIATE_ID`) and Affiliate Token (`FK_AFFILIATE_TOKEN`) in your dashboard. These are sometimes referred to as `Tracking ID` and `Access Token` or similar.
   - Copy these into your `backend/.env` file.

6. **Run the application**
   
   Terminal 1 (Backend):
   ```bash
   source venv/bin/activate
   flask run
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npm start
   ```

7. **Run the Puppeteer demo**
   ```bash
   cd scripts
   node demo.js
   ```

## Project Structure

```
ShopSmarter/
├── backend/
│   ├── api/
│   ├── services/
│   └── utils/
├── frontend/
│   └── src/
│       ├── components/
│       ├── hooks/
│       └── utils/
├── scripts/
├── requirements.txt
└── package.json
```

## Core Components

1. **~~Search & Scrape Service~~** (Replaced)
   - ~~Uses Google Custom Search API~~
   - ~~Scrapes product details using BeautifulSoup/Selenium~~
   - ~~Caches thumbnails locally~~

   **New: Product API Services**
   - **`AmazonApiService`**: Fetches structured product data from Amazon via Unwrangle API.
   - **`FlipkartApiService`**: Fetches product listings from Flipkart via their Affiliate API.
   - Both services standardize data for consistent processing.
   - Logic in `app.py` calls these services, merges results, and handles deduplication.

2. **Embedding & Similarity**
   - CLIP for image embeddings
   - FAISS for fast similarity search
   - Filters and ranks results

3. **LLM Refinement**
   - LangChain for query understanding
   - Re-ranks and filters results
   - Handles follow-up queries

4. **Frontend**
   - React components for image upload and results display
   - Stripe integration for test-mode checkout
   - Responsive product carousel

## Extending the Project

1. **Adding More E-commerce Sites**
   - Add site-specific scraping rules in `backend/services/scrapers/`
   - Update the search engine configuration

2. **Caching & Performance**
   - Implement Redis for embedding cache
   - Add rate limiting for API calls
   - Enable browser caching for thumbnails

3. **Styling & UI**
   - Customize the theme in `frontend/src/styles/`
   - Add more interactive features
   - Implement responsive design

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
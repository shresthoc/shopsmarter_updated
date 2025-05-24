import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState('');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedImage) return;

    setLoading(true);
    setResults([]);
    try {
      const formData = new FormData();
      const response = await fetch(selectedImage);
      const blob = await response.blob();
      formData.append('image', blob, 'image.jpg');
      if (prompt) {
        formData.append('prompt', prompt);
      }

      console.log('Sending request to backend...');
      const result = await fetch('http://localhost:5001/api/query', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
        },
        body: formData,
      });

      if (!result.ok) {
        const errorData = await result.json();
        throw new Error(errorData.details || `HTTP error! status: ${result.status}`);
      }

      const data = await result.json();
      console.log('Received response:', data);
      setResults(data.products || []);
    } catch (error) {
      console.error('Error details:', error);
      setResults([]);
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getCurrencySymbol = (currencyCode) => {
    if (!currencyCode) return '$';
    const symbols = {
      INR: '₹',
      USD: '$',
      EUR: '€',
      GBP: '£',
    };
    return symbols[currencyCode.toUpperCase()] || currencyCode;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ShopSmarter</h1>
        <p>Upload an image to find similar products</p>
      </header>

      <main>
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="image-upload">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              id="image-upload"
            />
            {selectedImage && (
              <img
                src={selectedImage}
                alt="Selected"
                className="preview-image"
              />
            )}
            <label htmlFor="image-upload">Choose an image</label>
          </div>

          <div className="prompt-input">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Add a description (optional)"
            />
          </div>

          <button 
            type="submit" 
            disabled={!selectedImage || loading}
            className={loading ? 'loading' : ''}
          >
            {loading ? 'Searching...' : 'Find Similar Products'}
          </button>
        </form>

        <div className="results">
          {results.map((product, index) => (
            <div key={product.id || product.product_url || index} className="product-card">
              <img src={product.image_url} alt={product.title} />
              <h3>{product.title}</h3>
              {product.price && (
                <p className="price-info">
                  Price: {getCurrencySymbol(product.currency)}
                  {product.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              )}
              {product.product_url && product.product_url.startsWith('http') ? (
                <a href={product.product_url} target="_blank" rel="noopener noreferrer" className="view-product-button">
                  View Product
                </a>
              ) : (
                <p className="no-link-message">Product link not available</p>
              )}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

export default App; 
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Helper to generate unique IDs for messages
const generateId = () => Math.random().toString(36).substr(2, 9);

// Placeholder for PWA install prompt
// let deferredPrompt;

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedImageFile, setSelectedImageFile] = useState(null); // To hold the actual file for submission
  // const [results, setResults] = useState([]); // Replaced by messages
  const [loading, setLoading] = useState(false);
  const [inputValue, setInputValue] = useState(''); // For chat input
  const [messages, setMessages] = useState([]);
  const [cart, setCart] = useState([]);
  const [isCartOpen, setIsCartOpen] = useState(false);
  const messagesEndRef = useRef(null); // To scroll to the bottom of messages

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add a welcome message
  useEffect(() => {
    setMessages([
      {
        id: generateId(),
        from: 'bot',
        text: 'Welcome to ShopSmarter! Upload an image or describe what you are looking for.',
      },
    ]);
    // PWA install listener - you can enable this if you set up PWA
    // window.addEventListener('beforeinstallprompt', (e) => {
    //   e.preventDefault();
    //   deferredPrompt = e;
    //   // Optionally, show an install button here
    // });
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setSelectedImageFile(file); // Store the file itself
    }
  };

  const handleSendMessage = async (event) => {
    if (event) event.preventDefault(); // Ensure it can be called without an event too

    const currentInput = inputValue.trim();
    const currentImage = selectedImage;
    const currentImageFile = selectedImageFile;

    if (!currentInput && !currentImage) return; // Don't send empty messages

    const userMessage = {
      id: generateId(),
      from: 'user',
      text: currentInput,
      image: currentImage, // Display uploaded image in user's chat bubble
    };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    setLoading(true);
    setInputValue('');
    setSelectedImage(null);
    setSelectedImageFile(null);
    // Clear the file input visually if possible (tricky, but selection is cleared)
    const fileInput = document.getElementById('chat-image-upload');
    if (fileInput) fileInput.value = null;

    try {
      const formData = new FormData();
      if (currentImageFile) {
        formData.append('image', currentImageFile, currentImageFile.name || 'image.jpg');
      }
      if (currentInput) {
        formData.append('prompt', currentInput); // 'prompt' is the key backend expects for text
      }
      
      // If neither image nor text, this case should have been caught earlier
      // but as a safeguard for formData:
      if (!currentImageFile && !currentInput) {
          throw new Error("Cannot send an empty query.");
      }

      console.log('Sending request to backend with prompt and/or image...');
      // Ensure the port and endpoint are correct, matching your Flask backend
      const result = await fetch('http://localhost:5001/api/query', { 
        method: 'POST',
        headers: {
          'Accept': 'application/json', 
          // 'Content-Type': 'multipart/form-data' is set automatically by browser with FormData
        },
        body: formData,
      });

      if (!result.ok) {
        const errorData = await result.json().catch(() => ({ details: 'Unknown error structure' }));
        throw new Error(errorData.details || `HTTP error! status: ${result.status}`);
      }

      const data = await result.json();
      console.log('Received response:', data);
      
      const botMessage = {
        id: generateId(),
        from: 'bot',
        text: data.text || (data.products && data.products.length > 0 ? `Here are some products I found:` : 'No products found matching your query.'),
        products: data.products || [],
      };
      setMessages((prevMessages) => [...prevMessages, botMessage]);

    } catch (error) {
      console.error('Error details:', error);
      const errorMessage = {
        id: generateId(),
        from: 'bot',
        text: 'Sorry, I encountered an error: ' + error.message,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const getCurrencySymbol = (currencyCode) => {
    if (!currencyCode) return '$'; // Default if not specified
    const symbols = {
      INR: '₹',
      USD: '$',
      EUR: '€',
      GBP: '£',
    };
    return symbols[String(currencyCode).toUpperCase()] || currencyCode;
  };

  const addToCart = (product) => {
    setCart((prevCart) => {
      // Optional: Check if product already in cart by URL or a unique ID
      const existingProduct = prevCart.find(item => item.product_url === product.product_url);
      if (existingProduct) {
        // Optional: alert('Product already in cart!');
        return prevCart; // Or update quantity if you add that feature
      }
      return [...prevCart, product];
    });
    // Send a notification message to chat
    const cartMessage = {
        id: generateId(),
        from: 'bot',
        text: `${product.title} added to cart!`
    };
    setMessages(prev => [...prev, cartMessage]);
  };

  const handleAddToCart = (product) => {
    setCart((prevCart) => {
      // Avoid adding duplicates
      if (prevCart.find(item => item.id === product.id)) {
        return prevCart;
      }
      return [...prevCart, product];
    });
  };

  const handleCartClick = () => {
    // Instead of an alert, we'll open a modal
    setIsCartOpen(true);
  };

  const closeCartModal = () => {
    setIsCartOpen(false);
  }

  // Basic ChatBubble component
  const ChatBubble = ({ from, children, image }) => (
    <div className={`chat-bubble ${from}`}>
      {image && <img src={image} alt="Uploaded preview" className="chat-image-preview" />}
      <div className="bubble-content">{children}</div>
    </div>
  );

  // ProductCard component (adapted from user's snippet and existing render logic)
  const ProductCard = ({ product, onAddToCart }) => {
    if (!product) {
      return null;
    }

    // Use the currency symbol from the data, or a default fallback.
    const currencySymbol = getCurrencySymbol(product.currency);
    const priceDisplay = product.price ? `${currencySymbol}${parseFloat(product.price).toFixed(2)}` : 'Price N/A';

    return (
      <div className="product-card">
        <div className="product-image-container">
          {product.photo_url ? (
            <img src={product.photo_url} alt={product.title || 'Product'} className="product-image" />
          ) : (
            <div className="product-image-na">N/A</div>
          )}
        </div>
        <div className="product-details">
          <h4 className="product-title">{product.title || 'N/A'}</h4>
          <p className="product-price">{priceDisplay}</p>
          <div className="product-tags">
            {(product.tags && product.tags.length > 0) ? product.tags.join(', ') : 'N/A'}
          </div>
          <p className="product-caption">
            <strong>Description:</strong> {product.caption || 'N/A'}
          </p>
          <div className="product-actions">
            {product.product_url ? (
              <a href={product.product_url} target="_blank" rel="noopener noreferrer" className="product-link">
                Link
              </a>
            ) : (
              <span className="product-link-na">Link N/A</span>
            )}
            <button onClick={() => onAddToCart(product)} className="add-to-cart-btn">
              Add to Cart
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ShopSmarter AI Chat</h1>
        <div className="cart-icon" onClick={handleCartClick}>
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path></svg>
          {cart.length > 0 && <span className="cart-count">{cart.length}</span>}
        </div>
      </header>

      {isCartOpen && (
        <div className="cart-modal-overlay">
          <div className="cart-modal">
            <div className="cart-modal-header">
              <h2>Your Cart</h2>
              <button onClick={closeCartModal} className="close-cart-btn">&times;</button>
            </div>
            <div className="cart-modal-body">
              {cart.length > 0 ? (
                <div className="product-list-cart">
                  {cart.map((product) => (
                    <ProductCard key={product.id} product={product} onAddToCart={() => {}} />
                  ))}
                </div>
              ) : (
                <p>Your cart is empty.</p>
              )}
            </div>
          </div>
        </div>
      )}

      <main className="chat-container">
        <div className="message-list">
          {messages.map((m) => (
            <ChatBubble key={m.id} from={m.from} image={m.image}>
              {m.text}
              {m.from === 'bot' && m.products && m.products.length > 0 && (
                <div className="product-list-chat">
                  {m.products.map((p, idx) => (
                    <ProductCard
                      key={p.product_url || p.id || idx} // Ensure unique key
                      product={p}
                      onAddToCart={addToCart}
                    />
                  ))}
                </div>
              )}
            </ChatBubble>
          ))}
          {loading && (
            <ChatBubble from="bot">
              <div className="loading-dots">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </ChatBubble>
          )}
          <div ref={messagesEndRef} /> {/* For scrolling to bottom */}
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-form">
          <div className="input-area">
            <label htmlFor="chat-image-upload" className="chat-image-upload-button">
              📎
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                id="chat-image-upload"
                style={{ display: 'none' }} 
              />
            </label>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type a message or upload an image..."
              className="chat-text-input"
            />
            <button 
              type="submit" 
              disabled={loading || (!inputValue.trim() && !selectedImage)}
              className="send-button"
            >
              Send
            </button>
          </div>
          {selectedImage && (
            <div className="image-preview-chat-input">
              <img src={selectedImage} alt="Preview" />
              <button type="button" onClick={() => { setSelectedImage(null); setSelectedImageFile(null); }} className="remove-image-button">
                &times;
              </button>
            </div>
          )}
        </form>
      </main>
    </div>
  );
}

export default App; 
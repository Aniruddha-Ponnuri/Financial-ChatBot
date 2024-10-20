// Chatbot.js
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Loader } from 'lucide-react';
import './Chatbot.css'; // Import the CSS file

const Chatbot = () => {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [summarizedHistory, setSummarizedHistory] = useState('');
  const [loading, setLoading] = useState(false);

  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversation]);

  const handleAskQuestion = async () => {
    if (!question) {
      alert('Please enter a question.');
      return;
    }
    try {
      setLoading(true);  // Show loading indicator
      const newConversation = [...conversation, { role: 'user', content: question }];
      setConversation(newConversation);

      const response = await axios.post('http://localhost:5000/ask', { 
        question,
        history: summarizedHistory,
      });

      setConversation([...newConversation, { role: 'assistant', content: response.data.answer }]);
      setSummarizedHistory(response.data.summarized_history);
      setQuestion('');
    } catch (error) {
      console.error('Error asking question');
    } finally {
      setLoading(false);  // Hide loading indicator
    }
  };

  return (
    <div className="chat-container">
      {/* Chat Messages */}
      <div className="chat-messages" ref={chatContainerRef}>
        {conversation.map((msg, index) => (
          <div
            key={index}
            className={msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}
            data-sender={msg.role === 'user' ? 'You' : 'Bot'}
          >
            {msg.content}
          </div>
        ))}
      </div>

      {/* Footer with Input and Send Button */}
      <div className="footer">
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="message-input"
          disabled={loading}  // Disable input while loading
        />
        <button 
          onClick={handleAskQuestion}
          disabled={loading}  // Disable button while loading
          className={`ask-button ${loading ? 'disabled' : ''}`}
        >
          {loading ? <Loader className="loader-spin mr-2" /> : <Send className="mr-2" />}
          {loading ? 'Asking...' : 'Ask'}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;

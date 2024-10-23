import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Loader } from 'lucide-react';
import './Chatbot.css';

const Chatbot = () => {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [summarizedHistory, setSummarizedHistory] = useState('');
  const [loading, setLoading] = useState(false);

  const chatContainerRef = useRef(null);
  const lastMessageRef = useRef(null);  // Reference for the last message

  // Auto scroll to the last message whenever the conversation updates
  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
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

      // Clear the input field immediately after sending the question
      setQuestion('');

      const response = await axios.post('http://localhost:5000/ask', { 
        question,
        history: summarizedHistory,
      });

      setConversation([...newConversation, { role: 'assistant', content: response.data.answer }]);
      setSummarizedHistory(response.data.summarized_history);
    } catch (error) {
      console.error('Error asking question');
    } finally {
      setLoading(false);  // Hide loading indicator
    }
  };

  // Handle 'Enter' key press in the input field
  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault(); // Prevent the default form submit action
      handleAskQuestion(); // Call the function to send the message
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
            dangerouslySetInnerHTML={{ __html: msg.content }} // Render HTML content
            ref={index === conversation.length - 1 ? lastMessageRef : null}  // Set ref to the last message
          />
        ))}
      </div>

      {/* Footer with Input and Ask Button */}
      <div className="footer">
        {/* Input for asking question */}
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="message-input"
          disabled={loading}  // Disable input while loading
          onKeyDown={handleKeyDown}  // Listen for Enter key press
        />

        {/* Ask Button */}
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

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Loader, ThumbsUp, ThumbsDown } from 'lucide-react';
import './Chatbot.css';

const Chatbot = () => {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [summarizedHistory, setSummarizedHistory] = useState('');
  const [loading, setLoading] = useState(false);
  const [useRL, setUseRL] = useState(true); // Toggle RL mode
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  const [feedbackGiven, setFeedbackGiven] = useState({}); // Track feedback per message

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
    
    const userQuestion = question; // Store the question before clearing
    
    try {
      setLoading(true);  // Show loading indicator
      const newConversation = [...conversation, { role: 'user', content: userQuestion }];
      setConversation(newConversation);

      // Clear the input field immediately after sending the question
      setQuestion('');

      const response = await axios.post('http://localhost:5000/ask', { 
        question: userQuestion,
        history: summarizedHistory,
        use_rl: useRL, // Send RL preference
      });

      // Add assistant message with metadata for feedback
      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        question: userQuestion, // Store original question for feedback
        rlUsed: response.data.rl_used || false,
        timestamp: Date.now(),
      };

      setConversation([...newConversation, assistantMessage]);
      setSummarizedHistory(response.data.summarized_history);
    } catch (error) {
      console.error('Error asking question:', error);
      alert('Failed to get response. Please try again.');
    } finally {
      setLoading(false);  // Hide loading indicator
    }
  };

  // Handle feedback submission
  const handleFeedback = async (messageIndex, rating) => {
    const message = conversation[messageIndex];
    
    if (!message || message.role !== 'assistant') {
      console.error('Invalid message for feedback');
      return;
    }

    try {
      // Submit feedback to backend
      await axios.post('http://localhost:5000/feedback', {
        question: message.question,
        answer: message.content,
        rating: rating, // 0 for negative, 1 for positive
        session_id: sessionId,
      });

      // Mark feedback as given for this message
      setFeedbackGiven(prev => ({
        ...prev,
        [messageIndex]: rating,
      }));

      console.log(`Feedback submitted: ${rating === 1 ? 'Positive' : 'Negative'}`);
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
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
      {/* Header with RL Toggle */}
      <div className="chat-header">
        <div className="header-content">
          <h2>Financial Chatbot</h2>
          <div className="rl-toggle">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={useRL}
                onChange={(e) => setUseRL(e.target.checked)}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
              <span className="toggle-text">
                {useRL ? '🤖 RL Mode' : '💬 Standard Mode'}
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="chat-messages" ref={chatContainerRef}>
        {conversation.map((msg, index) => (
          <div
            key={index}
            className={`message-wrapper ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}
            ref={index === conversation.length - 1 ? lastMessageRef : null}
          >
            <div
              className={msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}
              data-sender={msg.role === 'user' ? 'You' : 'Bot'}
            >
              <div dangerouslySetInnerHTML={{ __html: msg.content }} />
              
              {/* Feedback Buttons for Bot Messages */}
              {msg.role === 'assistant' && (
                <div className="feedback-buttons">
                  <button
                    className={`feedback-btn ${feedbackGiven[index] === 1 ? 'active-positive' : ''}`}
                    onClick={() => handleFeedback(index, 1)}
                    disabled={feedbackGiven[index] !== undefined}
                    title="Good response"
                  >
                    <ThumbsUp size={16} />
                  </button>
                  <button
                    className={`feedback-btn ${feedbackGiven[index] === 0 ? 'active-negative' : ''}`}
                    onClick={() => handleFeedback(index, 0)}
                    disabled={feedbackGiven[index] !== undefined}
                    title="Poor response"
                  >
                    <ThumbsDown size={16} />
                  </button>
                  {msg.rlUsed && (
                    <span className="rl-badge" title="Response generated using RL">
                      🤖 RL
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
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

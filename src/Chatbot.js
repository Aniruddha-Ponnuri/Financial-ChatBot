import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send } from 'lucide-react';

const Chatbot = () => {
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [summarizedHistory, setSummarizedHistory] = useState('');

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
    }
  };

  return (
    <div className="chat-container">
      {/* Chat Messages */}
      <div className="chat-messages" ref={chatContainerRef}>
        {conversation.map((msg, index) => (
          <div key={index} className={`clearfix ${msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
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
        />
        <button 
          onClick={handleAskQuestion}
          className="ml-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex items-center"
        >
          <Send className="mr-2" />
          Ask
        </button>
      </div>

      {/* Inline Styles */}
      <style jsx global>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          justify-content: space-between;
        }
        .chat-messages {
          flex-grow: 1;
          overflow-y: auto;
          padding: 10px;
        }
        .footer {
          display: flex;
          padding: 10px;
          background-color: #717171;
          border-radius: 10px;
          align-items: center;
          position: relative;
        }
        .message-input {
          flex-grow: 1;
          padding: 8px;
          border-radius: 5px;
          border: 1px solid #ccc;
          color: #1c1c1c;
        }
        .user-bubble {
          background-color: #007bff;
          color: white;
          padding: 10px;
          border-radius: 10px;
          margin-bottom: 10px;
          width: fit-content;
          max-width: 80%;
          float: right;
          clear: both;
        }
        .bot-bubble {
          background-color: #1e1e1e;
          padding: 10px;
          border-radius: 10px;
          margin-bottom: 10px;
          width: fit-content;
          max-width: 80%;
          color: white;
          float: left;
          clear: both;
        }
        .clearfix {
          overflow: auto;
        }
      `}</style>
    </div>
  );
};

export default Chatbot;

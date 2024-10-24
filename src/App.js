import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Chatbot from './Chatbot';
import './App.css'; 
import logo from './logo.png'; 
import chat from './chatbot.png';

const App = () => {
  const [showChatbot, setShowChatbot] = useState(false);

  const toggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };

  return (
    <Router>
      <div className="app-container">
        {/* Title bar */}
        <div className="titlebar">
          <img src={logo} alt="Logo" className="logo" />
        </div>

        {/* Right Panel */}
        <div className="right-panel">
          {/* Chatbot Overlay */}
          {showChatbot && (
            <div className="chatbot-overlay">
              <Chatbot />
            </div>
          )}

          {/* Chat Icon */}
          <div className="chat-icon" onClick={toggleChatbot}>
            <img src={chat} alt="Chat Icon" />
          </div>
        </div>
      </div>

      <Routes>
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  );
};

export default App;

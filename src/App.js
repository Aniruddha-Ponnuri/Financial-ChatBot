// App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import PdfUpload from './PdfUpload';
import Chatbot from './Chatbot';
import './App.css'; // Import global CSS

const App = () => {
  return (
    <Router>
      <div className="app-container">
        {/* Left Side - PDF Upload */}
        <div className="left-panel">
          <PdfUpload />
        </div>

        {/* Right Side - Chatbot */}
        <div className="right-panel">
          <Chatbot />
        </div>
      </div>

      <Routes>
        <Route path="/upload" element={<PdfUpload />} />
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  );
};

export default App;

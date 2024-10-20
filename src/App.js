import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import PdfUpload from './PdfUpload';
import Chatbot from './Chatbot';

const App = () => {
  return (
    <Router>
      <div style={{ display: 'flex', height: '100vh' }}>
        {/* Left Side - PDF Upload */}
        <div style={{ width: '50%', padding: '10px', boxSizing: 'border-box' }}>
          <PdfUpload />
        </div>

        {/* Right Side - Chatbot */}
        <div style={{ width: '50%', padding: '10px', boxSizing: 'border-box', backgroundColor: '#f1f1f1', display: 'flex', flexDirection: 'column' }}>
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

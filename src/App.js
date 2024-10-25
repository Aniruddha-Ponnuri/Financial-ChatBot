import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Chatbot from './Chatbot';
import './App.css'; // Import global CSS
import logo from './logo.png'; // Update the path accordingly

const App = () => {
  return (
    <Router>
      <div className="app-container">
        {/* Title bar */}
        <div className="titlebar">
          <img src={logo} alt="Logo" className="logo" />
          <h1 className="title"> </h1>
        </div>

        {/* Chatbot with PDF upload functionality integrated */}
        <div className="right-panel">
          <Chatbot />
        </div>
      </div>

      <Routes>
        <Route path="/chatbot" element={<Chatbot />} />
      </Routes>
    </Router>
  );
};

export default App;

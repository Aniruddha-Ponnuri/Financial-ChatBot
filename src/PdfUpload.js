// PdfUpload.js
import React, { useState } from 'react';
import axios from 'axios';
import { ArrowUpCircle, Loader } from 'lucide-react';
import './PdfUpload.css'; // Import the CSS file

const PdfUpload = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
    } else {
      alert('Please upload a PDF file.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const uploadPdf = async () => {
    if (!pdfFile) {
      alert('Please select a PDF file first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', pdfFile);

    try {
      setLoading(true); // Show loading indicator
      setStatusMessage('Uploading PDF...');
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatusMessage(response.data.message);
    } catch (error) {
      setStatusMessage('Error uploading PDF');
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className="pdf-upload-container">
      <h2>Upload PDF</h2>
      <p>Supported formats: PDF. Max file size: 10MB.</p>
      <div 
        className={`drag-drop-area ${isDragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {pdfFile ? (
          <p>{pdfFile.name}</p>
        ) : (
          <p>Drag and drop a PDF file here, or click to select a file.</p>
        )}
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept="application/pdf"  // Restrict to PDF files
          className="file-input"
        />
      </div>
      <button 
        onClick={uploadPdf}
        disabled={loading}  // Disable button during upload
        className={`upload-button ${loading ? 'disabled' : ''}`}
      >
        {loading ? <Loader className="loader-spin mr-2" /> : <ArrowUpCircle className="mr-2" />}
        {loading ? 'Uploading...' : 'Upload PDF'}
      </button>
      {statusMessage && <p className="status-message">{statusMessage}</p>}
    </div>
  );
};

export default PdfUpload;

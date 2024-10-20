import React, { useState } from 'react';
import axios from 'axios';
import { ArrowUpCircle, Loader } from 'lucide-react';
import './PdfUpload.css';  // Import the new CSS file

const PdfUpload = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);  // Progress for file upload
  const [error, setError] = useState('');  // To store error messages

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError('');
    } else {
      setError('Please upload a valid PDF file.');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setError('');
    } else {
      setError('Please upload a valid PDF file.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const uploadPdf = async () => {
    if (!pdfFile) {
      setError('No file selected. Please select a PDF file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', pdfFile);

    try {
      setLoading(true); // Show loading indicator
      setProgress(0);
      setStatusMessage('');
      setError('');

      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });
      setStatusMessage(response.data.message);
    } catch (error) {
      setStatusMessage('Error uploading PDF');
      setError('Failed to upload the file. Please try again.');
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className="pdf-upload-container">
      <h2>Upload PDF</h2>
      <p className="instructions">Supported formats: PDF. Max file size: 10MB.</p>
      <div
        className={`drag-drop-area ${error ? 'error-border' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {pdfFile ? (
          <p>{pdfFile.name}</p>
        ) : (
          <p>Drag and drop a PDF file here, or click to select a file.</p>
        )}
        <input
          type="file"
          onChange={handleFileChange}
          accept="application/pdf" // Restrict to PDF files
          className="file-input"
        />
      </div>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          <span>{progress}%</span>
        </div>
      )}

      <button
        onClick={uploadPdf}
        disabled={loading} // Disable button during upload
        className={`upload-button ${loading ? 'disabled' : ''}`}
      >
        {loading ? <Loader className="loader-spin mr-2" /> : <ArrowUpCircle className="mr-2" />}
        {loading ? 'Uploading...' : 'Upload PDF'}
      </button>

      {statusMessage && <p className="status-message">{statusMessage}</p>}
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default PdfUpload;

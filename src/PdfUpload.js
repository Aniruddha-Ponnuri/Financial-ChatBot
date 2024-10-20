import React, { useState } from 'react';
import axios from 'axios';
import './PdfUpload.css';  // Import the new CSS file

const PdfUpload = ({ onClose }) => {  // Added onClose prop to close the modal
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
      uploadPdf(file);  // Automatically trigger the upload
    } else {
      setError('Please upload a valid PDF file.');
    }
  };

  const uploadPdf = async (file) => {
    if (!file) {
      setError('No file selected. Please select a PDF file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

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
      
      // Close the modal after successful upload
      if (onClose) {
        setTimeout(() => {
          onClose(); // Trigger the close callback
        }, 1500); // Set a delay to give time for users to see the success message
      }

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
      <p className="instructions">Drag and drop a PDF file here, or click to select a file</p>
      
      <div className="drag-drop-area">
        <input
          type="file"
          onChange={handleFileChange}
          accept="application/pdf"
          className="file-input"
        />
        <p>Drop your PDF here or click to browse</p>
      </div>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          <span>{progress}%</span>
        </div>
      )}

      {statusMessage && <p className="status-message">{statusMessage}</p>}
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default PdfUpload;

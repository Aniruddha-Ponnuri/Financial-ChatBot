import React, { useState } from 'react';
import axios from 'axios';
import { ArrowUpCircle } from 'lucide-react';

const PdfUpload = () => {
  const [pdfFile, setPdfFile] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  const uploadPdf = async () => {
    if (!pdfFile) {
      alert('Please select a PDF file first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', pdfFile);

    try {
      setStatusMessage('Uploading PDF...');
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setStatusMessage(response.data.message);
    } catch (error) {
      setStatusMessage('Error uploading PDF');
    }
  };

  return (
    <div className="p-8 bg-white shadow-lg">
      <h2 className="text-2xl font-bold mb-6">Upload PDF</h2>
      <div className="mb-4">
        <input 
          type="file" 
          onChange={handleFileChange} 
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>
      <button 
        onClick={uploadPdf}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex items-center"
      >
        <ArrowUpCircle className="mr-2" />
        Upload PDF
      </button>
      {statusMessage && <p className="mt-4 text-sm text-gray-600">{statusMessage}</p>}
    </div>
  );
};

export default PdfUpload;

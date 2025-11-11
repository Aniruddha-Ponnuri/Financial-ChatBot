import axios from 'axios';
import config from '../config';

// Create axios instance with default config from centralized configuration
const apiClient = axios.create({
  baseURL: config.api.baseURL,
  headers: config.api.headers,
  timeout: config.api.timeout,
});

// API Service object
const apiService = {
  // Chat endpoints
  askQuestion: async (question, history, useRL, sessionId) => {
    const response = await apiClient.post(config.endpoints.ask, {
      question,
      history,
      use_rl: useRL,
      session_id: sessionId,
    });
    return response.data;
  },

  // Session endpoints
  getAllSessions: async () => {
    const response = await apiClient.get(config.endpoints.sessions);
    return response.data.sessions || [];
  },

  getSession: async (sessionId) => {
    const response = await apiClient.get(config.endpoints.sessionById(sessionId));
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await apiClient.delete(config.endpoints.sessionById(sessionId));
    return response.data;
  },

  // Feedback endpoint
  submitFeedback: async (question, answer, rating, sessionId) => {
    const response = await apiClient.post(config.endpoints.feedback, {
      question,
      answer,
      rating,
      session_id: sessionId,
    });
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await apiClient.get(config.endpoints.health);
    return response.data;
  },

  // Get current API base URL (useful for debugging)
  getBaseURL: () => config.api.baseURL,
};

// Error interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);
      throw new Error(error.response.data.error || 'Server error occurred');
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message);
      throw new Error('Network error - please check your connection');
    } else {
      // Something else happened
      console.error('Error:', error.message);
      throw error;
    }
  }
);

export default apiService;

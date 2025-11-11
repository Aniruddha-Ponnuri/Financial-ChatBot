/**
 * Application Configuration
 * Centralized configuration for the entire application.
 * All environment variables and constants should be defined here.
 */

const config = {
  // API Configuration
  api: {
    baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000',
    timeout: parseInt(process.env.REACT_APP_API_TIMEOUT || '30000', 10),
    headers: {
      'Content-Type': 'application/json',
    },
  },

  // API Endpoints
  endpoints: {
    ask: '/ask',
    sessions: '/sessions',
    sessionById: (id) => `/sessions/${id}`,
    feedback: '/feedback',
    health: '/health',
    candidates: '/candidates',
  },

  // Application Settings
  app: {
    name: 'Financial Chatbot',
    version: '1.0.0',
  },

  // Feature Flags
  features: {
    rlMode: true,
    sessionPersistence: true,
    feedback: true,
  },
};

// Validate required environment variables
const validateConfig = () => {
  if (!config.api.baseURL) {
    console.error('Missing required environment variable: REACT_APP_API_BASE_URL');
  }
};

validateConfig();

export default config;

# Financial Chatbot

An AI financial advisory chatbot focused on Indian markets. It supports Groq and NVIDIA LLMs, includes stock symbol extraction, and can enrich answers with real-time market data.

## Features

- Conversational financial advice tailored for Indian markets
- Real-time stock data via yfinance
- Automatic stock symbol extraction from questions
- Session history with titled conversations
- Centralized YAML prompt configuration
- Structured logging

## Architecture

```
Frontend (React)          Backend (Flask)
─────────────────         ───────────────

┌─────────────┐          ┌─────────────┐
│  Chatbot.js │◄────────►│   app.py    │
│  (UI/UX)    │   HTTP   │  (API)      │
└─────────────┘          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │database.py  │
                          │(SQLite)     │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │llm_service  │
                          │(Groq/NVIDIA)│
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │stock_data   │
                          │(yfinance)   │
                          └─────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- Groq or NVIDIA API key

### Backend Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
cd Backend
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

#### Groq
```env
ENVIRONMENT=DEV
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.1-8b-instant
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

#### NVIDIA
```env
ENVIRONMENT=DEV
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
LLM_MODEL_NAME=your_nvidia_model_name
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000
```

4. Edit `Backend/bot_config.yaml` to customize prompts and model settings.

### Frontend Setup

```bash
cd Frontend
npm install
```

## Running the Application

### Start Backend Server

```bash
cd Backend
python app.py
```

The backend runs on `http://localhost:5000`.

### Start Frontend Development Server

```bash
cd Frontend
npm start
```

## API Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/ask` | POST | Get chatbot response | `{question, history, session_id}` |
| `/sessions` | GET | List chat sessions | - |
| `/sessions/<session_id>` | GET | Session messages | - |
| `/sessions/<session_id>` | DELETE | Delete session | - |
| `/health` | GET | System health check | - |

## Project Structure

```
Financial-ChatBot/
├── Backend/
│   ├── app.py              # Flask server
│   ├── config.py           # Configuration loader
│   ├── bot_config.yaml     # Prompts and settings
│   ├── services/
│   │   └── llm_service.py  # Groq/NVIDIA integration
│   ├── utils/
│   │   ├── database.py     # SQLite chat sessions
│   │   └── stock_data.py   # Market data fetcher
├── Frontend/
│   ├── src/
│   │   ├── Chatbot.js      # React chatbot component
│   │   └── Chatbot.css     # Styles


## Troubleshooting

### Backend won't start

- Verify the API key in `.env` matches the provider.
- Confirm `LLM_PROVIDER` is set to `groq` or `nvidia`.

### Frontend can't connect

- Ensure backend is running on port 5000.
- Check `REACT_APP_API_BASE_URL` if using a custom URL.
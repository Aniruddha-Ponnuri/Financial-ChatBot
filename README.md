# Financial Chatbot with Reinforcement Learning 🤖💰

An intelligent financial advisory chatbot powered by AI with **Reinforcement Learning** capabilities that learns from user feedback to continuously improve response quality.

## 🌟 Features

### Core Capabilities
- 💬 **Conversational AI** - Natural language financial advice using Groq LLM (Llama 3.1)
- 🤖 **Reinforcement Learning** - Learns from user feedback to improve responses over time
- 📊 **Multi-Candidate Generation** - Generates multiple responses and selects the best one
- 👍👎 **User Feedback System** - Rate responses to train the AI
- 📈 **Continuous Learning** - Model updates incrementally with each feedback
- 🎚️ **RL Toggle** - Switch between RL and Standard mode on-the-fly
- 💾 **Persistent Storage** - SQLite database for feedback tracking
- 📝 **Conversation History** - Maintains context across messages
- 🇮🇳 **India-Focused** - Financial advice tailored for Indian markets

### Technical Highlights
- ⚡ Fast response generation with Groq API
- 🧠 Sentence embedding-based reward model
- 🔄 Incremental learning without full retraining
- 🎨 Modern React UI with feedback controls
- 📊 Real-time analytics and statistics
- 🔧 Centralized YAML configuration
- 📋 Comprehensive logging system

## 🏗️ Architecture

```
Frontend (React)          Backend (Flask)           ML Components
─────────────────         ───────────────          ──────────────
                          
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│  Chatbot.js │◄────────►│   app.py    │◄────────►│reward_model │
│  (UI/UX)    │   HTTP   │  (API)      │          │  (Scoring)  │
└─────────────┘          └──────┬──────┘          └─────────────┘
                                 │                         │
                          ┌──────▼──────┐          ┌─────▼──────┐
                          │database.py  │          │sentence    │
                          │(SQLite)     │          │transformers│
                          └─────────────┘          └────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn
- Groq API Key

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/Aniruddha-Ponnuri/Financial-ChatBot.git
cd Financial-ChatBot
```

2. **Create and activate virtual environment**
```bash
# Using conda
conda create -n chat python=3.10
conda activate chat

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install flask flask-cors groq python-dotenv sentence-transformers scikit-learn PyYAML
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
ENVIRONMENT=DEV
```

5. **Configure the bot**

Edit `Backend/bot_config.yaml` to customize:
- System prompts
- RL settings (candidates, temperature)
- Model parameters

### Frontend Setup

1. **Install Node dependencies**
```bash
npm install
```

2. **Verify axios and lucide-react are installed**
```bash
npm install axios lucide-react
```

## 🚀 Running the Application

### Start Backend Server

```bash
cd Backend
python app.py
```

The backend will start on `http://localhost:5000`

Expected output:
```
[INFO] Starting Financial Chatbot with RL capabilities
[INFO] RL Mode: Enabled
[INFO] Reward model trained on 0 samples
* Running on http://127.0.0.1:5000
```

### Start Frontend Development Server

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.  
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.

## 🎮 Using the Application

### User Interface

1. **Header Controls**
   - Toggle between **🤖 RL Mode** (multiple candidates) and **💬 Standard Mode** (single response)
   - RL toggle is enabled by default

2. **Chat Interface**
   - Type your financial question in the input field
   - Press Enter or click "Ask" button
   - Wait for the response (4-8 seconds in RL mode, 1-2 seconds in standard)

3. **Feedback System**
   - Each bot response shows two buttons: 👍 (good) and 👎 (poor)
   - Click to rate the response quality
   - Buttons highlight green (positive) or red (negative) after clicking
   - Feedback is sent to backend to improve the model

4. **RL Badge**
   - Responses generated using RL show a "🤖 RL" badge
   - Indicates the response was selected from multiple candidates

### Example Workflow

```
1. User: "Should I invest in mutual funds or stocks?"
   ↓
2. System generates 4 candidates (if RL enabled)
   ↓
3. Reward model scores and selects best response
   ↓
4. Bot: "Consider diversification..." [🤖 RL] [👍] [👎]
   ↓
5. User clicks 👍 (positive feedback)
   ↓
6. System learns and improves future responses
```

## 🧪 Testing

### Run the Test Suite

```bash
cd Backend
python test_rl.py
```

This will test:
- Health endpoint
- Standard question handling
- RL-enabled question handling
- Feedback submission
- Statistics retrieval
- Candidate generation

### Manual Testing

1. **Test Standard Mode**
   ```bash
   curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What is diversification?","use_rl":false}'
   ```

2. **Test RL Mode**
   ```bash
   curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What is diversification?","use_rl":true}'
   ```

3. **Submit Feedback**
   ```bash
   curl -X POST http://localhost:5000/feedback \
     -H "Content-Type: application/json" \
     -d '{"question":"What is SIP?","answer":"Systematic Investment Plan...","rating":1}'
   ```

4. **Check Statistics**
   ```bash
   curl http://localhost:5000/feedback/stats
   ```

## 📊 API Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/ask` | POST | Get chatbot response | `{question, history, use_rl}` |
| `/feedback` | POST | Submit user rating | `{question, answer, rating, session_id}` |
| `/feedback/stats` | GET | Get feedback statistics | - |
| `/generate_candidates` | POST | Generate multiple responses | `{question, n}` |
| `/health` | GET | System health check | - |

## ⚙️ Configuration

### RL Settings (`Backend/bot_config.yaml`)

```yaml
rl_config:
  enabled: true              # Enable/disable RL
  n_candidates: 4            # Number of responses to generate
  temperature_min: 0.7       # Minimum temperature for diversity
  temperature_max: 1.2       # Maximum temperature for diversity
  reward_model:
    embedding_model: "all-MiniLM-L6-v2"
    min_samples_for_training: 20
```

### Customize Prompts

All prompts are in `Backend/bot_config.yaml`:

```yaml
prompts:
  system_prompt: "You are a financial assistant..."
  general_question_prompt: |
    You are an AI assistant. Answer the following question...
  financial_prompt_template: |
    You are an AI financial assistant. Use your knowledge...
```

## 📈 How RL Works

### Learning Process

1. **Initial State** (No training data)
   - Reward model returns uniform scores
   - Responses selected randomly from candidates

2. **Feedback Collection** (Users rate responses)
   - Positive feedback (👍) = rating 1
   - Negative feedback (👎) = rating 0
   - Stored in SQLite database

3. **Model Training** (After 20+ samples)
   - Embeddings created: `"Q: {question}\nA: {answer}"`
   - SGD Classifier trained on feedback
   - Model saved to disk

4. **Improved Selection** (Ongoing)
   - Future candidates scored accurately
   - Higher quality responses selected
   - Continuous improvement with more feedback

### Performance Metrics

- **Response Time**: 4-8 seconds (RL) vs 1-2 seconds (Standard)
- **Training Threshold**: 20 minimum samples
- **Optimal Feedback**: 50/50 balance of positive/negative
- **Model Updates**: Incremental (no full retraining)

## 📁 Project Structure

```
Financial-ChatBot/
├── Backend/
│   ├── app.py                    # Flask server with RL endpoints
│   ├── config.py                 # Configuration loader
│   ├── logger.py                 # Custom logging system
│   ├── database.py               # SQLite feedback handler
│   ├── reward_model.py           # ML reward model
│   ├── bot_config.yaml           # Configuration file
│   ├── test_rl.py                # Test suite
│   ├── feedback.db               # SQLite database (auto-created)
│   ├── model_data/               # Saved models
│   │   └── reward_classifier.pkl
├── src/
│   ├── Chatbot.js                # React chatbot component
│   ├── Chatbot.css               # Chatbot styles
│   ├── App.js                    # Main app component
│   └── index.js                  # React entry point
├── public/
│   └── index.html                # HTML template
├── FRONTEND_RL_GUIDE.md          # Frontend documentation
├── COMPLETE_RL_SUMMARY.md        # Full implementation guide
├── QUICK_REFERENCE.md            # Quick reference card
├── SYSTEM_FLOW_DIAGRAM.md        # Architecture diagram
├── README.md                     # This file
├── package.json                  # Node dependencies
└── .env                          # Environment variables
```

## 🔧 Troubleshooting

### Backend Issues

**Problem**: Import errors for ML packages
```bash
# Solution: Install dependencies
pip install sentence-transformers scikit-learn PyYAML
```

**Problem**: Backend won't start
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify GROQ_API_KEY is set
cat .env  # Should contain GROQ_API_KEY=...
```

**Problem**: RL not working
```bash
# Check configuration
cat Backend/bot_config.yaml  # rl_config.enabled should be true

# Check health endpoint
curl http://localhost:5000/health
```

### Frontend Issues

**Problem**: Can't connect to backend
```bash
# Ensure backend is running on port 5000
# Check Chatbot.js has correct URL: http://localhost:5000
```

**Problem**: Feedback buttons not working
```bash
# Check browser console for errors
# Verify axios is installed: npm list axios
```

### Database Issues

**Problem**: Feedback not saving
```bash
# Check database file exists
ls Backend/feedback.db

# Check logs for errors
# Look for: [ERROR] Error saving feedback
```

## 📚 Documentation

- **[Backend RL Guide](Backend/RL_GUIDE.md)** - Complete backend documentation
- **[Frontend RL Guide](FRONTEND_RL_GUIDE.md)** - Frontend integration guide
- **[Complete Summary](COMPLETE_RL_SUMMARY.md)** - Full implementation overview
- **[Quick Reference](QUICK_REFERENCE.md)** - Quick start guide
- **[System Flow](SYSTEM_FLOW_DIAGRAM.md)** - Architecture diagrams

## 🎯 Best Practices

1. **Collect Diverse Feedback**
   - Encourage both positive and negative ratings
   - Aim for balanced feedback (not all 👍 or all 👎)

2. **Monitor Performance**
   - Check `/feedback/stats` regularly
   - Watch logs for reward model updates

3. **Tune Settings**
   - Start with default `n_candidates: 4`
   - Adjust based on response time vs quality needs

4. **Manage Costs**
   - RL mode makes 4x API calls
   - Toggle RL off for cost-sensitive scenarios

5. **Clean Old Data**
   - Periodically remove old feedback (90+ days)
   - Prevents database bloat

## 🚀 Deployment

### Production Considerations

1. **Environment Variables**
   - Set `ENVIRONMENT=PROD` in `.env`
   - Use production-grade API keys

2. **Database**
   - Consider PostgreSQL for production
   - Implement backup strategy

3. **Caching**
   - Add Redis for response caching
   - Cache model predictions

4. **Load Balancing**
   - Use Gunicorn or uWSGI for Flask
   - Multiple worker processes

5. **Monitoring**
   - Set up logging to file
   - Add error tracking (Sentry, etc.)

### Build for Production

```bash
# Build React app
npm run build

# Serve with production server
# (e.g., Nginx, Apache, or serve package)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Groq** - Fast LLM inference
- **Sentence Transformers** - Text embeddings
- **Create React App** - Frontend scaffolding
- **Scikit-learn** - Machine learning
- **Flask** - Web framework

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the troubleshooting section above

## 🔄 Recent Updates (v2.0)

- ✨ Added Reinforcement Learning system
- ✨ User feedback collection (thumbs up/down)
- ✨ Multi-candidate response generation
- ✨ Reward model for quality scoring
- ✨ RL mode toggle in UI
- ✨ SQLite feedback database
- ✨ Comprehensive logging system
- ✨ Centralized YAML configuration
- 📚 Complete documentation suite
- 🧪 Test suite included

---

**Built with ❤️ for better financial decision-making**

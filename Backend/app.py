import os
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Import custom modules
from config import DefaultConfig
from utils.logger import CustomLogger
from utils.database import FeedbackDatabase
from utils.reward_model import RewardModel
from utils.helpers import format_message_as_html, remove_html_tags
from utils.stock_data import StockDataFetcher
import json

# Initialize logger
logger = CustomLogger()
logger.info("=" * 80)
logger.info("Financial Chatbot - Backend Initialization")
logger.info("=" * 80)

# Load environment and configuration
logger.info("Loading environment variables")
load_dotenv()

logger.info("Initializing default configuration")
DefaultConfig.initialise()
config = DefaultConfig.bot_config
logger.info("Configuration loaded successfully")

# Initialize Groq client
logger.info("Initializing Groq client")
groq_client = Groq()
logger.info("Groq client initialized")

# Initialize Flask app
logger.info("Initializing Flask application")
app = Flask(__name__)
CORS(app)
logger.info("Flask app initialized with CORS enabled")

# Initialize database
db_path = os.path.join(os.path.dirname(__file__), config.get('database.path', 'feedback.db'))
logger.info(f"Initializing feedback database at: {db_path}")
feedback_db = FeedbackDatabase(db_path, logger)
logger.info("Feedback database initialized")

# Initialize stock data fetcher
logger.info("Initializing stock data fetcher")
stock_fetcher = StockDataFetcher(logger)
logger.info("Stock data fetcher initialized successfully")

# Initialize reward model if RL is enabled
reward_model = None
if config.is_rl_enabled():
    try:
        logger.info("RL is enabled - initializing reward model")
        reward_model = RewardModel(config, logger)
        logger.info("Reward model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize reward model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.warning("Continuing without RL capabilities")
else:
    logger.info("RL is disabled - skipping reward model initialization")

parsed_knowledge_base = ''  # This holds the parsed knowledge base content
logger.info("Backend initialization complete")
logger.info("=" * 80)


def parse_with_groq(content, parse_description):
    try:
        logger.info("Starting Groq parsing operation")
        logger.info(f"Parse description: {parse_description}")
        logger.info(f"Content length: {len(content)} characters")
        
        # Use prompt template from config
        prompt_template = config.get_prompt('parsing_prompt_template')
        if not prompt_template:
            # Fallback if not in config
            logger.warning("parsing_prompt_template not found in config, using fallback")
            input_prompt = f"""
            Parse the following content and return it formatted as HTML (use <b> for bold, <br> for line breaks, and <ul> for bullet points):
            \"\"\"{content}\"\"\"
            \n\n{parse_description}
            """
        else:
            input_prompt = prompt_template.format(
                content=content,
                parse_description=parse_description
            )

        logger.info("Calling Groq API for content parsing")
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": input_prompt}],
            model=config.get_model_config('name', 'llama-3.1-8b-instant')
        )

        parsed_content = response.choices[0].message.content.strip()
        logger.info(f"Parsing completed successfully, output length: {len(parsed_content)} characters")
        return parsed_content

    except Exception as e:
        logger.error(f"Error parsing content: {str(e)}")
        logger.error(f"Parse description was: {parse_description}")
        return f"Error parsing content: {str(e)}"


def generate_session_title(question: str) -> str:
    """
    Generate a concise session title from the first question using LLM.
    Max 50 characters.
    """
    try:
        logger.info("Generating session title from first question")
        
        # Use prompt from config
        prompt_template = config.get_prompt('session_title_prompt')
        if not prompt_template:
            # Fallback prompt if not in config
            logger.warning("session_title_prompt not found in config, using fallback")
            prompt_template = """Generate a very short, concise title (maximum 50 characters) for a chat session based on this first question. 
Return ONLY the title, nothing else.

Question: {question}

Title:"""
        
        prompt = prompt_template.format(question=question)
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.get_model_config('name', 'llama-3.1-8b-instant'),
            temperature=0.5,
            max_tokens=30
        )
        
        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"\'')
        # Truncate to 50 chars
        title = title[:50]
        
        logger.info(f"Generated title: '{title}'")
        return title
        
    except Exception as e:
        logger.error(f"Error generating session title: {e}")
        # Fallback: use first few words of question
        words = question.split()[:5]
        return ' '.join(words)[:50]


def extract_stock_symbols(question):
    """
    Use LLM to extract stock symbols from user question.
    Returns tuple: (list of symbols, is_stock_query boolean)
    """
    try:
        logger.info(f"Extracting stock symbols from question: '{question[:100]}...'")
        
        prompt = config.get_prompt('stock_symbol_extraction_prompt')
        if not prompt:
            logger.warning("Stock symbol extraction prompt not found in config")
            return [], False
        
        formatted_prompt = prompt.format(question=question)
        logger.info("Formatted extraction prompt created")
        
        logger.info("Calling Groq API for symbol extraction with temperature=0.3")
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            model=config.get_model_config('name', 'llama-3.1-8b-instant'),
            temperature=0.3,  # Low temperature for consistent extraction
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"Symbol extraction raw response: {result_text}")
        
        # Parse JSON response
        import json
        # Try to find JSON in the response
        if '{' in result_text and '}' in result_text:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            json_str = result_text[json_start:json_end]
            logger.info(f"Extracted JSON string: {json_str}")
            
            result = json.loads(json_str)
            
            symbols = result.get('symbols', [])
            is_stock_query = result.get('is_stock_query', False)
            
            logger.info(f"Successfully extracted - symbols: {symbols}, is_stock_query: {is_stock_query}")
            return symbols, is_stock_query
        else:
            logger.warning(f"No JSON found in LLM response: {result_text}")
            return [], False
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in symbol extraction: {e}")
        logger.error(f"Problematic text: {result_text}")
        return [], False
    except Exception as e:
        logger.error(f"Error extracting stock symbols: {e}")
        logger.error(f"Question was: {question}")
        return [], False


# Helper function to generate multiple candidate responses
def generate_candidates(question, history, knowledge_base_prompt, n_candidates, stock_context=""):
    """Generate multiple candidate responses with varying parameters"""
    logger.info(f"Starting candidate generation - requested: {n_candidates} candidates")
    logger.info(f"Question length: {len(question)} chars, History length: {len(history)} chars")
    logger.info(f"Stock context {'present' if stock_context else 'not present'}")
    
    candidates = []
    system_prompt = config.get_prompt('system_prompt')
    model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
    
    # Get temperature range for diversity
    temp_min = config.get_rl_config('temperature_min', 0.7)
    temp_max = config.get_rl_config('temperature_max', 1.2)
    logger.info(f"Temperature range: {temp_min} to {temp_max}")
    
    # Build the appropriate prompt with stock context if available
    if stock_context:
        logger.info("Building stock-specific prompt with real-time data")
        prompt_template = config.get_prompt('stock_financial_prompt_template')
        if not prompt_template:
            # Fallback to regular financial prompt with stock context
            logger.info("Stock prompt template not found, using fallback financial prompt")
            prompt_template = config.get_prompt('financial_prompt_template')
            prompt = prompt_template.format(
                knowledge_base_prompt=f"{knowledge_base_prompt}\n\nREAL-TIME STOCK DATA:\n{stock_context}",
                history=history,
                question=question
            )
        else:
            prompt = prompt_template.format(
                stock_context=stock_context,
                knowledge_base_prompt=knowledge_base_prompt,
                history=history,
                question=question
            )
    elif not history:
        logger.info("No history present, using general question prompt")
        prompt_template = config.get_prompt('general_question_prompt')
        prompt = prompt_template.format(question=question)
    else:
        logger.info("Using financial prompt template with history")
        prompt_template = config.get_prompt('financial_prompt_template')
        prompt = prompt_template.format(
            knowledge_base_prompt=knowledge_base_prompt,
            history=history,
            question=question
        )
    
    logger.info(f"Generating {n_candidates} candidate responses with model: {model_name}")
    
    for i in range(n_candidates):
        try:
            # Vary temperature for diversity
            temperature = temp_min + (temp_max - temp_min) * (i / max(1, n_candidates - 1))
            logger.info(f"Generating candidate {i+1}/{n_candidates} with temperature={temperature:.3f}")
            
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            answer = response.choices[0].message.content.strip()
            candidates.append(answer)
            logger.info(f"Candidate {i+1} generated successfully (length: {len(answer)} chars)")
            
        except Exception as e:
            logger.error(f"Error generating candidate {i+1}/{n_candidates}: {e}")
            logger.warning("Continuing with remaining candidates")
            # Continue with other candidates even if one fails
    
    logger.info(f"Successfully generated {len(candidates)}/{n_candidates} candidates")
    return candidates


def select_best_candidate(question, candidates):
    """Use reward model to select the best candidate response"""
    logger.info(f"Selecting best candidate from {len(candidates)} options")
    
    if not candidates:
        logger.warning("No candidates provided for selection")
        return None
    
    if len(candidates) == 1:
        logger.info("Only one candidate available, returning it")
        return candidates[0]
    
    if reward_model is None or not reward_model.is_ready():
        logger.warning("Reward model not ready, selecting first candidate as fallback")
        return candidates[0]
    
    try:
        logger.info("Formatting candidates for reward model evaluation")
        # Format texts for reward model
        texts = [f"Q: {question}\nA: {candidate}" for candidate in candidates]
        
        # Get scores from reward model
        logger.info("Requesting scores from reward model")
        scores = reward_model.predict_scores(texts)
        
        # Select candidate with highest score
        best_idx = scores.index(max(scores))
        best_candidate = candidates[best_idx]
        
        logger.info(f"Selected candidate {best_idx+1}/{len(candidates)} with score {scores[best_idx]:.4f}")
        logger.info(f"All candidate scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(f"Best candidate length: {len(best_candidate)} chars")
        
        return best_candidate
        
    except Exception as e:
        logger.error(f"Error in candidate selection: {e}")
        logger.warning("Falling back to first candidate due to error")
        return candidates[0]


# Endpoint to ask a question using the knowledge base
@app.route('/ask', methods=['POST'])
def ask_question():
    global parsed_knowledge_base
    logger.info("=" * 60)
    logger.info("Received POST request to /ask endpoint")
    
    question = request.json.get('question')
    history = request.json.get('history', '')
    use_rl = request.json.get('use_rl', config.is_rl_enabled())  # Allow override
    session_id = request.json.get('session_id')  # Get session ID if provided
    
    logger.info(f"Question received: '{question[:100]}...' (length: {len(question) if question else 0})")
    logger.info(f"History length: {len(history)} chars")
    logger.info(f"RL mode: {'enabled' if use_rl else 'disabled'}")
    logger.info(f"Session ID: {session_id if session_id else 'None'}")
    
    if not question:
        logger.error("Question parameter missing from request")
        return jsonify({"error": "Question is required"}), 400

    # Check if the knowledge base is empty or not
    if parsed_knowledge_base:
        knowledge_base_prompt = f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
        logger.info(f"Using knowledge base (length: {len(parsed_knowledge_base)} chars)")
    else:
        knowledge_base_prompt = ""
        logger.info("No knowledge base available")

    try:
        # Extract stock symbols from the question
        logger.info("Starting stock symbol extraction process")
        symbols, is_stock_query = extract_stock_symbols(question)
        stock_context = ""
        
        # Fetch real-time stock data if this is a stock query
        if is_stock_query and symbols:
            logger.info(f"Stock query detected - symbols: {symbols}")
            stock_data_parts = []
            
            for symbol in symbols:
                logger.info(f"Fetching stock data for symbol: {symbol}")
                stock_info_str = stock_fetcher.format_stock_context(symbol, include_historical=True)
                if stock_info_str and "Unable to fetch" not in stock_info_str:
                    stock_data_parts.append(stock_info_str)
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"Could not fetch data for symbol: {symbol}")
            
            if stock_data_parts:
                stock_context = "\n\n".join(stock_data_parts)
                logger.info(f"Stock context compiled for {len(stock_data_parts)} symbols (length: {len(stock_context)} chars)")
            else:
                logger.warning("No stock data could be fetched for any symbol")
        else:
            logger.info(f"Not a stock query (is_stock_query={is_stock_query}, symbols={symbols})")
        
        # Use RL-based generation if enabled and reward model is available
        if use_rl and reward_model is not None:
            logger.info("Using RL-based response generation")
            n_candidates = config.get_rl_config('n_candidates', 4)
            logger.info(f"Will generate {n_candidates} candidates for RL selection")
            
            # Generate multiple candidates with stock context if available
            candidates = generate_candidates(
                question, 
                history, 
                knowledge_base_prompt, 
                n_candidates,
                stock_context=stock_context
            )
            
            if not candidates:
                logger.error("Failed to generate any candidate responses")
                return jsonify({"error": "Failed to generate any responses"}), 500
            
            # Select best candidate using reward model
            logger.info("Selecting best candidate from generated options")
            answer = select_best_candidate(question, candidates)
            
            logger.info("RL-based response generation completed successfully")
        else:
            # Standard single response generation
            logger.info("Using standard (non-RL) response generation")
            system_prompt = config.get_prompt('system_prompt')
            model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
            logger.info(f"Model: {model_name}")
            
            # Build prompt - use stock-specific prompt if we have stock data
            if stock_context:
                logger.info("Building stock-specific prompt")
                prompt_template = config.get_prompt('stock_financial_prompt_template')
                if not prompt_template:
                    # Fallback to regular financial prompt
                    logger.info("Stock template not found, using fallback financial prompt")
                    prompt_template = config.get_prompt('financial_prompt_template')
                    prompt = prompt_template.format(
                        knowledge_base_prompt=f"{knowledge_base_prompt}\n\nREAL-TIME STOCK DATA:\n{stock_context}",
                        history=history,
                        question=question
                    )
                else:
                    prompt = prompt_template.format(
                        stock_context=stock_context,
                        knowledge_base_prompt=knowledge_base_prompt,
                        history=history,
                        question=question
                    )
            elif not history:
                logger.info("Building general question prompt (no history)")
                prompt_template = config.get_prompt('general_question_prompt')
                prompt = prompt_template.format(question=question)
            else:
                logger.info("Building financial prompt with history")
                prompt_template = config.get_prompt('financial_prompt_template')
                prompt = prompt_template.format(
                    knowledge_base_prompt=knowledge_base_prompt,
                    history=history,
                    question=question
                )
            
            logger.info("Calling Groq API for response generation")
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Standard response generated successfully (length: {len(answer)} chars)")

        # Format the answer as HTML with line breaks and bullet points
        logger.info("Formatting answer as HTML")
        formatted_answer = format_message_as_html(answer)
        
        logger.info("Updating and summarizing conversation history")
        updated_history = f"{history}\nHuman: {question}\nAI: {formatted_answer}"
        summarized_history = summarize_conversation(updated_history)
        
        # Handle session management
        is_new_session = False
        if session_id:
            # Check if this is a new session (first message)
            session = feedback_db.get_session(session_id)
            if not session:
                # Create new session with generated title
                logger.info("Creating new session")
                title = generate_session_title(question)
                feedback_db.create_session(session_id, title)
                is_new_session = True
            
            # Save user message
            feedback_db.save_message(session_id, 'user', question, rl_used=False)
            # Save assistant message
            feedback_db.save_message(session_id, 'assistant', formatted_answer, 
                                   rl_used=(use_rl and reward_model is not None))
            logger.info(f"Messages saved to session {session_id}")
        
        logger.info("Successfully processed /ask request")
        logger.info("=" * 60)
        
        response_data = {
            "answer": formatted_answer, 
            "summarized_history": summarized_history,
            "rl_used": use_rl and reward_model is not None,
            "stock_symbols": symbols if is_stock_query else []
        }
        
        if is_new_session and session_id:
            # Return session info for new sessions
            session = feedback_db.get_session(session_id)
            response_data["session"] = session
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Critical error in /ask endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500


# Function to summarize conversation using Groq API
# Removes HTML tags before summarizing
def summarize_conversation(conversation, max_tokens=1000):
    try:
        logger.info("Starting conversation summarization")
        logger.info(f"Input conversation length: {len(conversation)} chars")
        
        # Remove any HTML tags from the conversation
        cleaned_conversation = remove_html_tags(conversation)
        logger.info(f"Cleaned conversation length: {len(cleaned_conversation)} chars")
        
        # Get configuration
        model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
        summarization_prompt = config.get_prompt('summarization_prompt')
        logger.info(f"Using model: {model_name} for summarization")

        # Generate summary without HTML tags
        logger.info("Calling Groq API for summarization")
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": summarization_prompt},
                {"role": "user", "content": cleaned_conversation}
            ]
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"Summarization complete (output length: {len(summary)} chars)")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}")
        logger.error(f"Conversation length was: {len(conversation)} chars")
        return "Error summarizing conversation"


# RL Feedback Endpoints

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback on a response.
    Expected JSON: {
        "question": str,
        "answer": str,
        "rating": int (0 or 1),
        "session_id": str (optional)
    }
    """
    try:
        logger.info("=" * 60)
        logger.info("Received POST request to /feedback endpoint")
        
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        rating = data.get('rating')
        session_id = data.get('session_id')
        
        logger.info(f"Feedback data - Question length: {len(question) if question else 0}")
        logger.info(f"Feedback data - Answer length: {len(answer) if answer else 0}")
        logger.info(f"Feedback data - Rating: {rating}")
        logger.info(f"Feedback data - Session ID: {session_id}")
        
        if not question or not answer or rating is None:
            logger.error("Missing required fields in feedback submission")
            return jsonify({"error": "question, answer, and rating are required"}), 400
        
        # Validate rating
        try:
            rating_int = int(rating)
            if rating_int not in (0, 1):
                logger.error(f"Invalid rating value: {rating_int} (must be 0 or 1)")
                return jsonify({"error": "rating must be 0 (negative) or 1 (positive)"}), 400
        except (ValueError, TypeError):
            logger.error(f"Invalid rating type: {type(rating)}")
            return jsonify({"error": "rating must be an integer (0 or 1)"}), 400
        
        # Save feedback to database
        logger.info("Saving feedback to database")
        feedback_id = feedback_db.save_feedback(
            question=question,
            answer=answer,
            rating=rating_int,
            session_id=session_id
        )
        
        logger.info(f"Feedback saved successfully with ID: {feedback_id}")
        
        # Asynchronously update reward model if RL is enabled
        if reward_model is not None and config.is_rl_enabled():
            logger.info("RL enabled - triggering async reward model update")
            threading.Thread(target=async_update_reward_model, daemon=True).start()
        else:
            logger.info("RL disabled or reward model unavailable - skipping model update")
        
        logger.info("Feedback submission completed successfully")
        logger.info("=" * 60)
        
        return jsonify({
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Feedback recorded successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        return jsonify({"error": str(e)}), 500


def async_update_reward_model():
    """Background task to update reward model with new feedback"""
    try:
        logger.info("Starting async reward model update")
        min_samples = config.get_rl_config('reward_model.min_samples_for_training', 20)
        logger.info(f"Minimum samples required for training: {min_samples}")
        
        # Get training data from database
        logger.info("Retrieving feedback data from database")
        texts, labels = feedback_db.get_feedback_for_training(min_samples=0)
        
        logger.info(f"Retrieved {len(texts)} training samples from database")
        
        if len(texts) < min_samples:
            logger.info(f"Insufficient samples for training: {len(texts)}/{min_samples} - skipping update")
            return
        
        logger.info(f"Sufficient samples available - updating reward model with {len(texts)} samples")
        reward_model.update(texts, labels)
        logger.info("Reward model update completed successfully")
        
    except Exception as e:
        logger.error(f"Error updating reward model: {e}")
        logger.error(f"Error type: {type(e).__name__}")


@app.route('/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get statistics about feedback data"""
    try:
        logger.info("Received GET request to /feedback/stats endpoint")
        
        logger.info("Retrieving feedback statistics from database")
        stats = feedback_db.get_feedback_stats()
        logger.info(f"Database stats: {stats}")
        
        # Add reward model info if available
        if reward_model is not None:
            logger.info("Adding reward model statistics")
            stats['reward_model_ready'] = reward_model.is_ready()
            stats['reward_model_training_count'] = reward_model.get_training_count()
            logger.info(f"Reward model ready: {stats['reward_model_ready']}, training count: {stats['reward_model_training_count']}")
        else:
            logger.info("No reward model available")
            stats['reward_model_ready'] = False
            stats['reward_model_training_count'] = 0
        
        stats['rl_enabled'] = config.is_rl_enabled()
        logger.info(f"RL enabled: {stats['rl_enabled']}")
        
        logger.info("Successfully compiled feedback statistics")
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate_candidates', methods=['POST'])
def generate_candidates_endpoint():
    """
    Generate multiple candidate responses (for testing/debugging).
    Expected JSON: {
        "question": str,
        "history": str (optional),
        "n": int (optional, defaults to config value)
    }
    """
    try:
        logger.info("=" * 60)
        logger.info("Received POST request to /generate_candidates endpoint")
        
        data = request.json
        question = data.get('question')
        history = data.get('history', '')
        n = data.get('n', config.get_rl_config('n_candidates', 4))
        
        logger.info(f"Question: '{question[:100] if question else None}...'")
        logger.info(f"History length: {len(history)} chars")
        logger.info(f"Number of candidates requested: {n}")
        
        if not question:
            logger.error("Question parameter missing")
            return jsonify({"error": "question is required"}), 400
        
        # Check knowledge base
        if parsed_knowledge_base:
            knowledge_base_prompt = f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
            logger.info(f"Using knowledge base (length: {len(parsed_knowledge_base)} chars)")
        else:
            knowledge_base_prompt = ""
            logger.info("No knowledge base available")
        
        # Generate candidates
        logger.info(f"Generating {n} candidates")
        candidates = generate_candidates(question, history, knowledge_base_prompt, n)
        logger.info(f"Successfully generated {len(candidates)} candidates")
        
        # Get scores if reward model is available
        scores = None
        if reward_model is not None and reward_model.is_ready():
            logger.info("Reward model available - calculating scores")
            texts = [f"Q: {question}\nA: {c}" for c in candidates]
            scores = reward_model.predict_scores(texts)
            logger.info(f"Scores calculated: {scores}")
        else:
            logger.info("Reward model not available - no scores calculated")
        
        logger.info("Successfully completed candidate generation")
        logger.info("=" * 60)
        
        return jsonify({
            "candidates": candidates,
            "scores": scores,
            "count": len(candidates)
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating candidates: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        return jsonify({"error": str(e)}), 500


# Chat Session Management Endpoints

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions."""
    try:
        logger.info("Received GET request to /sessions endpoint")
        sessions = feedback_db.get_all_sessions()
        logger.info(f"Returning {len(sessions)} sessions")
        
        return jsonify({"sessions": sessions}), 200
        
    except Exception as e:
        logger.error(f"Error retrieving sessions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific session with all its messages."""
    try:
        logger.info(f"Received GET request to /sessions/{session_id} endpoint")
        
        session = feedback_db.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        messages = feedback_db.get_session_messages(session_id)
        
        # Format messages for frontend
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "rlUsed": bool(msg["rl_used"])
            })
        
        logger.info(f"Returning session with {len(formatted_messages)} messages")
        
        return jsonify({
            "session": session,
            "messages": formatted_messages
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session."""
    try:
        logger.info(f"Received DELETE request to /sessions/{session_id} endpoint")
        
        success = feedback_db.delete_session(session_id)
        
        if success:
            logger.info(f"Session {session_id} deleted successfully")
            return jsonify({"status": "success", "message": "Session deleted"}), 200
        else:
            return jsonify({"error": "Failed to delete session"}), 500
        
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    logger.info("Received GET request to /health endpoint")
    
    rl_enabled = config.is_rl_enabled()
    reward_model_ready = reward_model.is_ready() if reward_model else False
    
    logger.info(f"Health check - RL enabled: {rl_enabled}")
    logger.info(f"Health check - Reward model ready: {reward_model_ready}")
    logger.info("Health check - Database connected: True")
    
    return jsonify({
        "status": "healthy",
        "rl_enabled": rl_enabled,
        "reward_model_ready": reward_model_ready,
        "database_connected": True
    }), 200


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Starting Financial Chatbot with RL capabilities")
    logger.info(f"RL Mode: {'Enabled' if config.is_rl_enabled() else 'Disabled'}")
    if reward_model:
        training_count = reward_model.get_training_count()
        logger.info(f"Reward model initialized and trained on {training_count} samples")
        logger.info(f"Reward model ready: {reward_model.is_ready()}")
    else:
        logger.warning("Reward model not initialized - RL features unavailable")
    
    logger.info("Stock data fetcher initialized and ready")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Flask app running in {'DEBUG' if True else 'PRODUCTION'} mode")
    logger.info("=" * 80)
    
    app.run(debug=True)

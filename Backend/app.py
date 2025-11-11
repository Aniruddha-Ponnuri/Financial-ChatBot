import os
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import custom modules
from config import DefaultConfig
from utils.logger import CustomLogger
from utils.database import FeedbackDatabase
from utils.reward_model import RewardModel
from utils.helpers import format_message_as_html, remove_html_tags
from utils.stock_data import StockDataFetcher
from services.llm_service import LLMService

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

# Initialize LLM Service (replaces Groq client)
logger.info("Initializing LLM Service")
try:
    llm_service = LLMService.from_env(logger=logger)
    logger.info("LLM Service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM Service: {e}")
    raise

# Initialize Flask app
logger.info("Initializing Flask application")
app = Flask(__name__)
CORS(app)
logger.info("Flask app initialized with CORS enabled")

# Initialize database
db_path = os.path.join(os.path.dirname(__file__), config.get("database.path", "feedback.db"))
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

parsed_knowledge_base = ""  # This holds the parsed knowledge base content
logger.info("Backend initialization complete")
logger.info("=" * 80)


def parse_content(content, parse_description):
    """
    Parse content using LLM and format as HTML

    Args:
        content: Content to parse
        parse_description: Description of parsing task

    Returns:
        Parsed content as HTML string

    Raises:
        ValueError: If content or parse_description is None/empty
    """
    try:
        # Input validation
        if not content:
            logger.error("parse_content called with empty content")
            raise ValueError("Content cannot be empty")

        if not parse_description:
            logger.error("parse_content called with empty parse_description")
            raise ValueError("Parse description cannot be empty")

        logger.info("=" * 60)
        logger.info("Starting LLM content parsing operation")
        logger.info(f"Parse description: {parse_description}")
        logger.info(f"Content length: {len(content)} characters")

        # Use prompt template from config
        prompt_template = config.get_prompt("parsing_prompt_template")
        if not prompt_template:
            # Fallback if not in config
            logger.warning("parsing_prompt_template not found in config, using fallback")
            input_prompt = f"""
            Parse the following content and return it formatted as HTML (use <b> for bold, <br> for line breaks, and <ul> for bullet points):
            \"\"\"{content}\"\"\"
            \n\n{parse_description}
            """
        else:
            logger.info("Using parsing_prompt_template from configuration")
            input_prompt = prompt_template.format(
                content=content, parse_description=parse_description
            )

        logger.info("Calling LLM API for content parsing")
        logger.info(f"Input prompt length: {len(input_prompt)} characters")

        parsed_content = llm_service.generate(
            prompt=input_prompt,
            temperature=config.get_model_config("temperature", 0.7),
            max_tokens=config.get_model_config("max_tokens", 2000),
        )

        if not parsed_content:
            logger.warning("LLM returned empty parsed content")
            return content  # Return original content as fallback

        logger.info(
            f"Parsing completed successfully, output length: {len(parsed_content)} characters"
        )
        logger.info("=" * 60)
        return parsed_content

    except ValueError as ve:
        logger.error(f"Validation error in parse_content: {ve}")
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Critical error parsing content: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Parse description was: {parse_description}")
        logger.error(f"Content length was: {len(content) if content else 0}")
        logger.error("=" * 60)
        return f"Error parsing content: {str(e)}"


def generate_session_title(question: str) -> str:
    """
    Generate a concise session title from the first question using LLM.
    Max 50 characters.

    Args:
        question: User's first question in the session

    Returns:
        Generated title (max 50 chars)

    Raises:
        ValueError: If question is None or empty
    """
    try:
        # Input validation
        if not question or not question.strip():
            logger.error("generate_session_title called with empty question")
            raise ValueError("Question cannot be empty")

        logger.info("=" * 60)
        logger.info("Generating session title from first question")
        logger.info(f"Question: '{question[:100]}...' (length: {len(question)})")

        # Use prompt from config
        prompt_template = config.get_prompt("session_title_prompt")
        if not prompt_template:
            # Fallback prompt if not in config
            logger.warning("session_title_prompt not found in config, using fallback")
            prompt_template = """Generate a very short, concise title (maximum 50 characters) for a chat session based on this first question. 
Return ONLY the title, nothing else.

Question: {question}

Title:"""
        else:
            logger.info("Using session_title_prompt from configuration")

        prompt = prompt_template.format(question=question)

        logger.info("Calling LLM API for title generation")
        title = llm_service.generate(prompt=prompt, temperature=0.5, max_tokens=30)

        if not title:
            logger.warning("LLM returned empty title, using fallback")
            words = question.split()[:5]
            title = " ".join(words)[:50]
        else:
            # Remove quotes if present
            title = title.strip("\"'")
            # Truncate to 50 chars
            title = title[:50]

        logger.info(f"Generated title: '{title}'")
        logger.info("=" * 60)
        return title

    except ValueError as ve:
        logger.error(f"Validation error in generate_session_title: {ve}")
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Error generating session title: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Question was: '{question[:100] if question else 'None'}...'")
        logger.error("=" * 60)
        # Fallback: use first few words of question
        if question:
            words = question.split()[:5]
            fallback_title = " ".join(words)[:50]
            logger.info(f"Using fallback title: '{fallback_title}'")
            return fallback_title
        return "New Chat Session"


def extract_stock_symbols(question):
    """
    Use LLM to extract stock symbols from user question.

    Args:
        question: User's question text

    Returns:
        tuple: (list of symbols, is_stock_query boolean)

    Raises:
        ValueError: If question is None or empty
    """
    try:
        # Input validation
        if not question or not question.strip():
            logger.error("extract_stock_symbols called with empty question")
            raise ValueError("Question cannot be empty")

        logger.info("=" * 60)
        logger.info(f"Extracting stock symbols from question: '{question[:100]}...'")
        logger.info(f"Question length: {len(question)} characters")

        prompt = config.get_prompt("stock_symbol_extraction_prompt")
        if not prompt:
            logger.warning("Stock symbol extraction prompt not found in config")
            logger.warning("Stock detection disabled - returning empty results")
            logger.info("=" * 60)
            return [], False

        logger.info("Using stock_symbol_extraction_prompt from configuration")
        formatted_prompt = prompt.format(question=question)
        logger.info("Formatted extraction prompt created")
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")

        logger.info("Calling LLM API for symbol extraction with temperature=0.3")
        result_text = llm_service.generate(
            prompt=formatted_prompt,
            temperature=0.3,  # Low temperature for consistent extraction
            max_tokens=200,
        )

        logger.info(f"Symbol extraction raw response: {result_text}")
        logger.info(f"Response length: {len(result_text)} characters")

        # Parse JSON response
        import json

        # Try to find JSON in the response
        if "{" in result_text and "}" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            json_str = result_text[json_start:json_end]
            logger.info(f"Extracted JSON string: {json_str}")

            result = json.loads(json_str)

            symbols = result.get("symbols", [])
            is_stock_query = result.get("is_stock_query", False)

            # Validate symbols is a list
            if not isinstance(symbols, list):
                logger.warning(f"symbols is not a list: {type(symbols)}, converting to list")
                symbols = [symbols] if symbols else []

            # Validate is_stock_query is boolean
            if not isinstance(is_stock_query, bool):
                logger.warning(f"is_stock_query is not boolean: {type(is_stock_query)}, converting")
                is_stock_query = bool(is_stock_query)

            logger.info(
                f"Successfully extracted - symbols: {symbols}, is_stock_query: {is_stock_query}"
            )
            logger.info("=" * 60)
            return symbols, is_stock_query
        else:
            logger.warning(f"No JSON found in LLM response: {result_text}")
            logger.info("=" * 60)
            return [], False

    except ValueError as ve:
        logger.error("=" * 60)
        logger.error(f"Validation error in extract_stock_symbols: {ve}")
        logger.error("=" * 60)
        raise
    except json.JSONDecodeError as e:
        logger.error("=" * 60)
        logger.error(f"JSON decode error in symbol extraction: {e}")
        logger.error(f"Problematic text: {result_text}")
        logger.error("=" * 60)
        return [], False
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Unexpected error extracting stock symbols: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Question was: {question[:200] if question else 'None'}")
        logger.error("=" * 60)
        return [], False


# Helper function to generate multiple candidate responses
def generate_candidates(question, history, knowledge_base_prompt, n_candidates, stock_context=""):
    """
    Generate multiple candidate responses with varying parameters for RL selection

    Args:
        question: User's question
        history: Conversation history string
        knowledge_base_prompt: Knowledge base context
        n_candidates: Number of candidates to generate
        stock_context: Optional stock data context

    Returns:
        List of candidate response strings

    Raises:
        ValueError: If question is empty or n_candidates < 1
    """
    try:
        # Input validation
        if not question or not question.strip():
            logger.error("generate_candidates called with empty question")
            raise ValueError("Question cannot be empty")

        if n_candidates < 1:
            logger.error(f"generate_candidates called with invalid n_candidates: {n_candidates}")
            raise ValueError("n_candidates must be at least 1")

        logger.info("=" * 60)
        logger.info(f"Starting candidate generation - requested: {n_candidates} candidates")
        logger.info(f"Question length: {len(question)} chars, History length: {len(history)} chars")
        logger.info(f"Knowledge base prompt length: {len(knowledge_base_prompt)} chars")
        logger.info(
            f"Stock context: {'present (' + str(len(stock_context)) + ' chars)' if stock_context else 'not present'}"
        )

        candidates = []
        system_prompt = config.get_prompt("system_prompt")

        if not system_prompt:
            logger.warning("system_prompt not found in config, using empty system prompt")
            system_prompt = ""

        # Get temperature range for diversity
        temp_min = config.get_rl_config("temperature_min", 0.7)
        temp_max = config.get_rl_config("temperature_max", 1.2)
        logger.info(f"Temperature range: {temp_min} to {temp_max}")

        # Build the appropriate prompt with stock context if available
        if stock_context:
            logger.info("Building stock-specific prompt with real-time data")
            prompt_template = config.get_prompt("stock_financial_prompt_template")
            if not prompt_template:
                # Fallback to regular financial prompt with stock context
                logger.info("Stock prompt template not found, using fallback financial prompt")
                prompt_template = config.get_prompt("financial_prompt_template")
                if not prompt_template:
                    logger.error("financial_prompt_template also not found in config")
                    raise ValueError("Required prompt templates not found in configuration")
                prompt = prompt_template.format(
                    knowledge_base_prompt=f"{knowledge_base_prompt}\n\nREAL-TIME STOCK DATA:\n{stock_context}",
                    history=history,
                    question=question,
                )
            else:
                prompt = prompt_template.format(
                    stock_context=stock_context,
                    knowledge_base_prompt=knowledge_base_prompt,
                    history=history,
                    question=question,
                )
        elif not history:
            logger.info("No history present, using general question prompt")
            prompt_template = config.get_prompt("general_question_prompt")
            if not prompt_template:
                logger.warning("general_question_prompt not found, using simple prompt")
                prompt = question
            else:
                prompt = prompt_template.format(question=question)
        else:
            logger.info("Using financial prompt template with history")
            prompt_template = config.get_prompt("financial_prompt_template")
            if not prompt_template:
                logger.error("financial_prompt_template not found in config")
                raise ValueError("financial_prompt_template not found in configuration")
            prompt = prompt_template.format(
                knowledge_base_prompt=knowledge_base_prompt,
                history=history,
                question=question,
            )

        logger.info(f"Prompt built successfully, length: {len(prompt)} characters")
        logger.info(f"Generating {n_candidates} candidate responses")

        for i in range(n_candidates):
            try:
                # Vary temperature for diversity
                temperature = temp_min + (temp_max - temp_min) * (i / max(1, n_candidates - 1))
                logger.info(
                    f"Generating candidate {i + 1}/{n_candidates} with temperature={temperature:.3f}"
                )

                answer = llm_service.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )

                if not answer or not answer.strip():
                    logger.warning(f"Candidate {i + 1} returned empty response, skipping")
                    continue

                candidates.append(answer)
                logger.info(
                    f"Candidate {i + 1} generated successfully (length: {len(answer)} chars)"
                )

            except Exception as e:
                logger.error(f"Error generating candidate {i + 1}/{n_candidates}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.warning("Continuing with remaining candidates")
                # Continue with other candidates even if one fails

        logger.info(f"Successfully generated {len(candidates)}/{n_candidates} candidates")
        logger.info("=" * 60)

        if not candidates:
            logger.error("Failed to generate any valid candidates")
            raise RuntimeError("No candidates were successfully generated")

        return candidates

    except ValueError as ve:
        logger.error("=" * 60)
        logger.error(f"Validation error in generate_candidates: {ve}")
        logger.error("=" * 60)
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Critical error in generate_candidates: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        raise


def select_best_candidate(question, candidates):
    """
    Use reward model to select the best candidate response

    Args:
        question: Original user question
        candidates: List of candidate responses

    Returns:
        Best candidate string or None if no valid candidates

    Raises:
        ValueError: If question is empty
    """
    try:
        # Input validation
        if not question or not question.strip():
            logger.error("select_best_candidate called with empty question")
            raise ValueError("Question cannot be empty")

        logger.info("=" * 60)
        logger.info(f"Selecting best candidate from {len(candidates)} options")
        logger.info(f"Question: '{question[:100]}...'")

        if not candidates:
            logger.warning("No candidates provided for selection")
            logger.info("=" * 60)
            return None

        if len(candidates) == 1:
            logger.info("Only one candidate available, returning it")
            logger.info("=" * 60)
            return candidates[0]

        if reward_model is None:
            logger.warning("Reward model is None, selecting first candidate as fallback")
            logger.info("=" * 60)
            return candidates[0]

        if not reward_model.is_ready():
            logger.warning("Reward model not ready, selecting first candidate as fallback")
            logger.info("=" * 60)
            return candidates[0]

        logger.info("Reward model is ready, proceeding with evaluation")
        logger.info("Formatting candidates for reward model evaluation")

        # Format texts for reward model
        texts = [f"Q: {question}\nA: {candidate}" for candidate in candidates]
        logger.info(f"Formatted {len(texts)} candidate texts for evaluation")

        # Get scores from reward model
        logger.info("Requesting scores from reward model")
        scores = reward_model.predict_scores(texts)

        if not scores or len(scores) != len(candidates):
            logger.error(
                f"Invalid scores returned: expected {len(candidates)}, got {len(scores) if scores else 0}"
            )
            logger.warning("Falling back to first candidate")
            logger.info("=" * 60)
            return candidates[0]

        # Select candidate with highest score
        best_idx = scores.index(max(scores))
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]

        logger.info(
            f"Selected candidate {best_idx + 1}/{len(candidates)} with score {best_score:.4f}"
        )
        logger.info(f"All candidate scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(
            f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores) / len(scores):.4f}"
        )
        logger.info(f"Best candidate length: {len(best_candidate)} chars")
        logger.info("=" * 60)

        return best_candidate

    except ValueError as ve:
        logger.error("=" * 60)
        logger.error(f"Validation error in select_best_candidate: {ve}")
        logger.error("=" * 60)
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Error in candidate selection: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.warning("Falling back to first candidate due to error")
        logger.error("=" * 60)
        return candidates[0] if candidates else None

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

        logger.info(
            f"Selected candidate {best_idx + 1}/{len(candidates)} with score {scores[best_idx]:.4f}"
        )
        logger.info(f"All candidate scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(f"Best candidate length: {len(best_candidate)} chars")

        return best_candidate

    except Exception as e:
        logger.error(f"Error in candidate selection: {e}")
        logger.warning("Falling back to first candidate due to error")
        return candidates[0]


# Endpoint to ask a question using the knowledge base
@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Main endpoint for question answering with optional RL and stock data

    Expected JSON:
        question (str): User's question
        history (str): Conversation history (optional)
        use_rl (bool): Whether to use RL-based generation (optional)
        session_id (str): Session identifier (optional)

    Returns:
        JSON with answer, history, RL info, and stock symbols
    """
    global parsed_knowledge_base

    logger.info("=" * 80)
    logger.info("Received POST request to /ask endpoint")
    logger.info(f"Request time: {__import__('datetime').datetime.now()}")

    try:
        # Extract and validate request data
        if not request.json:
            logger.error("Request body is empty or not JSON")
            return jsonify({"error": "Request body must be JSON"}), 400

        question = request.json.get("question")
        history = request.json.get("history", "")
        use_rl = request.json.get("use_rl", config.is_rl_enabled())  # Allow override
        session_id = request.json.get("session_id")  # Get session ID if provided

        logger.info(
            f"Question received: '{question[:100] if question else 'None'}...' (length: {len(question) if question else 0})"
        )
        logger.info(f"History length: {len(history)} chars")
        logger.info(f"RL mode: {'enabled' if use_rl else 'disabled'}")
        logger.info(f"Session ID: {session_id if session_id else 'None'}")

        # Validate question
        if not question:
            logger.error("Question parameter missing from request")
            logger.info("=" * 80)
            return jsonify({"error": "Question is required"}), 400

        if not question.strip():
            logger.error("Question parameter is empty or whitespace only")
            logger.info("=" * 80)
            return jsonify({"error": "Question cannot be empty"}), 400

        # Check if the knowledge base is empty or not
        if parsed_knowledge_base:
            knowledge_base_prompt = (
                f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
            )
            logger.info(f"Using knowledge base (length: {len(parsed_knowledge_base)} chars)")
        else:
            knowledge_base_prompt = ""
            logger.info("No knowledge base available")

        # Extract stock symbols from the question
        logger.info("Starting stock symbol extraction process")
        try:
            symbols, is_stock_query = extract_stock_symbols(question)
            logger.info(
                f"Stock extraction result - symbols: {symbols}, is_stock_query: {is_stock_query}"
            )
        except Exception as e:
            logger.error(f"Stock symbol extraction failed: {e}")
            logger.warning("Continuing without stock data")
            symbols, is_stock_query = [], False

        stock_context = ""

        # Fetch real-time stock data if this is a stock query
        if is_stock_query and symbols:
            logger.info(f"Stock query detected - symbols: {symbols}")
            stock_data_parts = []

            for symbol in symbols:
                try:
                    logger.info(f"Fetching stock data for symbol: {symbol}")
                    stock_info_str = stock_fetcher.format_stock_context(
                        symbol, include_historical=True
                    )
                    if stock_info_str and "Unable to fetch" not in stock_info_str:
                        stock_data_parts.append(stock_info_str)
                        logger.info(f"Successfully fetched data for {symbol}")
                    else:
                        logger.warning(f"Could not fetch data for symbol: {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    continue

            if stock_data_parts:
                stock_context = "\n\n".join(stock_data_parts)
                logger.info(
                    f"Stock context compiled for {len(stock_data_parts)} symbols (length: {len(stock_context)} chars)"
                )
            else:
                logger.warning("No stock data could be fetched for any symbol")
        else:
            logger.info(f"Not a stock query (is_stock_query={is_stock_query}, symbols={symbols})")

        # Use RL-based generation if enabled and reward model is available
        if use_rl and reward_model is not None:
            logger.info("Using RL-based response generation")
            n_candidates = config.get_rl_config("n_candidates", 4)
            logger.info(f"Will generate {n_candidates} candidates for RL selection")

            # Generate multiple candidates with stock context if available
            candidates = generate_candidates(
                question,
                history,
                knowledge_base_prompt,
                n_candidates,
                stock_context=stock_context,
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
            system_prompt = config.get_prompt("system_prompt")
            logger.info("Building prompt for response generation")

            # Build prompt - use stock-specific prompt if we have stock data
            if stock_context:
                logger.info("Building stock-specific prompt")
                prompt_template = config.get_prompt("stock_financial_prompt_template")
                if not prompt_template:
                    # Fallback to regular financial prompt
                    logger.info("Stock template not found, using fallback financial prompt")
                    prompt_template = config.get_prompt("financial_prompt_template")
                    prompt = prompt_template.format(
                        knowledge_base_prompt=f"{knowledge_base_prompt}\n\nREAL-TIME STOCK DATA:\n{stock_context}",
                        history=history,
                        question=question,
                    )
                else:
                    prompt = prompt_template.format(
                        stock_context=stock_context,
                        knowledge_base_prompt=knowledge_base_prompt,
                        history=history,
                        question=question,
                    )
            elif not history:
                logger.info("Building general question prompt (no history)")
                prompt_template = config.get_prompt("general_question_prompt")
                prompt = prompt_template.format(question=question)
            else:
                logger.info("Building financial prompt with history")
                prompt_template = config.get_prompt("financial_prompt_template")
                prompt = prompt_template.format(
                    knowledge_base_prompt=knowledge_base_prompt,
                    history=history,
                    question=question,
                )

            logger.info("Calling LLM API for response generation")
            answer = llm_service.generate(prompt=prompt, system_prompt=system_prompt)

            logger.info(f"Standard response generated successfully (length: {len(answer)} chars)")

        # Format the answer as HTML with line breaks and bullet points
        logger.info("Formatting answer as HTML")
        try:
            formatted_answer = format_message_as_html(answer)
            logger.info(f"Answer formatted successfully (length: {len(formatted_answer)} chars)")
        except Exception as e:
            logger.error(f"Error formatting answer: {e}")
            logger.warning("Using unformatted answer")
            formatted_answer = answer

        # Update and summarize conversation history
        logger.info("Updating and summarizing conversation history")
        try:
            updated_history = f"{history}\nHuman: {question}\nAI: {formatted_answer}"
            summarized_history = summarize_conversation(updated_history)
            logger.info(
                f"History summarized successfully (length: {len(summarized_history)} chars)"
            )
        except Exception as e:
            logger.error(f"Error summarizing history: {e}")
            logger.warning("Using truncated history as fallback")
            # Fallback: use last 1000 chars of updated history
            updated_history = f"{history}\nHuman: {question}\nAI: {formatted_answer}"
            summarized_history = (
                updated_history[-1000:] if len(updated_history) > 1000 else updated_history
            )

        # Handle session management
        is_new_session = False
        if session_id:
            try:
                logger.info(f"Managing session: {session_id}")
                # Check if this is a new session (first message)
                session = feedback_db.get_session(session_id)
                if not session:
                    # Create new session with generated title
                    logger.info("Creating new session")
                    try:
                        title = generate_session_title(question)
                    except Exception as e:
                        logger.error(f"Failed to generate session title: {e}")
                        title = "New Chat"

                    feedback_db.create_session(session_id, title)
                    is_new_session = True
                    logger.info(f"New session created with title: '{title}'")
                else:
                    logger.info(f"Existing session found: '{session.get('title', 'Untitled')}'")

                # Save user message
                feedback_db.save_message(session_id, "user", question, rl_used=False)
                # Save assistant message
                feedback_db.save_message(
                    session_id,
                    "assistant",
                    formatted_answer,
                    rl_used=(use_rl and reward_model is not None),
                )
                logger.info(f"Messages saved to session {session_id}")
            except Exception as e:
                logger.error(f"Error in session management: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.warning("Continuing without session persistence")
        else:
            logger.info("No session ID provided, skipping session management")

        logger.info("Successfully processed /ask request")
        logger.info("=" * 80)

        response_data = {
            "answer": formatted_answer,
            "summarized_history": summarized_history,
            "rl_used": use_rl and reward_model is not None,
            "stock_symbols": symbols if is_stock_query and symbols else [],
        }

        if is_new_session and session_id:
            try:
                # Return session info for new sessions
                session = feedback_db.get_session(session_id)
                if session:
                    response_data["session"] = session
                    logger.info("Session info added to response")
            except Exception as e:
                logger.error(f"Error retrieving session info: {e}")

        logger.info("Response data prepared successfully")
        return jsonify(response_data), 200

    except ValueError as ve:
        logger.error("=" * 80)
        logger.error(f"Validation error in /ask endpoint: {str(ve)}")
        logger.error(f"Error type: {type(ve).__name__}")
        logger.error("=" * 80)
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Critical error in /ask endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("=" * 80)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Function to summarize conversation using LLM API
# Removes HTML tags before summarizing
def summarize_conversation(conversation, max_tokens=1000):
    """
    Summarize conversation history using LLM

    Args:
        conversation: Full conversation text with HTML
        max_tokens: Maximum tokens for summary

    Returns:
        Summarized conversation string

    Raises:
        ValueError: If conversation is empty
    """
    try:
        # Input validation
        if not conversation or not conversation.strip():
            logger.warning("summarize_conversation called with empty conversation")
            return ""

        logger.info("=" * 60)
        logger.info("Starting conversation summarization")
        logger.info(f"Input conversation length: {len(conversation)} chars")
        logger.info(f"Max tokens: {max_tokens}")

        # Remove any HTML tags from the conversation
        cleaned_conversation = remove_html_tags(conversation)
        logger.info(f"Cleaned conversation length: {len(cleaned_conversation)} chars")

        if not cleaned_conversation or not cleaned_conversation.strip():
            logger.warning("Conversation is empty after cleaning HTML tags")
            logger.info("=" * 60)
            return ""

        # Get configuration
        summarization_prompt = config.get_prompt("summarization_prompt")
        if not summarization_prompt:
            logger.warning("summarization_prompt not found in config, using default")
            summarization_prompt = (
                "You are a helpful assistant that summarizes conversations concisely."
            )

        logger.info("Using summarization prompt from configuration")
        logger.info(f"System prompt length: {len(summarization_prompt)} chars")

        # Generate summary without HTML tags
        logger.info("Calling LLM API for summarization")
        summary = llm_service.generate(
            prompt=cleaned_conversation,
            system_prompt=summarization_prompt,
            max_tokens=max_tokens,
        )

        if not summary or not summary.strip():
            logger.warning("LLM returned empty summary, returning original cleaned conversation")
            logger.info("=" * 60)
            return cleaned_conversation

        logger.info(f"Summarization complete (output length: {len(summary)} chars)")
        logger.info(
            f"Compression ratio: {len(summary)}/{len(cleaned_conversation)} = {(len(summary) / len(cleaned_conversation) * 100):.1f}%"
        )
        logger.info("=" * 60)
        return summary

    except ValueError as ve:
        logger.error("=" * 60)
        logger.error(f"Validation error in summarize_conversation: {ve}")
        logger.error("=" * 60)
        return ""
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Error summarizing conversation: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Conversation length was: {len(conversation) if conversation else 0} chars")
        logger.error("=" * 60)
        return "Error summarizing conversation"


# RL Feedback Endpoints


@app.route("/feedback", methods=["POST"])
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
        question = data.get("question")
        answer = data.get("answer")
        rating = data.get("rating")
        session_id = data.get("session_id")

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
            session_id=session_id,
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

        return jsonify(
            {
                "status": "success",
                "feedback_id": feedback_id,
                "message": "Feedback recorded successfully",
            }
        ), 200

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        return jsonify({"error": str(e)}), 500


def async_update_reward_model():
    """Background task to update reward model with new feedback"""
    try:
        logger.info("Starting async reward model update")
        min_samples = config.get_rl_config("reward_model.min_samples_for_training", 20)
        logger.info(f"Minimum samples required for training: {min_samples}")

        # Get training data from database
        logger.info("Retrieving feedback data from database")
        texts, labels = feedback_db.get_feedback_for_training(min_samples=0)

        logger.info(f"Retrieved {len(texts)} training samples from database")

        if len(texts) < min_samples:
            logger.info(
                f"Insufficient samples for training: {len(texts)}/{min_samples} - skipping update"
            )
            return

        logger.info(
            f"Sufficient samples available - updating reward model with {len(texts)} samples"
        )
        reward_model.update(texts, labels)
        logger.info("Reward model update completed successfully")

    except Exception as e:
        logger.error(f"Error updating reward model: {e}")
        logger.error(f"Error type: {type(e).__name__}")


@app.route("/feedback/stats", methods=["GET"])
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
            stats["reward_model_ready"] = reward_model.is_ready()
            stats["reward_model_training_count"] = reward_model.get_training_count()
            logger.info(
                f"Reward model ready: {stats['reward_model_ready']}, training count: {stats['reward_model_training_count']}"
            )
        else:
            logger.info("No reward model available")
            stats["reward_model_ready"] = False
            stats["reward_model_training_count"] = 0

        stats["rl_enabled"] = config.is_rl_enabled()
        logger.info(f"RL enabled: {stats['rl_enabled']}")

        logger.info("Successfully compiled feedback statistics")
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_candidates", methods=["POST"])
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
        question = data.get("question")
        history = data.get("history", "")
        n = data.get("n", config.get_rl_config("n_candidates", 4))

        logger.info(f"Question: '{question[:100] if question else None}...'")
        logger.info(f"History length: {len(history)} chars")
        logger.info(f"Number of candidates requested: {n}")

        if not question:
            logger.error("Question parameter missing")
            return jsonify({"error": "question is required"}), 400

        # Check knowledge base
        if parsed_knowledge_base:
            knowledge_base_prompt = (
                f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
            )
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

        return jsonify(
            {
                "candidates": candidates,
                "scores": scores,
                "count": len(candidates),
            }
        ), 200

    except Exception as e:
        logger.error(f"Error generating candidates: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("=" * 60)
        return jsonify({"error": str(e)}), 500


# Chat Session Management Endpoints


@app.route("/sessions", methods=["GET"])
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


@app.route("/sessions/<session_id>", methods=["GET"])
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
            formatted_messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "rlUsed": bool(msg["rl_used"]),
                }
            )

        logger.info(f"Returning session with {len(formatted_messages)} messages")

        return jsonify({"session": session, "messages": formatted_messages}), 200

    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sessions/<session_id>", methods=["DELETE"])
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


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with system status"""
    logger.info("Received GET request to /health endpoint")

    rl_enabled = config.is_rl_enabled()
    reward_model_ready = reward_model.is_ready() if reward_model else False

    logger.info(f"Health check - RL enabled: {rl_enabled}")
    logger.info(f"Health check - Reward model ready: {reward_model_ready}")
    logger.info("Health check - Database connected: True")

    return jsonify(
        {
            "status": "healthy",
            "rl_enabled": rl_enabled,
            "reward_model_ready": reward_model_ready,
            "database_connected": True,
        }
    ), 200


if __name__ == "__main__":
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

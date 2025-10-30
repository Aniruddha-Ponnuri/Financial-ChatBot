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

# Initialize logger
logger = CustomLogger()

# Load environment and configuration
load_dotenv()
DefaultConfig.initialise()
config = DefaultConfig.bot_config

# Initialize Groq client
groq_client = Groq()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize database
db_path = os.path.join(os.path.dirname(__file__), config.get('database.path', 'feedback.db'))
feedback_db = FeedbackDatabase(db_path, logger)

# Initialize reward model if RL is enabled
reward_model = None
if config.is_rl_enabled():
    try:
        reward_model = RewardModel(config, logger)
        logger.info("Reward model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize reward model: {e}")
        logger.warning("Continuing without RL capabilities")

parsed_knowledge_base = ''  # This holds the parsed knowledge base content


def parse_with_groq(content, parse_description):
    try:
        # Modified prompt to return the content formatted in HTML
        input_prompt = f"""
        Parse the following content and return it formatted as HTML (use <b> for bold, <br> for line breaks, and <ul> for bullet points):
        \"\"\"{content}\"\"\"
        \n\n{parse_description}
        """

        # Call the Groq API without splitting the content
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": input_prompt}],
            model="llama-3.1-8b-instant"
        )

        # Extract and return the parsed content from the response
        parsed_content = response.choices[0].message.content.strip()
        logger.info("Parsing completed successfully")
        return parsed_content

    except Exception as e:
        logger.error(f"Error parsing content: {str(e)}")
        return f"Error parsing content: {str(e)}"

# Helper function to generate multiple candidate responses
def generate_candidates(question, history, knowledge_base_prompt, n_candidates):
    """Generate multiple candidate responses with varying parameters"""
    candidates = []
    system_prompt = config.get_prompt('system_prompt')
    model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
    
    # Get temperature range for diversity
    temp_min = config.get_rl_config('temperature_min', 0.7)
    temp_max = config.get_rl_config('temperature_max', 1.2)
    
    # Build the appropriate prompt
    if not history:
        prompt_template = config.get_prompt('general_question_prompt')
        prompt = prompt_template.format(question=question)
    else:
        prompt_template = config.get_prompt('financial_prompt_template')
        prompt = prompt_template.format(
            knowledge_base_prompt=knowledge_base_prompt,
            history=history,
            question=question
        )
    
    logger.info(f"Generating {n_candidates} candidate responses")
    
    for i in range(n_candidates):
        try:
            # Vary temperature for diversity
            temperature = temp_min + (temp_max - temp_min) * (i / max(1, n_candidates - 1))
            
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
            logger.info(f"Generated candidate {i+1}/{n_candidates} (temp={temperature:.2f})")
            
        except Exception as e:
            logger.error(f"Error generating candidate {i+1}: {e}")
            # Continue with other candidates even if one fails
    
    return candidates


def select_best_candidate(question, candidates):
    """Use reward model to select the best candidate response"""
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    if reward_model is None or not reward_model.is_ready():
        logger.warning("Reward model not ready, selecting random candidate")
        return candidates[0]
    
    try:
        # Format texts for reward model
        texts = [f"Q: {question}\nA: {candidate}" for candidate in candidates]
        
        # Get scores from reward model
        scores = reward_model.predict_scores(texts)
        
        # Select candidate with highest score
        best_idx = scores.index(max(scores))
        best_candidate = candidates[best_idx]
        
        logger.info(f"Selected candidate {best_idx+1} with score {scores[best_idx]:.3f}")
        logger.info(f"All scores: {[f'{s:.3f}' for s in scores]}")
        
        return best_candidate
        
    except Exception as e:
        logger.error(f"Error in candidate selection: {e}")
        return candidates[0]


# Endpoint to ask a question using the knowledge base
@app.route('/ask', methods=['POST'])
def ask_question():
    global parsed_knowledge_base
    question = request.json.get('question')
    history = request.json.get('history', '')
    use_rl = request.json.get('use_rl', config.is_rl_enabled())  # Allow override
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Check if the knowledge base is empty or not
    if parsed_knowledge_base:
        knowledge_base_prompt = f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
    else:
        knowledge_base_prompt = ""

    try:
        # Use RL-based generation if enabled and reward model is available
        if use_rl and reward_model is not None:
            n_candidates = config.get_rl_config('n_candidates', 4)
            
            # Generate multiple candidates
            candidates = generate_candidates(question, history, knowledge_base_prompt, n_candidates)
            
            if not candidates:
                return jsonify({"error": "Failed to generate any responses"}), 500
            
            # Select best candidate using reward model
            answer = select_best_candidate(question, candidates)
            
            logger.info("Used RL-based response generation")
        else:
            # Standard single response generation
            system_prompt = config.get_prompt('system_prompt')
            model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
            
            # Build prompt
            if not history:
                prompt_template = config.get_prompt('general_question_prompt')
                prompt = prompt_template.format(question=question)
            else:
                prompt_template = config.get_prompt('financial_prompt_template')
                prompt = prompt_template.format(
                    knowledge_base_prompt=knowledge_base_prompt,
                    history=history,
                    question=question
                )
            
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Used standard response generation")

        # Format the answer as HTML with line breaks and bullet points
        formatted_answer = format_message_as_html(answer)
        
        updated_history = f"{history}\nHuman: {question}\nAI: {formatted_answer}"
        summarized_history = summarize_conversation(updated_history)
        
        return jsonify({
            "answer": formatted_answer, 
            "summarized_history": summarized_history,
            "rl_used": use_rl and reward_model is not None
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500


# Function to summarize conversation using Groq API
# Removes HTML tags before summarizing
def summarize_conversation(conversation, max_tokens=1000):
    try:
        # Remove any HTML tags from the conversation
        cleaned_conversation = remove_html_tags(conversation)
        
        # Get configuration
        model_name = config.get_model_config('name', 'llama-3.1-8b-instant')
        summarization_prompt = config.get_prompt('summarization_prompt')

        # Generate summary without HTML tags
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": summarization_prompt},
                {"role": "user", "content": cleaned_conversation}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}")
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
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        rating = data.get('rating')
        session_id = data.get('session_id')
        
        if not question or not answer or rating is None:
            return jsonify({"error": "question, answer, and rating are required"}), 400
        
        # Validate rating
        try:
            rating_int = int(rating)
            if rating_int not in (0, 1):
                return jsonify({"error": "rating must be 0 (negative) or 1 (positive)"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "rating must be an integer (0 or 1)"}), 400
        
        # Save feedback to database
        feedback_id = feedback_db.save_feedback(
            question=question,
            answer=answer,
            rating=rating_int,
            session_id=session_id
        )
        
        logger.info(f"Feedback saved with ID: {feedback_id}")
        
        # Asynchronously update reward model if RL is enabled
        if reward_model is not None and config.is_rl_enabled():
            threading.Thread(target=async_update_reward_model, daemon=True).start()
        
        return jsonify({
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Feedback recorded successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({"error": str(e)}), 500


def async_update_reward_model():
    """Background task to update reward model with new feedback"""
    try:
        min_samples = config.get_rl_config('reward_model.min_samples_for_training', 20)
        
        # Get training data from database
        texts, labels = feedback_db.get_feedback_for_training(min_samples=0)
        
        if len(texts) < min_samples:
            logger.info(f"Not enough samples for training yet: {len(texts)}/{min_samples}")
            return
        
        logger.info(f"Updating reward model with {len(texts)} samples")
        reward_model.update(texts, labels)
        logger.info("Reward model updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating reward model: {e}")


@app.route('/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get statistics about feedback data"""
    try:
        stats = feedback_db.get_feedback_stats()
        
        # Add reward model info if available
        if reward_model is not None:
            stats['reward_model_ready'] = reward_model.is_ready()
            stats['reward_model_training_count'] = reward_model.get_training_count()
        else:
            stats['reward_model_ready'] = False
            stats['reward_model_training_count'] = 0
        
        stats['rl_enabled'] = config.is_rl_enabled()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
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
        data = request.json
        question = data.get('question')
        history = data.get('history', '')
        n = data.get('n', config.get_rl_config('n_candidates', 4))
        
        if not question:
            return jsonify({"error": "question is required"}), 400
        
        # Check knowledge base
        if parsed_knowledge_base:
            knowledge_base_prompt = f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
        else:
            knowledge_base_prompt = ""
        
        # Generate candidates
        candidates = generate_candidates(question, history, knowledge_base_prompt, n)
        
        # Get scores if reward model is available
        scores = None
        if reward_model is not None and reward_model.is_ready():
            texts = [f"Q: {question}\nA: {c}" for c in candidates]
            scores = reward_model.predict_scores(texts)
        
        return jsonify({
            "candidates": candidates,
            "scores": scores,
            "count": len(candidates)
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating candidates: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system status"""
    return jsonify({
        "status": "healthy",
        "rl_enabled": config.is_rl_enabled(),
        "reward_model_ready": reward_model.is_ready() if reward_model else False,
        "database_connected": True
    }), 200


if __name__ == '__main__':
    logger.info("Starting Financial Chatbot with RL capabilities")
    logger.info(f"RL Mode: {'Enabled' if config.is_rl_enabled() else 'Disabled'}")
    if reward_model:
        logger.info(f"Reward model trained on {reward_model.get_training_count()} samples")
    app.run(debug=True)

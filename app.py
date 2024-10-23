import os
import logging
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

groq_client = Groq()
app = Flask(__name__)
CORS(app)

parsed_knowledge_base = ''  # This holds the parsed knowledge base content


# Helper function to format the message into HTML-friendly format
def format_message_as_html(message):
    formatted_message = message.replace("\n", "<br>")
    return formatted_message

# Helper function to remove HTML tags for summarization
def remove_html_tags(text):
    clean = re.compile('<.*?>')  # Regex to remove HTML tags
    return re.sub(clean, '', text)

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
        logging.info("Parsing completed successfully")
        return parsed_content

    except Exception as e:
        logging.error(f"Error parsing content: {str(e)}")
        return f"Error parsing content: {str(e)}"

# Endpoint to ask a question using the knowledge base
@app.route('/ask', methods=['POST'])
def ask_question():
    global parsed_knowledge_base
    question = request.json.get('question')
    history = request.json.get('history', '')

    # Check if the knowledge base is empty or not
    if parsed_knowledge_base:
        knowledge_base_prompt = f"Here is some knowledge that can help:\n{parsed_knowledge_base}\n\n"
    else:
        knowledge_base_prompt = ""

    # If history is empty, give a general response
    if not history:
        prompt = f"""
        You are an AI assistant. Answer the following general financial question based on your knowledge And use data from India:
        
        Financial question: {question}
        """
    else:
        # Tailored prompt for financial assistance, incorporating the knowledge base if available
        prompt = f"""
        You are an AI financial assistant. Use your knowledge base and the following financial data to provide informed financial decisions and analysis.
        And use data from India
        
        {knowledge_base_prompt}
        
        Previous conversation summary:
        {history}
        
        Financial question or data: {question}
        
        Please consider the following aspects in your response:
        - Market trends
        - Investment risk
        - Growth potential
        - Financial ratios and analysis
        - Any other financial insights to help in decision making
        """

    try:
        # Call Groq API to generate the answer using groq_client
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial assistant that provides analysis and decisions based on financial data and conversation history."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content.strip()

        # Format the answer as HTML with line breaks and bullet points
        formatted_answer = format_message_as_html(answer)
        
        updated_history = f"{history}\nHuman: {question}\nAI: {formatted_answer}"
        summarized_history = summarize_conversation(updated_history)
        
        return jsonify({"answer": formatted_answer, "summarized_history": summarized_history}), 200
    except Exception as e:
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500


# Function to summarize conversation using Groq API
# Removes HTML tags before summarizing
def summarize_conversation(conversation, max_tokens=1000):
    try:
        # Remove any HTML tags from the conversation
        cleaned_conversation = remove_html_tags(conversation)

        # Generate summary without HTML tags
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize the following financial conversation concisely:"},
                {"role": "user", "content": cleaned_conversation}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing conversation: {str(e)}")
        return "Error summarizing conversation"

if __name__ == '__main__':
    app.run(debug=True)

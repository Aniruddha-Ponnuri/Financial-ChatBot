import os
import logging
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
app = Flask(__name__)
CORS(app)  

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

extracted_text = ''  
parsed_knowledge_base = ''  

def split_content(content, max_length=6000):
    return [
        content[i:i + max_length] for i in range(0, len(content), max_length)
    ]

def parse_with_chatgpt(dom_chunks, parse_description):
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        try:
            input_prompt = f"Parse the following content:\n\"\"\"{chunk}\"\"\"\n\n{parse_description}"
            completion = openai_client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": input_prompt}],
                max_tokens=1500, 
                temperature=0.01, 
                n=1,
                stop=None
            )
            parsed_content = completion.choices[0].message.content.strip()
            parsed_results.append(parsed_content)
            
            logging.info(f"Parsed batch: {i} of {len(dom_chunks)}")
        
        except Exception as e:
            logging.error(f"Error parsing chunk {i}: {str(e)}")
            parsed_results.append(f"Error parsing chunk {i}: {str(e)}")

    return "\n".join(parsed_results)

# Endpoint to upload a PDF and extract its content, split, and parse it
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global extracted_text, parsed_knowledge_base
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file uploaded"}), 400
    
    pdf_file = request.files['file']
    
    if pdf_file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        logging.error("File is not a PDF")
        return jsonify({"error": "Uploaded file is not a PDF"}), 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
            pdf_file.save(temp_path)
        
        logging.info(f"File saved temporarily at {temp_path}")
        with pdfplumber.open(temp_path) as pdf:
            extracted_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        os.unlink(temp_path)
        
        if not extracted_text:
            logging.warning("No text extracted from the PDF")
            return jsonify({"error": "Failed to extract text from the PDF"}), 500
        
        logging.info("PDF content extracted successfully")
        
        # Split the extracted text into chunks
        chunks = split_content(extracted_text)
        logging.info(f"PDF content split into {len(chunks)} chunks")
        
        # Parse the chunks using ChatGPT API
        parsed_knowledge_base = parse_with_chatgpt(chunks, "Parse the PDF content to create a concise knowledge base for answering user questions.")
        logging.info("Parsed knowledge base created")
        
        return jsonify({"message": "PDF content extracted and parsed successfully."}), 200
    
    except Exception as e:
        logging.exception(f"Error processing PDF: {str(e)}")
        return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500

# Endpoint to ask a question using the knowledge base (your version)
@app.route('/ask', methods=['POST'])
def ask_question():
    global parsed_knowledge_base
    question = request.json.get('question')
    history = request.json.get('history', '')
   
    if not parsed_knowledge_base:
        return jsonify({"error": "No parsed knowledge base available. Please upload a PDF first."}), 400
   
    prompt = f"""
    Use the knowledge base created earlier to answer the following question:
    
    Previous conversation summary:
    {history}
    
    Question: {question}
    """
   
    try:
        # Call OpenAI to generate the answer using openai_client.chat.completions.create
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the knowledge base and conversation history."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.01,
            n=1,
            stop=None
        )
        
        # Accessing the message content from the API response
        answer = completion.choices[0].message.content.strip()
        
        # Update conversation history and summarize
        updated_history = f"{history}\nHuman: {question}\nAI: {answer}"
        summarized_history = summarize_conversation(updated_history)
        
        return jsonify({"answer": answer, "summarized_history": summarized_history}), 200
    except Exception as e:
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500

# Function to summarize conversation using ChatGPT API
def summarize_conversation(conversation, max_tokens=1000):
    try:
        summary = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely:"},
                {"role": "user", "content": conversation}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return summary.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing conversation: {str(e)}")
        return "Error summarizing conversation"

if __name__ == '__main__':
    app.run(debug=True)

import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# This line loads the API keys from your .env file
load_dotenv()

# --- Configuration ---
app = Flask(__name__)

# Securely load API keys from the environment
SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Base url's 
SUPERMEMORY_BASE_URL = "https://api.supermemory.ai/v3"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


# --- API Client Functions ---

def add_memory_to_supermemory(content_string, source_filename):
    """
    Adds a document by sending its content as a string within a JSON payload.
    """
    print(f"Sending content from '{source_filename}' to Supermemory as JSON...")
    headers = {
        "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "content": content_string,
        "source": source_filename
    }
    response = requests.post(f"{SUPERMEMORY_BASE_URL}/memories", json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(f"Successfully posted content. Supermemory response: {result}")
    return result

def search_supermemory(query):
    """
    Searches for documents using a POST request as per the documentation.
    """
    print(f"Searching Supermemory documents (v3) for: '{query}'")
    headers = {
        "Authorization": f"Bearer {SUPERMEMORY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "limit": 1
    }
    response = requests.post(f"{SUPERMEMORY_BASE_URL}/search", json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print("Received search results from Supermemory.")
    return result.get("results", [])

def generate_cohesive_answer(question, context, source):
    """
    Uses the Google Gemini API to generate a natural, cohesive answer.
    """
    print("Sending retrieved context to Gemini for answer generation...")
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""
    You are an expert assistant. Your task is to answer the user's question based *only* on the following context retrieved from the document named '{source}'.
    Do not use any outside knowledge. If the answer is not in the context, state that clearly.

    **Context:**
    ---
    {context}
    ---

    **User's Question:** {question}

    **Answer:**
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    print("Received generated answer from Gemini.")
    return result['candidates'][0]['content']['parts'][0]['text']

# --- Flask Web Routes ---

@app.route('/')
def index():
    if not SUPERMEMORY_API_KEY or not GEMINI_API_KEY:
        return "<h1>Configuration Error: API keys are missing. Please check your .env file.</h1>", 500
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file_route():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    try:
        content_string = file.read().decode('utf-8', errors='ignore')
        add_memory_to_supermemory(content_string, file.filename)
        return jsonify({"success": True, "message": f"'{file.filename}' was successfully added to Supermemory."})
    except requests.exceptions.HTTPError as e:
        return jsonify({"success": False, "error": f"API Error: {e.response.status_code} - {e.response.text}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_route():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        search_results = search_supermemory(question)
        if not search_results or not search_results[0].get('chunks'):
            return jsonify({"answer": "I'm sorry, I couldn't find any information in your documents related to that question."})

        top_result = search_results[0]
        top_chunk = top_result['chunks'][0]
        context = top_chunk.get('content', '')
        source = top_result.get('title', 'an uploaded document')

        final_answer = generate_cohesive_answer(question, context, source)
        return jsonify({"answer": final_answer})
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"API Error: {e.response.status_code} - {e.response.text}"}), 500
    except (IndexError, KeyError) as e:
        return jsonify({"error": f"Could not parse the search response from Supermemory. Error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
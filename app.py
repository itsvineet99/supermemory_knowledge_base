import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from supermemory import Supermemory

# This line loads the API keys from your .env file
load_dotenv()

# --- Configuration ---
app = Flask(__name__)

# Securely load API keys from the environment
SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


try:
    supermemory_client = Supermemory(api_key=SUPERMEMORY_API_KEY)
except Exception as e:
    print(f"Error initializing Supermemory client: {e}")
    supermemory_client = None

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"


# --- API Client Functions ---

def upload_document_to_supermemory(file):
    filename = file.filename
    print(f"Uploading '{filename}' as a Document object via the SDK...")
    
    # Read the file content into bytes
    file_content = file.read()
    
    result = supermemory_client.memories.upload_file(
        file=(filename, file_content),
        container_tags=[filename]
    )
    
    print(f"Successfully submitted Document for processing. Response: {result}")
    return result

def search_with_supermemory(query):
    print(f"Searching Supermemory documents (v3) via SDK for: '{query}'")
    
    results = supermemory_client.search.documents(
        q=query,
        limit=3,
        document_threshold=0.5,
        chunk_threshold=0.5,
        rerank=True,
        include_summary=True,
    )
    
    print("Received search results from Supermemory SDK.")
    return results


def generate_cohesive_answer(question, context, source):
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
    if not supermemory_client or not GEMINI_API_KEY:
        return "<h1>Configuration Error: Client failed to initialize. Please check your API keys.</h1>", 500
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file_route():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    try:
        upload_document_to_supermemory(file)
        return jsonify({"success": True, "message": f"'{file.filename}' was successfully sent to Supermemory."})
    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def query_route():
    question = request.get_json().get('question')
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
    # search_results is a SearchDocumentsResponse
        search_results = search_with_supermemory(question)

        # Extract the actual list of hits
        results_list = search_results.results

        # Handle case: no results or no chunks
        if not results_list or not getattr(results_list[0], "chunks", []):
            return jsonify({
                "answer": "I'm sorry, I couldn't find any information in your documents related to that question."
            })


        for i, result in enumerate(results_list):
            chunks = getattr(result, "chunks", [])
            if chunks:
                source = getattr(result, "title", f"Source {i+1}")
                context = getattr(chunks[0], "content", "")

        final_answer = generate_cohesive_answer(question, context, source)
        return jsonify({"answer": final_answer})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
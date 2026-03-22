# app.py

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from gemini_utils import extract_text_from_pdf, get_summary, get_chat_answer

app = Flask(__name__)

# File upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store document context for Q&A
chat_context = {"text": ""}

@app.route('/')
def index():
    return render_template('index.html')  # Serve frontend

@app.route('/result', methods=['POST'])
def result():
    try:
        file = request.files['pdf']
        summary_length = request.form.get('summary_length', 'medium')
        summary_style = request.form.get('summary_style', 'concise')

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract and summarize
        full_text = extract_text_from_pdf(filepath)
        chat_context["text"] = full_text  # Save for future questions
        summary = get_summary(full_text, summary_length, summary_style)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not chat_context["text"]:
            return jsonify({"answer": "Please upload a PDF before asking questions."})

        answer = get_chat_answer(chat_context["text"], user_message)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

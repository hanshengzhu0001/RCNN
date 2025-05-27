from flask import Flask, request, render_template, jsonify
import whisper
import os
from datetime import datetime
from openai import OpenAI

app = Flask(__name__)
model = whisper.load_model("base")  #  "medium", "large-v3"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def postprocess_with_gpt(text):
    prompt = f"""
    You are a text polishing assistant. Please remove the colloquial expressions and interjections from the following transcribed speech and rewrite it in written language:
    ---
    {text}
    ---
    Please only return the polished text.
    """
    GPT_client = OpenAI(
        api_key="",
    )

    GPT_completion = GPT_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a meticulous language polishing expert."},
            {"role": "user", "content": prompt},
        ],
    )
    return GPT_completion.choices[0].message.content

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio_data"]
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = model.transcribe(filepath)
    raw_text = result["text"].strip()
    polished_text = postprocess_with_gpt(raw_text)

    return jsonify({
        "transcript": raw_text,
        "polished": polished_text
    })

if __name__ == "__main__":
    app.run(debug=True)

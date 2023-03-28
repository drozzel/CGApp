from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import gzip
import os

app = Flask(__name__)

# Load compressed model from file
model_path = "path/to/compressed/model/file.gz"
with gzip.open(model_path, "rb") as f:
    buffer = f.read()
model = torch.load(buffer)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@app.route('/', methods=['POST'])
def generate_text():
    data = request.get_json()
    input_text = data['input_text']

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response
    generated_ids = model.generate(input_ids, max_length=1000, do_sample=True)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    response = {'generated_text': generated_text}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

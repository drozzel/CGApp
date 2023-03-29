import os
import torch
from transformers import pipeline, BertTokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model_name = 'gpt2-small'
model = pipeline('text-generation', model=model_name)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get the input text from the request
    input_text = request.json['text']

    # Tokenize the input text and generate new text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids)

    # Decode the generated output and return as a JSON response
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'text': output_text})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

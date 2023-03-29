from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# load the compressed Chat GPT model and tokenizer
model_path = os.path.join(os.getcwd(), "model.bin")
tokenizer_path = os.path.join(os.getcwd(), "tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# define the chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    # get the input text from the request
    input_text = request.json['text']

    # encode the input text and generate a response
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    response = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)

    # return the response text
    return {'response': response_text}

if __name__ == '__main__':
    # use waitress as the web server
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)

from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import gzip
import shutil

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
    # compress the model and tokenizer files using gzip
    with open(model_path, 'rb') as f_in, gzip.open(model_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    with open(tokenizer_path, 'rb') as f_in, gzip.open(tokenizer_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    # remove the uncompressed model and tokenizer files
    os.remove(model_path)
    os.remove(tokenizer_path)

    # use waitress as the web server
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)

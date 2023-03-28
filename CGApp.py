from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json['input_text']

    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({'response_text': response_text})


if __name__ == '__main__':
    app.run(debug=True)

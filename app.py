from fastapi import FastAPI
from pydantic import BaseModel
from cachetools import TTLCache
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI()

class InputText(BaseModel):
    input_text: str

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
cache = TTLCache(maxsize=500, ttl=300)

@app.post("/generate")
async def generate(input_text: InputText):
    if input_text.input_text in cache:
        response_text = cache[input_text.input_text]
    else:
        input_ids = tokenizer.encode(input_text.input_text + tokenizer.eos_token, return_tensors='pt')
        output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        cache[input_text.input_text] = response_text

    return {"response_text": response_text}

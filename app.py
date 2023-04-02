from fastapi import FastAPI
from pydantic import BaseModel
from cachetools import TTLCache
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI()

class InputText(BaseModel):
    input_text: str

cache = TTLCache(maxsize=1000, ttl=300)

@app.get("/generate")
async def generate(input_text: InputText):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    if input_text.input_text in cache:
        response_text = cache[input_text.input_text]
    else:
        input_ids = tokenizer.encode(input_text.input_text + tokenizer.eos_token, return_tensors='pt')
        output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response_text": response_text}

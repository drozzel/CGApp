from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from cachetools import TTLCache
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

app = FastAPI()

class InputText(BaseModel):
    input_text: str

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

cache = TTLCache(maxsize=1000, ttl=300)

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return response_text

@app.post("/generate")
async def generate(input_text: InputText, background_tasks: BackgroundTasks):
    if input_text.input_text in cache:
        response_text = cache[input_text.input_text]
    else:
        background_tasks.add_task(cache.__setitem__, input_text.input_text, await generate_response(input_text.input_text))
        response_text = "Generating response, please try again in a few seconds"
    return {"response_text": response_text}

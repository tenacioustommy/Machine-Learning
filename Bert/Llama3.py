from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
model_id = "Llama3-8B-Chinese:latest"
client = OpenAI(base_url="http://localhost:11434/v1",api_key='Llama3-8B-Chinese:latest')
messages = [
    {"role": "user", "content": "写一首诗吧"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

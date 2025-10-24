from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8881/v1", # Your vLLM server URL
    api_key="EMPTY"
)

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 5% of 234?"}
    ],
    temperature=0.6,
    max_tokens=1220, # uses the default thinking budget provided to to vllm server via server.sh script
)

print("*" * 100)
print(result, "\n")

result = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about GPUs, it should start with praising CPUs then eventually explain GPUS and praise GPUs because they have caused an AI revolution. Dont think too much."}
    ],
    temperature=1.0,
    max_tokens=1220,
    logprobs=True,
    extra_body={
        "vllm_xargs": {
            "thinking_budget": 50,
            "thinking_budget_grace_period": 30,
            "end_token_ids": json.dumps([1010, 1010, 13]),
        }
    }
)

print("*" * 100)
print(result, "\n")

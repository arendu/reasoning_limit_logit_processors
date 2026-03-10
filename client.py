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
    logprobs=True,
    temperature=0.6,
    max_tokens=1220, # uses the default thinking budget provided to to vllm server via server.sh script
)

print("*" * 100)
print(result, "\n")
# print the list of tokens (not token ids but the actual tokens) in a human readable format
for token_logprob in result.choices[0].logprobs.content:
    print(token_logprob.token, end="")
print("\n")
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
            "thinking_budget": 200,
            "thinking_budget_grace_period": 30,
            "end_token_ids": "\n\n I have been forced to end thinking.</think>",
        }
    }
)

print("*" * 100)
print(result, "\n")
# print the list of tokens (not token ids but the actual tokens) in a human readable format
for token_logprob in result.choices[0].logprobs.content:
    print(token_logprob.token, end="")
print("\n")
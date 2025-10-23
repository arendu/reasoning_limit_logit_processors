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
    # Standard sampling parameters go here:
    temperature=0.6,
    max_tokens=1220,
    # All non-OpenAI or custom parameters go into `extra_body`
    extra_body={
        "vllm_xargs": {
            "thinking_budget": 150,
            "thinking_budget_grace_period": 30,
            "end_token_ids": json.dumps([1871, 5565, 11483, 6139, 1046, 2259, 74045, 1062])
        }
    }
)

print(result)

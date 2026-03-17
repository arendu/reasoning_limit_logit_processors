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
        {"role": "user", "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."}
    ],
    temperature=1.0,
    max_tokens=5220,
    logprobs=True,
    extra_body={
        "vllm_xargs": {
            "thinking_budget": 700,
            "thinking_budget_grace_period": 130,
            "enable_milestones": False,
            "end_token_ids": "\n\n I have been forced to end thinking.</think>",
        }
    }
)

print("*" * 100)
# print(result, "\n")
tokens = [t.token for t in result.choices[0].logprobs.content]
full_text = "".join(tokens)
print(full_text)
print()

import re
match = re.search(r'\[Injected tokens:(\[.*?\])\]', full_text)
if match:
    injected_positions = json.loads(match.group(1))
    print(f"Found {len(injected_positions)} injected positions: {injected_positions}")
    print("-" * 60)
    for pos in injected_positions:
        if pos < len(tokens):
            print(f"  pos {pos:5d} -> {tokens[pos]!r}")
        else:
            print(f"  pos {pos:5d} -> (beyond logprobs range)")
else:
    print("No [Injected tokens:...] metadata found in output")

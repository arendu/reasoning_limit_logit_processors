from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8881/v1", # Your vLLM server URL
    api_key="EMPTY"
)

#result = client.chat.completions.create(
#    model="model",
#    messages=[
#        {"role": "system", "content": "You are a helpful assistant."},
#        {"role": "user", "content": "What is 5% of 234?"}
#    ],
#    logprobs=True,
#    temperature=0.6,
#    max_tokens=1220, # uses the default thinking budget provided to to vllm server via server.sh script
#)
#
#print("*" * 100)
#print(result, "\n")
## print the list of tokens (not token ids but the actual tokens) in a human readable format
#for token_logprob in result.choices[0].logprobs.content:
#    print(token_logprob.token, end="")
#print("\n")

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
            "thinking_budget": 400,
            "thinking_budget_grace_period": 30,
            "end_token_ids": "\n\n I have been forced to end thinking.</think>",
        }
    }
)

print("*" * 100)
# print(result, "\n")
for idx, token in enumerate(result.choices[0].logprobs.content):
    print(f"{token.token}", end="")
print("\n")
"""
Utility script to get the the token ids for a sequence of tokens.
"""
from cmath import phase
from transformers import AutoTokenizer

def phrase_to_ids(phrase: str, model_name: str = "Qwen/Qwen3-8B"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Encode phrase into token IDs
    encoded = tokenizer(phrase, add_special_tokens=False)

    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Phrase: {phrase!r}")
    print("Token IDs:", token_ids)
    print("Tokens:  ", tokens)

    return token_ids, tokens


if __name__ == "__main__":
    # Example phrases
    model_name = "nvidia/NVIDIA-Nemotron-Nano-12B-v2" # you can replace with other models from HF Qwen/Qwen3-8B for example...
    print(model_name)
    phrase_to_ids(" Reached thinking limit. </think>", model_name)
    phrase_to_ids(" </think>", model_name)
    phrase_to_ids("\n</think>\n", model_name)
    phrase_to_ids("</think>", model_name)
    phrase_to_ids("see how they fit together.\n</think>\n\n", model_name)
    phrase_to_ids("What is 2 * 4?", model_name)
    phrase_to_ids("<my_spl_token> Reached <SPECIAL_100> thinking limit. </think>", model_name)
    phrase_to_ids("\n<SPECIAL_11>Assistant\n<think>\n", model_name)

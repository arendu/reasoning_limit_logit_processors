# copied from here https://docs.vllm.ai/en/v0.10.1.1/examples/offline_inference/logits_processor.html

from types import DynamicClassAttribute
from typing import Optional, Dict, Any, List

import torch
import json
import os
import copy

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

#import os
#os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def _build_newline_tokens(tokenizer) -> set:
    """Scan the full vocab and return the set of token IDs whose decoded text ends with a newline."""
    newline_tokens = set()
    for token_id in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
            if decoded.endswith("\n\n"):
                newline_tokens.add(token_id)
        except Exception:
            continue
    print(f"Built newline token set: {len(newline_tokens)} tokens")
    return newline_tokens



class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, 
            vllm_config: VllmConfig, 
            device: torch.device, 
            is_pin_memory: bool):
        args_file = os.getenv("THINKING_BUDGET_LOGITS_PROCESSOR_ARGS_FILE")
        if args_file and os.path.exists(args_file):
            with open(args_file) as f:
                cfg_env = json.load(f)
        else:
            cfg_env = json.loads(os.getenv("THINKING_BUDGET_LOGITS_PROCESSOR_ARGS", "{}"))

        print("cfg_env in init:", cfg_env)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_env.get("model", "Qwen/Qwen3-8B"))
        self.thinking_budget = cfg_env.get("thinking_budget", -1)
        self.thinking_budget_grace_period = cfg_env.get("thinking_budget_grace_period", -1)
        self.enable_milestones = cfg_env.get("enable_milestones", False)
        self.milestone_frequency = cfg_env.get("milestone_frequency", 10)
        end_token_ids_raw = cfg_env.get("end_token_ids", [])
        prompt_think_ids_raw = cfg_env.get("prompt_think_ids", [])
        end_think_ids_raw = cfg_env.get("end_think_ids", [])

        self.end_token_ids = self._to_ids(end_token_ids_raw)
        self.prompt_think_ids = self._to_ids(prompt_think_ids_raw)
        self.end_think_ids = [self._to_ids(e) for e in end_think_ids_raw] if end_think_ids_raw and isinstance(end_think_ids_raw[0], (str, list)) else [self._to_ids(end_think_ids_raw)]

        self.newline_tokens = _build_newline_tokens(self.tokenizer)

        print(f"Resolved end_token_ids: {self.end_token_ids}")
        print(f"Resolved prompt_think_ids: {self.prompt_think_ids}")
        print(f"Resolved end_think_ids: {self.end_think_ids}")
        print(f"enable_milestones: {self.enable_milestones}")
        self.available_milestones = {i : self._get_milestone_token_ids(i) for i in list(range(self.milestone_frequency, 101-self.milestone_frequency, self.milestone_frequency))} if self.enable_milestones else {}
        for k, v in self.available_milestones.items():
            print(f"initializing available_milestones[{k}]: {v}")

        self.enable_injection_position_metadata = cfg_env.get("enable_injection_position_metadata", False)
        print(f"enable_injection_position_metadata: {self.enable_injection_position_metadata}")

        self.eod_token_ids = self._build_eod_token_ids(cfg_env) if self.enable_injection_position_metadata else set()
        print(f"EOD token IDs: {self.eod_token_ids}")

        self.logit_processor_state: dict[int, dict[Any, Any]] = {}

    def _to_ids(self, value) -> list:
        """Convert a string or list of ints to token IDs (see get_token_ids.py)."""
        if isinstance(value, str):
            return self.tokenizer(value, add_special_tokens=False)["input_ids"]
        if isinstance(value, list) and all(isinstance(v, int) for v in value):
            return value
        return value

    def _build_eod_token_ids(self, cfg_env: dict) -> set:
        """Build the set of end-of-document token IDs to intercept."""
        eod_ids = set()
        if cfg_env.get("eod_token_ids"):
            for tid in self._to_ids(cfg_env["eod_token_ids"]):
                eod_ids.add(tid)
        if self.tokenizer.eos_token_id is not None:
            eod_ids.add(self.tokenizer.eos_token_id)
        for special_name in ["<|im_end|>", "<|endoftext|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(special_name)
            if isinstance(tid, int) and tid != self.tokenizer.unk_token_id:
                eod_ids.add(tid)
        return eod_ids

    def _maybe_suppress_eod(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]) -> torch.Tensor:
        """Suppress EOD tokens unless EOD is the argmax, in which case inject delay tokens first."""
        if state["_is_delaying_eod"]:
            if state["_delay_eod_ids"]:
                tok = state["_delay_eod_ids"].pop(0)
                logits[idx, :] = float("-inf")
                logits[idx, tok] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                print(f"[ThinkingBudget] delaying EOD, injecting token={tok}, remaining={len(state['_delay_eod_ids'])}")
            else:
                eod_id = state["_original_eod_id"]
                logits[idx, :] = float("-inf")
                logits[idx, eod_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                state["_is_delaying_eod"] = False
                print(f"[ThinkingBudget] EOD delay complete, forcing EOD token={eod_id}. output_tokens={len(state.get('output_tok_ids', []))}")
            return logits

        argmax_id = logits[idx].argmax().item()
        if argmax_id in self.eod_token_ids:
            metadata_str = " [Injected tokens:" + json.dumps(state["injected_positions"]) + "]"
            state["_is_delaying_eod"] = True
            state["_original_eod_id"] = argmax_id
            state["_delay_eod_ids"] = self.tokenizer(metadata_str, add_special_tokens=False)["input_ids"]
            tok = state["_delay_eod_ids"].pop(0)
            logits[idx, :] = float("-inf")
            logits[idx, tok] = 1.0
            state["injected_positions"].append(len(state["output_tok_ids"]))
            print(f"[ThinkingBudget] EOD is argmax (token={argmax_id}), starting delay injection. metadata={metadata_str!r}")
        else:
            for eod_id in self.eod_token_ids:
                logits[idx, eod_id] = float("-inf")
        return logits

    def is_argmax_invariant(self) -> bool:
        return False  # This processor does not affect sampling

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return
        # Add new requests
        for index, sampling_params, prompt_tok_ids, output_tok_ids in batch_update.added:
            print(sampling_params.extra_args, "sampling_params.extra_args")
            state = self.logit_processor_state.get(index, {})
            state["output_tok_ids"] = output_tok_ids
            state["thinking_budget"] = self.thinking_budget
            state["thinking_budget_grace_period"] = self.thinking_budget_grace_period
            state["end_token_ids"] = self.end_token_ids
            state["is_thinking"] = False
            
            if sampling_params.extra_args:
                """
                sampling params can overwrite ones from the cfg_env
                """
                state["thinking_budget"] = sampling_params.extra_args.get("thinking_budget", self.thinking_budget)
                state["thinking_budget_grace_period"] = sampling_params.extra_args.get("thinking_budget_grace_period", self.thinking_budget_grace_period)
                if "end_token_ids" in sampling_params.extra_args:
                    state["end_token_ids"] = self._to_ids(sampling_params.extra_args["end_token_ids"])
                else:
                    state["end_token_ids"] = self.end_token_ids

            state["_injecting_milestone"] = False
            state["_is_delaying_eod"] = False
            state["_delay_eod_ids"] = []
            state["injected_positions"] = []
            if prompt_tok_ids[-len(self.prompt_think_ids):] == self.prompt_think_ids:
                print("model starting thinking via prompt_think_ids match...")
                state["is_thinking"] = True
                state["available_milestones"] = copy.deepcopy(self.available_milestones)
                state["start_of_end"] = False
                state["end_of_end"] = False
            self.logit_processor_state[index] = state
        # Remove finished requests
        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)
        # Handle moved requests
        for a, b, direction in batch_update.moved:
            a_val = self.logit_processor_state.pop(a, None)
            b_val = self.logit_processor_state.pop(b, None)
            if a_val is not None:
                self.logit_processor_state[b] = a_val
            if direction.name == "SWAP" and b_val is not None:
                self.logit_processor_state[a] = b_val
    
    def _suffix_prefix_overlap(self, a, b):
        m = min(len(a), len(b))
        for k in range(m, 0, -1):           # try longest first
            if a[-k:] == b[:k]:
                return k
        return 0

    def _get_milestone_token_ids(self, pct: int) -> list:
        """Tokenize a milestone marker like '[42% reasoning completed]'."""
        marker = f"[~{pct}% tokens consumed]"
        print(f"[ThinkingBudget] _get_milestone_token_ids: marker={marker}")
        return self.tokenizer(marker, add_special_tokens=False)["input_ids"]

    def _maybe_inject_milestone(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state.get("end_of_end") or state.get("start_of_end"):
            return logits

        # Continue injecting milestone tokens if mid-injection
        if state.get("_injecting_milestone", False):
            if len(state["_milestone_ids"]):
                milestone_tok_id = state["_milestone_ids"].pop(0)
                logits[idx, :] = float("-inf")
                logits[idx, milestone_tok_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                print(f"[ThinkingBudget] continue injecting milestone tokens remaining milestone_tokens={len(state['_milestone_ids'])}")
            else:
                print(f"[ThinkingBudget] milestone injection completed... setting _injecting_milestone to False")
                state["_injecting_milestone"] = False
            return logits


        # Check for newline while still under budget
        if not state.get("_injecting_milestone", False) and len(state["output_tok_ids"]) > 0 and state["output_tok_ids"][-1] in self.newline_tokens:
            budget = state["thinking_budget"]
            tokens_so_far = len(state["output_tok_ids"])
            pct = min(int(tokens_so_far / budget * 100), 100)
            pct_milestone = 10 * (pct // 10)
            print(f"[ThinkingBudget] idx={idx} tokens_so_far={tokens_so_far} budget={budget} pct={pct} pct_milestone={pct_milestone} available_milestones={state.get('available_milestones', {})}")
            if pct_milestone in state["available_milestones"]:
                milestone_tok_ids = state["available_milestones"].pop(pct_milestone)
                state["_milestone_ids"] = milestone_tok_ids 
                state["_injecting_milestone"] = True
                logits[idx, :] = float("-inf")
                milestone_tok_id = state["_milestone_ids"].pop(0)
                logits[idx, milestone_tok_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                print(f"[ThinkingBudget] injected milestone {pct_milestone} remaining milestone_tokens={len(state['_milestone_ids'])}")
            else:
                print(f"[ThinkingBudget] not injecting milestone {pct_milestone} because it is already injected or not in the available milestones tokens_so_far={tokens_so_far} budget={budget} pct={pct} pct_milestone={pct_milestone} available_milestones={state.get('available_milestones', [])}")

        return logits

    def _maybe_end_thinking(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state["end_of_end"]:
            return logits

        for eti in self.end_think_ids:
            check_if_think_ended_naturally = list(state["output_tok_ids"][-len(eti):])
            if check_if_think_ended_naturally == eti:
                # if thinking ends normally don't intervene...
                state["start_of_end"] = True
                state["end_of_end"] = True


        real_tokens = len(state["output_tok_ids"])
        if real_tokens >= state["thinking_budget"] + state["thinking_budget_grace_period"] and not state["start_of_end"]:
            state["start_of_end"] = True

        if real_tokens >= state["thinking_budget"] and state["output_tok_ids"][-1] in self.newline_tokens and not state["start_of_end"]:
            state["start_of_end"] = True
        
        if state["start_of_end"] and not state["end_of_end"]:
            end_token_ids = state["end_token_ids"]
            last_n_inputs = list(state["output_tok_ids"][-len(end_token_ids):])
            overlap = self._suffix_prefix_overlap(last_n_inputs, end_token_ids)
            if overlap < len(end_token_ids):
                logits[idx, :] = float("-inf") 
                insert_id = end_token_ids[overlap]
                logits[idx, insert_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
            else:
                state["end_of_end"] = True
        return logits

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, state in self.logit_processor_state.items():
            if not state.get("is_thinking", False):
                last_n_inputs = list(state["output_tok_ids"][-len(self.prompt_think_ids):])
                if last_n_inputs == self.prompt_think_ids:
                    state["is_thinking"] = True
                    state["available_milestones"] = copy.deepcopy(self.available_milestones)
                    state["start_of_end"] = False
                    state["end_of_end"] = False
                    print("model starting thinking via output tokens...")
            if state.get("is_thinking", False):
                logits = self._maybe_inject_milestone(idx, logits, state)
            if state.get("is_thinking", False):
                logits = self._maybe_end_thinking(idx, logits, state)
            if self.enable_injection_position_metadata:
                logits = self._maybe_suppress_eod(idx, logits, state)
        return logits

def main():
    model = "Qwen/Qwen3-8B"
    msg = """Bob is an avid fan of the video game \"League of Leesins\", and today he celebrates as the League of Leesins World Championship comes to an end! \n\nThe tournament consisted of $n$ ($n \\ge 5$) teams around the world. Before the tournament starts, Bob has made a prediction of the rankings of each team, from $1$-st to $n$-th. After the final, he compared the prediction with the actual result and found out that the $i$-th team according to his prediction ended up at the $p_i$-th position ($1 \\le p_i \\le n$, all $p_i$ are unique). In other words, $p$ is a permutation of $1, 2, \\dots, n$.\n\nAs Bob's favorite League player is the famous \"3ga\", he decided to write down every $3$ consecutive elements of the permutation $p$. Formally, Bob created an array $q$ of $n-2$ triples, where $q_i = (p_i, p_{i+1}, p_{i+2})$ for each $1 \\le i \\le n-2$. Bob was very proud of his array, so he showed it to his friend Alice.\n\nAfter learning of Bob's array, Alice declared that she could retrieve the permutation $p$ even if Bob rearranges the elements of $q$ and the elements within each triple. Of course, Bob did not believe in such magic, so he did just the same as above to see Alice's respond.\n\nFor example, if $n = 5$ and $p = [1, 4, 2, 3, 5]$, then the original array $q$ will be $[(1, 4, 2), (4, 2, 3), (2, 3, 5)]$. Bob can then rearrange the numbers within each triple and the positions of the triples to get $[(4, 3, 2), (2, 3, 5), (4, 1, 2)]$. Note that $[(1, 4, 2), (4, 2, 2), (3, 3, 5)]$ is not a valid rearrangement of $q$, as Bob is not allowed to swap numbers belong to different triples.\n\nAs Alice's friend, you know for sure that Alice was just trying to show off, so you decided to save her some face by giving her any permutation $p$ that is consistent with the array $q$ she was given. \n\n\n-----Input-----\n\nThe first line contains a single integer $n$ ($5 \\le n \\le 10^5$) — the size of permutation $p$.\n\nThe $i$-th of the next $n-2$ lines contains $3$ integers $q_{i, 1}$, $q_{i, 2}$, $q_{i, 3}$ ($1 \\le q_{i, j} \\le n$) — the elements of the $i$-th triple of the rearranged (shuffled) array $q_i$, in random order. Remember, that the numbers within each triple can be rearranged and also the positions of the triples can be rearranged.\n\nIt is guaranteed that there is at least one permutation $p$ that is consistent with the input. \n\n\n-----Output-----\n\nPrint $n$ distinct integers $p_1, p_2, \\ldots, p_n$ ($1 \\le p_i \\le n$) such that $p$ is consistent with array $q$. \n\nIf there are multiple answers, print any. \n\n\n-----Example-----\nInput\n5\n4 3 2\n2 3 5\n4 1 2\n\nOutput\n1 4 2 3 5\n\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"""
    messages = [{"role": "system", "content": "You are a helpful assistant. /think"},{"role": "user", "content": msg}]
    messages2= [{"role": "system", "content": "You are a helpful assistant. /think"},{"role": "user", "content": "Write a haiku about a cat"}]
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False),
            tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True, add_special_tokens=False)]

    sampling_params_list = [SamplingParams(temperature=0.6, max_tokens=1220, extra_args={"thinking_budget": 1050, "thinking_budget_grace_period": 30, "end_token_ids": " Reached thinking limit. </think>"}),
                            SamplingParams(temperature=0.6, max_tokens=1260, extra_args={"thinking_budget": 600, "thinking_budget_grace_period": 20, "end_token_ids": " </think>"})]

    llm = LLM(
            model=model,
            logits_processors=[ThinkingBudgetLogitsProcessor],
            trust_remote_code=True,
            )
    outputs = llm.generate(prompts, sampling_params_list)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

if __name__ == "__main__":
    main()

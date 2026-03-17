import logging
from typing import Optional, Dict, Any

import torch
import json
import os
import copy

from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

logger = logging.getLogger("ThinkingBudgetLogitsProcessor")


def _build_newline_tokens(tokenizer) -> set:
    """Scan the full vocab and return the set of token IDs whose decoded text ends with a double newline."""
    newline_tokens = set()
    for token_id in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([token_id])
            if decoded.endswith("\n\n"):
                newline_tokens.add(token_id)
        except Exception:
            continue
    logger.info("Built newline token set: %d tokens", len(newline_tokens))
    return newline_tokens


class ThinkingBudgetLogitsProcessorBase(LogitsProcessor):
    DEFAULT_MODEL = "Qwen/Qwen3-8B"
    DEFAULT_PROMPT_THINK_IDS = "<think>"
    DEFAULT_END_THINK_IDS = ["</think>"]

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

        logger.info("cfg_env in init: %s", cfg_env)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_env.get("model", self.DEFAULT_MODEL))
        self.thinking_budget = cfg_env.get("thinking_budget", -1)
        self.thinking_budget_grace_period = cfg_env.get("thinking_budget_grace_period", -1)
        self.enable_milestones = cfg_env.get("enable_milestones", False)
        self.milestone_frequency = cfg_env.get("milestone_frequency", 10)
        end_token_ids_raw = cfg_env.get("end_token_ids", [])
        prompt_think_ids_raw = cfg_env.get("prompt_think_ids", self.DEFAULT_PROMPT_THINK_IDS)
        end_think_ids_raw = cfg_env.get("end_think_ids", self.DEFAULT_END_THINK_IDS)

        self.end_token_ids = self._to_ids(end_token_ids_raw)
        self.prompt_think_ids = self._to_ids(prompt_think_ids_raw)
        self.end_think_ids = [self._to_ids(e) for e in end_think_ids_raw] if end_think_ids_raw and isinstance(end_think_ids_raw[0], (str, list)) else [self._to_ids(end_think_ids_raw)]

        self.newline_tokens = _build_newline_tokens(self.tokenizer)

        logger.info("Resolved end_token_ids: %s", self.end_token_ids)
        logger.info("Resolved prompt_think_ids: %s", self.prompt_think_ids)
        logger.info("Resolved end_think_ids: %s", self.end_think_ids)
        logger.info("enable_milestones: %s", self.enable_milestones)
        self.available_milestones = {i : self._get_milestone_token_ids(i) for i in list(range(self.milestone_frequency, 101-self.milestone_frequency, self.milestone_frequency))} if self.enable_milestones else {}
        for k, v in self.available_milestones.items():
            logger.info("initializing available_milestones[%d]: %s", k, v)

        self.enable_injection_position_metadata = cfg_env.get("enable_injection_position_metadata", False)
        logger.info("enable_injection_position_metadata: %s", self.enable_injection_position_metadata)

        self.eod_token_ids = self._build_eod_token_ids(cfg_env) if self.enable_injection_position_metadata else set()
        logger.info("EOD token IDs: %s", self.eod_token_ids)

        self.logit_processor_state: dict[int, dict[Any, Any]] = {}
        self._validate_config()

    def _validate_config(self):
        """Check that resolved config values are sane for this model."""
        expected_suffix = self._to_ids(self.DEFAULT_PROMPT_THINK_IDS)
        if not self.prompt_think_ids:
            raise ValueError("prompt_think_ids is empty — thinking detection will not work")
        elif self.prompt_think_ids[-len(expected_suffix):] != expected_suffix:
            actual_text = self.tokenizer.decode(self.prompt_think_ids, skip_special_tokens=False)
            expected_text = self.tokenizer.decode(expected_suffix, skip_special_tokens=False)
            raise ValueError(f"prompt_think_ids ends with {actual_text!r} but expected suffix {expected_text!r}")
        else:
            logger.info("prompt_think_ids validated")

        if not self.end_think_ids or self.end_think_ids == [[]]:
            raise ValueError("end_think_ids is empty — natural end-of-thinking detection will not work")
        else:
            all_ok = True
            for default_str in self.DEFAULT_END_THINK_IDS:
                expected_suffix = self._to_ids(default_str)
                for eti in self.end_think_ids:
                    if eti and eti[-len(expected_suffix):] != expected_suffix:
                        all_ok = False
            if not all_ok:
                raise ValueError(f"not all end_think_ids end with expected suffix from {self.DEFAULT_END_THINK_IDS!r}")
            else:
                logger.info("end_think_ids validated")

    def _to_ids(self, value) -> list:
        """Convert a string or list of ints to token IDs."""
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
                logger.debug("delaying EOD, injecting token=%d, remaining=%d", tok, len(state['_delay_eod_ids']))
            else:
                eod_id = state["_original_eod_id"]
                logits[idx, :] = float("-inf")
                logits[idx, eod_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                state["_is_delaying_eod"] = False
                logger.info("EOD delay complete, forcing EOD token=%d. output_tokens=%d", eod_id, len(state.get('output_tok_ids', [])))
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
            logger.info("EOD is argmax (token=%d), starting delay injection. metadata=%r", argmax_id, metadata_str)
        else:
            for eod_id in self.eod_token_ids:
                logits[idx, eod_id] = float("-inf")
        return logits

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return
        for index, sampling_params, prompt_tok_ids, output_tok_ids in batch_update.added:
            logger.debug("extra_args: %s", sampling_params.extra_args)
            state = self.logit_processor_state.get(index, {})
            state["output_tok_ids"] = output_tok_ids
            state["thinking_budget"] = self.thinking_budget
            state["thinking_budget_grace_period"] = self.thinking_budget_grace_period
            state["end_token_ids"] = self.end_token_ids
            state["is_thinking"] = False

            if sampling_params.extra_args:
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
                logger.info("model starting thinking via prompt_think_ids match...")
                state["is_thinking"] = True
                state["available_milestones"] = copy.deepcopy(self.available_milestones)
                state["start_of_end"] = False
                state["end_of_end"] = False
            self.logit_processor_state[index] = state
        for index in batch_update.removed:
            self.logit_processor_state.pop(index, None)
        for a, b, direction in batch_update.moved:
            a_val = self.logit_processor_state.pop(a, None)
            b_val = self.logit_processor_state.pop(b, None)
            if a_val is not None:
                self.logit_processor_state[b] = a_val
            if direction.name == "SWAP" and b_val is not None:
                self.logit_processor_state[a] = b_val

    def _suffix_prefix_overlap(self, a, b):
        m = min(len(a), len(b))
        for k in range(m, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    def _get_milestone_token_ids(self, pct: int) -> list:
        """Tokenize a milestone marker like '[~42% tokens consumed]'."""
        marker = f"[~{pct}% tokens consumed]"
        logger.debug("_get_milestone_token_ids: marker=%s", marker)
        return self.tokenizer(marker, add_special_tokens=False)["input_ids"]

    def _maybe_inject_milestone(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state.get("end_of_end") or state.get("start_of_end"):
            return logits

        if state.get("_injecting_milestone", False):
            if len(state["_milestone_ids"]):
                milestone_tok_id = state["_milestone_ids"].pop(0)
                logits[idx, :] = float("-inf")
                logits[idx, milestone_tok_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                logger.debug("continue injecting milestone tokens remaining=%d", len(state['_milestone_ids']))
            else:
                logger.debug("milestone injection completed")
                state["_injecting_milestone"] = False
            return logits

        if not state.get("_injecting_milestone", False) and len(state["output_tok_ids"]) > 0 and state["output_tok_ids"][-1] in self.newline_tokens:
            budget = state["thinking_budget"]
            tokens_so_far = len(state["output_tok_ids"])
            pct = min(int(tokens_so_far / budget * 100), 100)
            pct_milestone = self.milestone_frequency * (pct // self.milestone_frequency)
            logger.debug("idx=%d tokens_so_far=%d budget=%d pct=%d pct_milestone=%d", idx, tokens_so_far, budget, pct, pct_milestone)
            if pct_milestone in state["available_milestones"]:
                milestone_tok_ids = state["available_milestones"].pop(pct_milestone)
                state["_milestone_ids"] = milestone_tok_ids
                state["_injecting_milestone"] = True
                logits[idx, :] = float("-inf")
                milestone_tok_id = state["_milestone_ids"].pop(0)
                logits[idx, milestone_tok_id] = 1.0
                state["injected_positions"].append(len(state["output_tok_ids"]))
                logger.info("injected milestone %d, remaining=%d", pct_milestone, len(state['_milestone_ids']))

        return logits

    def _maybe_end_thinking(self, idx: int, logits: torch.Tensor, state: Dict[Any, Any]):
        if state["end_of_end"]:
            return logits

        for eti in self.end_think_ids:
            check_if_think_ended_naturally = list(state["output_tok_ids"][-len(eti):])
            if check_if_think_ended_naturally == eti:
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
                    if "available_milestones" not in state:
                        state["available_milestones"] = copy.deepcopy(self.available_milestones)
                    state["start_of_end"] = False
                    state["end_of_end"] = False
                    logger.info("model starting thinking via output tokens...")
            if state.get("is_thinking", False):
                logits = self._maybe_inject_milestone(idx, logits, state)
            if state.get("is_thinking", False):
                logits = self._maybe_end_thinking(idx, logits, state)
            if self.enable_injection_position_metadata:
                logits = self._maybe_suppress_eod(idx, logits, state)
        return logits

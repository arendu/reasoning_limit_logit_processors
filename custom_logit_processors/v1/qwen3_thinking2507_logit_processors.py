from .base_logit_processor import ThinkingBudgetLogitsProcessorBase


class ThinkingBudgetLogitsProcessor(ThinkingBudgetLogitsProcessorBase):
    DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
    DEFAULT_PROMPT_THINK_IDS = "<think>\n"
    DEFAULT_END_THINK_IDS = ["</think>"]

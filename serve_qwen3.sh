pip uninstall custom_logit_processors
pip install -e . 
export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": " Reached thinking limit. </think>", "end_think_ids": ["</think>"], "prompt_think_ids": "<think>", "model": "Qwen/Qwen3-8B"}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model Qwen/Qwen3-8B \
	--logits-processors "custom_logit_processors.v1.qwen3_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

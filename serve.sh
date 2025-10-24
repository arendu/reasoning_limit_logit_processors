export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": [2259,74045,1062], "end_think_ids": [[1885, 74045, 3318],[1885, 74045, 1062],[1885, 74045, 1561],[2259,74045,1062]], "prompt_think_ids": [49250, 2077, 1561]}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model nvidia/NVIDIA-Nemotron-Nano-12B-v2 \
	--logits-processors "custom_logit_processors.v1.nano_v2_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

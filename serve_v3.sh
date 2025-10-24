export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": [1871, 5565, 11483, 6139, 1046, 1032, 13], "end_think_ids": [[13]], "prompt_think_ids": [12, 1010]}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024 \
	--logits-processors "custom_logit_processors.v1.nano_v3_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

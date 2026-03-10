export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": " Reached thinking limit. </think>", "end_think_ids": ["</think>"], "prompt_think_ids": "<think>\n", "model": "nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024"}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024 \
	--logits-processors "custom_logit_processors.v1.nemotron_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": " Reached thinking limit. </think>", "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "enable_milestones": false, "milestone_frequency": 10, "enable_injection_position_metadata": false}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024 \
	--logits-processors "custom_logit_processors.v1.nemotron_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

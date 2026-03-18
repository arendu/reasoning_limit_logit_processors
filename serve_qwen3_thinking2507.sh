pip uninstall custom_logit_processors
pip install -e . 
export THINKING_BUDGET_LOGITS_PROCESSOR_ARGS='{"thinking_budget": 150, "thinking_budget_grace_period": 30, "end_token_ids": " Reached thinking limit. </think>", "prompt_think_ids": "<think>\n",  "model": "Qwen/Qwen3-4B-Thinking-2507", "enable_milestones": true, "milestone_frequency": 10, "enable_injection_position_metadata": true}'
python3 -m vllm.entrypoints.openai.api_server \
	--served-model-name "model"  \
	--model Qwen/Qwen3-4B-Thinking-2507 \
	--logits-processors "custom_logit_processors.v1.qwen3_thinking2507_logit_processors:ThinkingBudgetLogitsProcessor" \
	--port 8881 \
	--trust-remote-code 

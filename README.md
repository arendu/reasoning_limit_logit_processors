# Custom Logit Processors

vLLM v1 logit processors for thinking-budget control, milestone injection, and injection position metadata.

Supports Qwen3, Nemotron Nano V3, and Nemotron Nano V2.

## Setup

Start an interactive session using `run.sh`, then install:

```bash
pip install -e .
```

## Serving a Model

Each model has a serve script that sets server-level config via `THINKING_BUDGET_LOGITS_PROCESSOR_ARGS` and starts a vLLM OpenAI-compatible server on port 8881.

```bash
# Qwen3-8B
bash serve_qwen3.sh

# Nemotron Nano V3
bash serve_v3.sh

# Nemotron Nano V2
bash serve_v2.sh
```

### Server-Level Config

All server-level options are set in the `THINKING_BUDGET_LOGITS_PROCESSOR_ARGS` env var (JSON string) or via `THINKING_BUDGET_LOGITS_PROCESSOR_ARGS_FILE` (path to a JSON file). The file takes precedence if both are set.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | varies by subclass | HuggingFace model name for tokenizer |
| `thinking_budget` | int | -1 (disabled) | Max thinking tokens before forcing end |
| `thinking_budget_grace_period` | int | -1 | Hard cutoff after budget + grace tokens |
| `end_token_ids` | string or list[int] | `[]` | Token sequence injected when budget forces end-of-thinking |
| `prompt_think_ids` | string | `"<think>"` (Qwen3) / `"<think>\n"` (Nemotron) | Token sequence at end of prompt that signals thinking mode |
| `end_think_ids` | list[string] | `["</think>"]` | Token sequences for natural end-of-thinking detection |
| `enable_milestones` | bool | false | Inject progress markers (e.g. `[~30% tokens consumed]`) during thinking |
| `milestone_frequency` | int | 10 | Milestone interval as percentage (10 = every 10%) |
| `enable_injection_position_metadata` | bool | false | Append `[Injected tokens:[...]]` metadata before EOD |

Example:

```json
{
  "thinking_budget": 150,
  "thinking_budget_grace_period": 30,
  "end_token_ids": " Reached thinking limit. </think>",
  "model": "Qwen/Qwen3-8B",
  "enable_milestones": true,
  "milestone_frequency": 10,
  "enable_injection_position_metadata": true
}
```

### Logit Processor Classes

| Model | `--logits-processors` value |
|---|---|
| Qwen3 | `custom_logit_processors.v1.qwen3_logit_processors:ThinkingBudgetLogitsProcessor` |
| Nemotron V3 / V2 | `custom_logit_processors.v1.nemotron_logit_processors:ThinkingBudgetLogitsProcessor` |

## Client Usage

With a server running, use `client.py` to send requests:

```bash
python client.py
```

### Per-Request Overrides

The client can override these fields per-request via `extra_body.vllm_xargs`:

| Field | Description |
|---|---|
| `thinking_budget` | Override the thinking token budget |
| `thinking_budget_grace_period` | Override the grace period |
| `end_token_ids` | Override the end-of-thinking token sequence |

Example:

```python
result = client.chat.completions.create(
    model="model",
    messages=[...],
    temperature=1.0,
    max_tokens=5220,
    logprobs=True,
    extra_body={
        "vllm_xargs": {
            "thinking_budget": 700,
            "thinking_budget_grace_period": 130,
            "end_token_ids": "\n\n I have been forced to end thinking.</think>",
        }
    }
)
```

### Parsing Injection Metadata

When `enable_injection_position_metadata` is enabled on the server, the response will end with `[Injected tokens:[pos1, pos2, ...]]` before the EOD token. The client parses this to list which token positions were forced:

```python
import re, json

tokens = [t.token for t in result.choices[0].logprobs.content]
full_text = "".join(tokens)

match = re.search(r'\[Injected tokens:(\[.*?\])\]', full_text)
if match:
    positions = json.loads(match.group(1))
    for pos in positions:
        print(f"  pos {pos} -> {tokens[pos]!r}")
```

## Architecture

```
base_logit_processor.py          # ThinkingBudgetLogitsProcessorBase (all shared logic)
├── qwen3_logit_processors.py    # DEFAULT_MODEL = "Qwen/Qwen3-8B"
│                                # DEFAULT_PROMPT_THINK_IDS = "<think>"
└── nemotron_logit_processors.py # DEFAULT_MODEL = "nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024"
                                 # DEFAULT_PROMPT_THINK_IDS = "<think>\n"
```

Subclasses only set class-level defaults. All logic lives in the base class.

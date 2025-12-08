# vllm_logit_processors for Nano V3 (and V2)


## Getting started

First start an interactive session using `run.sh`

Next, install `custom_logit_processors` via pip `pip install -e .` This will install the runtime budget control logit processor for V3.

Start a server for a nano v3 checkpoint using `serve_v3.sh`. Once the server is running, you can ping it via the example client script: `client.py`.

Note: `serve_v3.sh` has some default runtime budget params. The `client.py` can override it for each call.
python -m sglang.launch_server \
    --model-path <local-path> \
    --port 39999 \
    --host 0.0.0.0 \
    --is-embedding \
    --tp-size=1 \
    --trust-remote-code
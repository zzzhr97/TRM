vllm serve <local-path> \
    --served-model-name "TIGER-Lab/general-verifier" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --port 23888 \
    --host 0.0.0.0 \
    --chat-template chat_templates/verifier.jinja
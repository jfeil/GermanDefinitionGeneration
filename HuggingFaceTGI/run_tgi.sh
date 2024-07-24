model=meta-llama/Meta-Llama-3-8B-Instruct
# model=meta-llama/Meta-Llama-3-8B
# model=meta-llama/Meta-Llama-3-70B-Instruct
token=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

# model=lucyknada/microsoft_WizardLM-2-7B
# model=mistralai/Mixtral-8x7B-Instruct-v0.1
model=meta-llama/Meta-Llama-3.1-8B-Instruct

docker run -i --name hugging-tgf --rm --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$token -v ./data:/data ghcr.io/huggingface/text-generation-inference --model-id $model --quantize bitsandbytes-nf4 --max-concurrent-requests 512

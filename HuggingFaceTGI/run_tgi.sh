model=meta-llama/Meta-Llama-3-8B-Instruct
model=meta-llama/Meta-Llama-3-8B
model=meta-llama/Meta-Llama-3-70B-Instruct
token=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
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

model=lucyknada/microsoft_WizardLM-2-7B

docker run -i --name hugging-tgf --rm --gpus all --shm-size 1g -p 8080:80 -e HUGGING_FACE_HUB_TOKEN=$token -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
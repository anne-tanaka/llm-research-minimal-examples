# Crystal Evaluation Reasoning: Minimal Example

This code is an independent implementation inspired by the methodology of the Crystal project, adapted for our evaluation task. The core idea follows a two-stage process (SFT followed by RL) to optimize the reasoning process of a language model.

-   **Original Paper**: [https://aclanthology.org/2023.emnlp-main.708.pdf]
-   **Original Repository**: https://github.com/liujch1998/crystal

While the approach is inspired by Crystal, the implementation (Llama 3.1, QLoRA, DeepSpeed) and code are original.

## Environment Setup

The test scripts in this directory are designed to be run within a specific Docker container. Please use the provided `Dockerfile` and `environment.yml` to build the environment.

### 1. Build the Docker Image

From this directory (`crystal-evaluation-reasoning`), run the following command to build the image. You can change the image name and tag (`-t` option) as you see fit.

```bash
docker build --network=host -t crystal-env:latest .
```

### 2. Run the Docker Container

After the image is built, use the following command to start a container. This command mounts the necessary directories for code, data, checkpoints, and logs.

Note: Please modify the host-side paths (the part before the colon :) in the -v options to match your own directory structure.

```bash
docker run --network host \
  --gpus all \
  -it --name [my-container-name] \
  -v /path/to/your/project_code:/workspace \
  -v /path/to/your/datasets:/datasets \
  -v /path/to/your/checkpoints:/checkpoints \
  -v /path/to/your/logs:/logs \
  -v ~/.ssh:/home/[USERNAME]/.ssh:ro \
  -v ~/.gitconfig:/home/[USERNAME]/.gitconfig \
  # other options
  --shm-size=164g \
  crystal-env:latest
```

## Contents
- [QLoRA - LLaMA fine-tuning](qlora-llama/)
- [QLoRA - Mistral fine-tuning](qlora-mistral/)
- [PPO with LLaMA](ppo/)
- [GRPO experiments](grpo/)


## QLoRA - LLaMA fine-tuning
### components
- Datasets loading
- Tokenization for llama
- Trainer with original loss function using PEFT QLoRA and DeepSpeed with ZeRO3
- Accuracy function
- Validation
- Saving checkpoints

# Crystal Evaluation Reasoning: Minimal Example

This code is an independent implementation inspired by the methodology of the Crystal project, adapted for our evaluation task. The core idea follows a two-stage process (SFT followed by RL) to optimize the reasoning process of a language model.

-   **Original Paper**: [https://aclanthology.org/2023.emnlp-main.708.pdf]
-   **Original Repository**: https://github.com/liujch1998/crystal

While the approach is inspired by Crystal, the implementation (Llama 3.1, QLoRA, DeepSpeed) and code are original.

## How to Run

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

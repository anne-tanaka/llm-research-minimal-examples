import os
import socket
import subprocess
import sys

def run_training_script():
    # Set environment variables
    os.environ["NCCL_IB_TIMEOUT"] = "22"

    # Define the available IP and port for inter-process communication
    main_ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Main IP Address: {main_ip_address}")
    available_port = 20001  # Replace with an available port as needed 20001, 19999

    # Llama QLoRA使用バージョン
    command = [
        "accelerate", "launch",
        "--config_file", "/workspace/configs/deepspeed_config_z3_qlora.yaml",  # configファイルの設定を追加
        "--num_machines", "1",               # Single-node
        "--machine_rank", "0",               # Single process
        "--num_processes", "4",              # multi GPU process
        "--main_process_ip", main_ip_address,
        "--main_process_port", str(available_port),
        "--mixed_precision", "bf16",  # "bf16" から変更
        "/workspace/test_codes/qlora-llama/test_trainer_loss.py",
        "--model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct", 
        # "--packing",
        # "--lr_scheduler_type", "cosine",
        # "--weight_decay", "1e-4",
        "--warmup_ratio", "0.0", 
        "--max_grad_norm", "1.0",
        # "--per_device_train_batch_size",  "2",
        # "--per_device_eval_batch_size", "2",
         "--total_steps", "60",  # reduced for test
        "--batch_size", "2",
        "--gradient_accumulation_steps", "2",  # 2
        "--gradient_checkpointing",
        # "--use_reentrant",  # https://github.com/huggingface/trl/issues/835 を参考に、Falseに設定するためコメントアウト
        "--use_flash_attn",
        "--use_peft_lora",
        "--lora_r", "8", 
        "--lora_alpha", "16",
        "--lora_dropout", "0.1",
        "--lora_target_modules", "all-linear",
        "--use_4bit_quantization",
        "--use_nested_quant",
        "--bnb_4bit_compute_dtype", "bfloat16",
        "--bnb_4bit_quant_storage_dtype", "bfloat16",
        "--max_answer_len", "40", 
        "--log_interval", "1",  # added for test
        "--eval_interval", "0", "--save_interval", "0",  # max_anwer_lenを2から変更
        "--model_type", "Llama-3.1-8B-Instruct", 
        # "--engine", "davinci",  
        "--qk_loss_multiplier", "1.0", "--qa_loss_multiplier", "0.0",
        "--qka_loss_multiplier", "1.0", "--qk_steps", "10", "--qk_and_qka_steps", "20",
        "--run_name", "test_trainer_loss_markov_llama3.1-8b-instruct_20250915"
    ]

    log_file = "/logs/test_trainer_loss_llama3.1-8b-instruct_20250915.log"

    # Prepare nohup command with log file
    nohup_command = ["nohup"] + command

    # Run the training script and redirect output to the log file
    with open(log_file, "w") as log:
        result = subprocess.run(nohup_command, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp  # Detach from terminal
        )
        print(f"Training script executed. Logs are saved in {log_file}")
    
if __name__ == "__main__":
    print(f"Python Executable: {sys.executable}")  # Debug: Confirm the active Python environment
    run_training_script()
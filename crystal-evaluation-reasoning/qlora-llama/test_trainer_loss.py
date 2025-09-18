import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import deepspeed
import transformers
import accelerate
from accelerate.logging import get_logger
import math
import contextlib
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

import torch.nn as nn

import os  # to get hugging face hub token

from utils.utils import set_seed

# addition part for QLoRA
from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = get_logger(__name__)

# class added for QLoRA
# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )

def collate_fn(batch):
    global tokenizer
    global MAX_SEQ_LEN
    # batchに含まれる複数のデータ項目から task_ix の値だけを抽出し、それらをまとめてテンソルとして格納することで、PyTorchのモデルに入力する準備をする
    task_ixs = torch.tensor([item['task_ix'] for item in batch], dtype=torch.long)

    # for qk_loss
    input_ids_list_qk = []
    attention_mask_list_qk = []
    labels_start_indices_qk = [] # indices which starts knowledge parts
    # for qa_loss
    input_ids_list_qa = []
    attention_mask_list_qa = []
    labels_start_indices_qa = [] # indices which starts answer parts after question parts
    # for qka_loss
    input_ids_list_qka = []
    attention_mask_list_qka = []
    labels_start_indices_qka = [] # indices which starts answer parts after knowledge parts

    for item in batch:
        question = item['question']
        knowledge = item['knowledge']
        answer = item['answer']

        # ==== 1. input construction and indices identification for QK Loss ====
        # Combine the user's questions and the desired knowledge output into a single list.
        messages_full_qk = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": knowledge} # add knowledge here
        ]
        # use apply_chat_template to full_text
        full_text_qk = tokenizer.apply_chat_template(
            messages_full_qk,
            tokenize=False,
            add_generation_prompt=False  # template can automatically process the role of assistant
        )
        # tokenize full_text_qk using tokenizer(), and gain offsets_mapping
        encoded_full_qk = tokenizer(
            full_text_qk,
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors='pt',
            add_special_tokens=True, # maybe can be False depending on model
            return_offsets_mapping=True
        )

        input_ids_list_qk.append(encoded_full_qk.input_ids.squeeze(0))
        attention_mask_list_qk.append(encoded_full_qk.attention_mask.squeeze(0))
        offsets_qk = encoded_full_qk.offset_mapping.squeeze(0)

        # idetify knowledge start indices from offsets_mapping
        # search knowledge start indices in the template-applied text
        knowledge_start_char_idx = full_text_qk.find(knowledge)
        # error handling when we cannot find knowledge
        if knowledge_start_char_idx == -1:
            raise ValueError("Knowledge not found in the formatted text.")

        # search knowledge start token index using offsets_mapping
        knowledge_start_token_idx = -1
        for i, (start, end) in enumerate(offsets_qk):
            if start >= knowledge_start_char_idx:
                knowledge_start_token_idx = i
                break

        labels_start_indices_qk.append(knowledge_start_token_idx)

        # ==== 2. input construction and indices identification for QA Loss ====
        # Combine the user's questions and the desired answer output into a single list.
        messages_full_qa = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer} 
        ]
        # use apply_chat_template to full_text
        full_text_qa = tokenizer.apply_chat_template(
            messages_full_qa,
            tokenize=False,
            add_generation_prompt=False
        )
        # tokenize full_text_qa using tokenizer(), and gain offsets_mapping
        encoded_full_qa = tokenizer(
            full_text_qa,
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors='pt',
            add_special_tokens=True, 
            return_offsets_mapping=True
        )

        input_ids_list_qa.append(encoded_full_qa.input_ids.squeeze(0))
        attention_mask_list_qa.append(encoded_full_qa.attention_mask.squeeze(0))
        offsets_qa = encoded_full_qa.offset_mapping.squeeze(0)

        # idetify answer start indices from offsets_mapping
        # search answer start indices in the template-applied text
        answer_start_char_idx = full_text_qa.find(answer)
        # error handling when we cannot find answer
        if answer_start_char_idx == -1:
            raise ValueError("Answer not found in the formatted text.")

        # search answer start token index using offsets_mapping
        answer_start_token_idx = -1
        for i, (start, end) in enumerate(offsets_qa):
            if start >= answer_start_char_idx:
                answer_start_token_idx = i
                break

        labels_start_indices_qa.append(answer_start_token_idx)

        # ==== 3. input construction and indices identification for QKA Loss ====
        # Combine the user's questions and the desired answer output into a single list.
        messages_full_qka = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": knowledge}, # assume knowledge was already generated by assistant
            {"role": "assistant", "content": answer} # add answer here
        ]

        # use apply_chat_template to full_text
        full_text_qka = tokenizer.apply_chat_template(
            messages_full_qka,
            tokenize=False,
            add_generation_prompt=False
        )
        # tokenize full_text_qa using tokenizer(), and gain offsets_mapping
        encoded_full_qka = tokenizer(
            full_text_qka,
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors='pt',
            add_special_tokens=True,
            return_offsets_mapping=True
        )

        input_ids_list_qka.append(encoded_full_qka.input_ids.squeeze(0))
        attention_mask_list_qka.append(encoded_full_qka.attention_mask.squeeze(0))
        offsets_qka = encoded_full_qka.offset_mapping.squeeze(0)

        # idetify answer start indices from offsets_mapping
        # search answer start indices in the template-applied text
        answer_start_char_idx = full_text_qka.find(answer)
        # error handling when we cannot find answer
        if answer_start_char_idx == -1:
            raise ValueError("Answer not found in the formatted text.")

        # search answer start token index using offsets_mapping
        answer_start_token_idx = -1
        for i, (start, end) in enumerate(offsets_qka):
            if start >= answer_start_char_idx:
                answer_start_token_idx = i
                break

        labels_start_indices_qka.append(answer_start_token_idx)

        # ==== 4. padding for the whole batch ====
        # padding fitted for the max sequence of batch at the end of collate_fn
        # it is conveient to use tokenizer.pad() 
        # for qk_loss
        padded_batch_qk = tokenizer.pad(
            {'input_ids': input_ids_list_qk, 'attention_mask': attention_mask_list_qk},
            padding='longest', # pading fitted for the max sequence of batch
            return_tensors='pt',
            pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
        )

        # for qa_loss
        padded_batch_qa = tokenizer.pad(
            {'input_ids': input_ids_list_qa, 'attention_mask': attention_mask_list_qa},
            padding='longest', # pading fitted for the max sequence of batch
            return_tensors='pt',
            pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
        )

        # for qka_loss
        padded_batch_qka = tokenizer.pad(
            {'input_ids': input_ids_list_qka, 'attention_mask': attention_mask_list_qka},
            padding='longest', # pading fitted for the max sequence of batch
            return_tensors='pt',
            pad_to_multiple_of=8 # Consider padding to multiples of 8 for training efficiency
        )

    return {
        'task_ixs': task_ixs,
        'input_ids_qk': padded_batch_qk.input_ids,
        'attention_mask_qk': padded_batch_qk.attention_mask,
        'labels_start_indices_qk': torch.tensor(labels_start_indices_qk, dtype=torch.long),

        'input_ids_qa': padded_batch_qa.input_ids,
        'attention_mask_qa': padded_batch_qa.attention_mask,
        'labels_start_indices_qa': torch.tensor(labels_start_indices_qa, dtype=torch.long),

        'input_ids_qka': padded_batch_qka.input_ids,
        'attention_mask_qka': padded_batch_qka.attention_mask,
        'labels_start_indices_qka': torch.tensor(labels_start_indices_qka, dtype=torch.long),

        'raw_questions': [item['question'] for item in batch],
        'raw_knowledges': [item['knowledge'] for item in batch],
        'answers': [item['answer'] for item in batch],
    }
        

class Trainer:
    def __init__(self,
                 args,
                 accelerator,
                 train_dataloader,
                 eval_dataloader,
                 tokenizer,
                 model,
                 optimizer,
                 init_step,
                 eval_losses,
                 eval_accs,
                ):
        self.args = args
        self.accelerator = accelerator
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_sampler = iter(self.train_dataloader)  # https://docs.python.org/3/library/functions.html#iter 参照
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        if not self.args.nolog and accelerator.is_main_process:
            wandb.init(project='rainier_stageI', name=args.run_name, config=args)
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')  # https://docs.wandb.ai/guides/track/log/customize-logging-axes/ 参照
            wandb.define_metric('eval/*', step_metric='eval/step')
        
        num_samples_in_epoch = len(self.train_dataloader.dataset)
        num_batches_in_epoch = math.ceil(num_samples_in_epoch / self.args.batch_size)  # 1エポック(サンプルを全て見ること)に必要なバッチ処理の回数

        # 学習を中断した箇所から正確に再開するために、データセットを早送り（スキップ）するための処理
        # for _ in range((init_step * args.gradient_accumulation_steps) % len(self.train_dataloader)):
        ## 上記方法だとlen(self.train_dataloader)がaccelerator.prepareの後に0になってしまいZeroDivisionErrorが発生するので以下に変更
        ### https://discuss.pytorch.org/t/what-is-len-dataloader-equal-to/52472/4 参照して実装
        ### init_step: optimizer.step()した回数, args.gradient_accumulation_steps: 勾配累積ステップ数, 
        ### init_step*args.gradient_accumulation_steps: 中断するまでに処理したミニバッチの総数
        for _ in range((init_step * args.gradient_accumulation_steps) % num_batches_in_epoch):
            next(self.train_sampler)

        self.eval_losses = eval_losses
        self.eval_accs = eval_accs
        print("Trainer init finished!")

    # Trainerクラスの中に追加
    def _calculate_loss_and_tokens(self, batch, loss_type: str):
        """
        指定された損失タイプに基づいて、損失と非パディングトークン数を計算する。
        """
        input_ids = batch[f'input_ids_{loss_type}']
        attention_mask = batch[f'attention_mask_{loss_type}']
        labels_start_indices = batch[f'labels_start_indices_{loss_type}']

        # labelsテンソルを作成（この処理は一度だけ実行される）
        labels = input_ids.clone()  # input_idsの形状と同じラベルテンソルを初期化
        for i in range(labels.size(0)):  # バッチ内の各サンプルについてループ
            start_idx = labels_start_indices[i].item()  # 開始インデックス
            # 開始インデックスより前を全て-100にマスク
            labels[i, :start_idx] = -100
            # パディングトークンも除外
            labels[i, labels[i] == self.tokenizer.pad_token_id] = -100
        
        # 非パディングトークン数を計算
        num_tokens = (labels != -100).sum()

        # モデルに渡して損失を計算
        loss = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            labels=labels.to(self.model.device),
        ).loss

        return loss, num_tokens

    # Trainerクラスの中
    def qk_loss(self, batch):
        loss, _ = self._calculate_loss_and_tokens(batch, 'qk')
        return loss

    def qa_loss(self, batch):
        loss, _ = self._calculate_loss_and_tokens(batch, 'qa')
        return loss

    def qka_loss(self, batch):
        loss, _ = self._calculate_loss_and_tokens(batch, 'qka')
        return loss
    
    def train(self, step):
        # 訓練終了後に評価モードに戻す
        # --- テストコードなので以下は割愛する ---
        # self.model.eval()
        # self.save(step=step)  # 現在のトレーニングのステップ step におけるモデルの状態を保存
        # self.valid(step=step)  # 現在のステップでモデルの検証（validation）を実行
        # --- テストコード割愛部分終了 ---

        self.accelerator.wait_for_everyone()  # 分散トレーニングの場合、すべてのプロセスが同期するのを待つ
        self.model.train()  
        losses, qk_losses, qa_losses, qka_losses = [], [], [], []  # トレーニング中に計算された損失を記録しておくためのリスト

        batch_samples = []
        for _ in range(self.args.gradient_accumulation_steps):
            try:
                batch_samples.append(next(self.train_sampler))  # https://docs.python.org/ja/3/library/stdtypes.html#iterator-types 参照
            except StopIteration:
                self.train_sampler = iter(self.train_dataloader)
                batch_samples.append(next(self.train_sampler))
        # ロギング用に raw の平均 loss を保存
        raw_qk_losses, raw_qa_losses, raw_qka_losses = [], [], []
        # T_global, S_global を初期化
        T_global, S_global = torch.tensor(0.0, device=self.accelerator.device), torch.tensor(0.0, device=self.accelerator.device)
        # backward 用の合計損失・トークン数
        total_S = torch.tensor(0.0, device=self.accelerator.device)
        total_T = torch.tensor(0.0, device=self.accelerator.device)
        
        # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#gradient-accumulation-on-training-samples-of-variable-size 参照
        for i, batch in enumerate(batch_samples):
            # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unnecessary communications when accumulating
            # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
            # if (i < len(batch_samples) - 1 and self.accelerator.num_processes > 1):
            #     ctx = self.model.no_sync
            # else:
            #     ctx = contextlib.nullcontext
            # print("in Trainer train method, before loss calculation")
            # with ctx():
            # model.no_sync のコンテキストマネージャーはZeRO-3と互換性がないため削除。
            # DeepSpeedが勾配累積を自動で処理してくれるはず。
            S_local = torch.tensor(0.0, device=self.accelerator.device)
            T_local = torch.tensor(0.0, device=self.accelerator.device)
            
            # --- QA 損失 ---
            if self.args.qa_loss_multiplier > 0.0:
                qa_loss, qa_tokens = self._calculate_loss_and_tokens(batch, 'qa')
                # detach(): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html より「勾配計算のグラフからテンソルを切り離す」
                # cpu(): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html より「テンソルをCPUのメモリに移動（または複製）」
                ## cf. clone()(先行研究でcpu()の代わりに使っていたもの): https://docs.pytorch.org/docs/stable/generated/torch.clone.html より「テンソルを同じデバイス上（この場合はGPU上）に複製」
                raw_qa_losses.append(qa_loss.detach().cpu())
                S_local += self.args.qa_loss_multiplier * (qa_loss * qa_tokens)
                T_local += qa_tokens

            # --- QK 損失 ---
            if self.args.qk_loss_multiplier > 0.0 and (
                self.args.qka_loss_multiplier > 0.0 or step % self.args.qk_and_qka_steps < self.args.qk_steps
            ):
                qk_loss, qk_tokens = self._calculate_loss_and_tokens(batch, 'qk')
                raw_qk_losses.append(qk_loss.detach().cpu())
                S_local += self.args.qk_loss_multiplier * (qk_loss * qk_tokens)
                T_local += qk_tokens

            # --- QKA 損失 ---
            if self.args.qka_loss_multiplier > 0.0 and (
                self.args.qk_loss_multiplier > 0.0 or step % self.args.qk_and_qka_steps >= self.args.qk_steps
            ):
                qka_loss, qka_tokens = self._calculate_loss_and_tokens(batch, 'qka')
                raw_qka_losses.append(qka_loss.detach().cpu())
                S_local += self.args.qka_loss_multiplier * (qka_loss * qka_tokens)
                T_local += qka_tokens

            # --- DDP: 各プロセスで合計値を集約 ---
            ## https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#gradient-accumulation-on-training-samples-of-variable-size 参照
            S_global = self.accelerator.reduce(S_local, reduction="sum")  # ミニバッチ全体での合計損失
            T_global = self.accelerator.reduce(T_local, reduction="sum")  # ミニバッチ全体での合計トークン数

            if T_global.item() > 0:
                if T_global!=0:
                    loss = S_global / (T_global)
                    self.accelerator.backward(loss)
                else:
                    loss = S_global / (T_global + 1e-12)
                    self.accelerator.backward(loss)
                
            # optimizer step は勾配累積の最後で
            if (i + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()  # 複数ミニバッチ分の蓄積された勾配を使って、モデルの重みを一度だけ更新 https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html
                self.optimizer.zero_grad()  # 前回のステップで計算された勾配をリセット https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

        # qk_loss, qa_loss, qka_loss の記録のための処理
        # GPUごとに各バッチの損失を平均して最終的な値を計算。
        # CPU上のテンソルリストを処理し、結果をGPUに送る
        qk_loss = (
            torch.stack(raw_qk_losses).mean(dim=0, keepdim=True).to(self.accelerator.device) # <- .to(...) を追加
            if raw_qk_losses
            else torch.tensor([0.0], device=self.accelerator.device) # <- device=... を追加
        )
        qa_loss = (
            torch.stack(raw_qa_losses).mean(dim=0, keepdim=True).to(self.accelerator.device)
            if raw_qa_losses
            else torch.tensor([0.0], device=self.accelerator.device)
        )
        qka_loss = (
            torch.stack(raw_qka_losses).mean(dim=0, keepdim=True).to(self.accelerator.device)
            if raw_qka_losses
            else torch.tensor([0.0], device=self.accelerator.device)
        )
        # 複数GPUの結果をまとめる
        ## accelerator.gather: https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.gather
        qk_losses = self.accelerator.gather(qk_loss) # (num_gpus)
        qa_losses = self.accelerator.gather(qa_loss) # (num_gpus)
        qka_losses = self.accelerator.gather(qka_loss) # (num_gpus)

        # 全GPUのlossを平均
        ## .item(): テンソルからPythonの数値を取り出す(https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html)
        qk_loss = qk_losses.mean().item()
        qa_loss = qa_losses.mean().item()
        qka_loss = qka_losses.mean().item()

        if not self.args.nolog and self.accelerator.is_main_process:
            if step % self.args.log_interval == 0:  # 一定ステップごとに、WandB にトレーニングの損失値を記録
                wandb.log({
                    'train/step': step,
                    'train/qk_loss': qk_loss if raw_qk_losses else 0.0,
                    'train/qa_loss': qa_loss if raw_qa_losses else 0.0,
                    'train/qka_loss': qka_loss if raw_qka_losses else 0.0,
                    # global 損失は最後に出す
                    "train/loss": loss.item() if T_global.item() > 0 else 0.0,
                })


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--model_type', type=str, default='Llama-3.1-8B-Instruct', choices=["Llama-3.2-3B-Instruct", "Llama-3.2-1B-Instruct", "Llama-3.1-8B-Instruct", "Llama-3.1-8B"])
    parser.add_argument('--load_from_ckpt', default=None)
    parser.add_argument('--max_question_len', type=int, default=1200)  # 256から変更→1200→1400
    parser.add_argument('--max_knowledge_len', type=int, default=240)  # 32から変更→300
    parser.add_argument('--max_answer_len', type=int, default=128)  # 128
    parser.add_argument("--max_seq_len", type=int, default=2048, # MAX_SEQ_LENを引数として受け取る
                        help="Maximum sequence length for tokenization and model input.")


    # train
    parser.add_argument('--data_path', type=str, default='../data/{datapath}/{split}.tsv')
    # parser.add_argument('--engine', type=str, default='davinci', choices=['curie', 'davinci'])  # 先行研究ではあったが不要そうなのでコメントアウト
    parser.add_argument('--train_tasks', type=str, default='summeval,topicalchat') # 先行研究ではobqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--half_half', default=False, action='store_true')
    parser.add_argument('--qa_loss_multiplier', type=float, default=1.0)
    parser.add_argument('--qka_loss_multiplier', type=float, default=0.0)
    parser.add_argument('--qk_loss_multiplier', type=float, default=1.0)
    parser.add_argument('--qk_steps', type=int, default=500)
    parser.add_argument('--qk_and_qka_steps', type=int, default=1000)

    # other
    parser.add_argument('--log_interval', type=int, default=50, help='step interval to log stats')
    parser.add_argument('--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--nosave', default=False, action='store_true')
    parser.add_argument('--nolog', default=False, action='store_true')
    parser.add_argument('--eval_loop_cap', type=int, default=None, help='cap on number of eval loops')

    # QLoRA用に追加
    # ModelArguments の引数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--chat_template_format", type=str, default="none",
                        help="chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template.")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
                        help="comma separated list of target modules to apply LoRA layers to")
    parser.add_argument("--use_nested_quant", action="store_true",
                        help="Activate nested quantization for 4bit base models")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                        help="Compute dtype for 4bit base models")
    parser.add_argument("--bnb_4bit_quant_storage_dtype", type=str, default="uint8",
                        help="Quantization storage dtype for 4bit base models")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                        help="Quantization type fp4 or nf4")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Enables Flash attention for training.")
    parser.add_argument("--use_peft_lora", action="store_true",
                        help="Enables PEFT LoRA for training.")
    parser.add_argument("--use_8bit_quantization", action="store_true",
                        help="Enables loading model in 8bit.")
    parser.add_argument("--use_4bit_quantization", action="store_true",
                        help="Enables loading model in 4bit.")
    parser.add_argument("--use_reentrant", action="store_true",
                        help="Gradient Checkpointing param. Refer the related docs")
    parser.add_argument("--use_unsloth", action="store_true",
                        help="Enables UnSloth for training.")
    
    parser.add_argument("--packing", action="store_true", help="Whether to pack sequences.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # MAX_SEQ_LEN を args から取得するように修正
    global MAX_SEQ_LEN
    MAX_SEQ_LEN = args.max_seq_len

    set_seed()

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    # Load data
    print(f'Loading data ...')
    train_sample_data = [
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Outspoken MP Bob Katter has risked being drawn into a race row after accusing the Queensland Government of believing embattled MP Billy Gordon was 'a little black puppet on a string'. In a bizarre statement on his party's website, Mr Katter said the Government believes Mr Gordon would quit after he was recently accused of domestic violence - and later admitted to his criminal past. Mr Katter, federal MP for Kennedy, is of Lebanese descent. He said he's been personally subjected to racist attitudes from both major parties in the past, and that he sympathised with Mr Gordon's position. Federal MP Bob Katter, who released a statement today accusing the Queensland Government of assuming Billy Gordon would quit after he was recently accused of domestic violence Mr Gordon was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history from Premier Annastacia Palaszczuk . He announced on Wednesday he will remain in the Queensland parliament as an independent, throwing the Government, which holds a razor-thin majority, into chaos. 'In the last two cases where this occurred in the federal parliament, there was never any question raised about the member resigning. So why was it assumed here that Billy Gordon would resign?' Mr Katter's statement reads. 'I can't help but think that the Queensland Government thought they had a little black puppet on a string. Mr Gordon, who \u00a0was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history The statement has sparked criticism on social media, with users questioning his terminology. 'Irrespective of his weird use of the term 'black puppet' it's kinda odd how Bob Katter is speaking on Billy Gordon's behalf' wrote Benjamin James. Not that John wrote: 'Wow, Bob Katter calls Billy Gordon, 'black puppet on a string' and the lefty loons stay silent. What if Abbott had said it?' The statement has sparked criticism on social media, with some users questioning his terminology Mr Katter has since defended the statement, saying he used the term black puppet in light of his personal experiences Mr Katter has since defended the statement, saying he used the term black puppet 'because I'm dark and I come from Cloncurry.' 'I can understand and I can identify with the attitudes that these people have and I'm going to call it for what it is,' he said. 'I deliberately put that comment in with a forethought, because I think they deserve a belting for the racial overtones.' Mr Katter used federal MPs Tony Windsor and Rob Oakeshott as examples of independents becoming puppets for the government and sacrificing their independence, but he vowed Mr Gordon would not make that mistake. output: bob katter has risked being drawn into a race row after accusing the queensland government of believing embattled mp billy gordon was ' a little black puppet on a string ' . mr katter said he 's been personally subjected to racist attitudes from both major parties in the past , and that he sympathised with mr gordon 's position . mr gordon was forced to quit after he was recently accused of domestic violence .",
            'answer': "{\"coherence\": 5, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary presents a clear narrative about Bob Katter's controversial statement and its implications. However, it could improve by providing more context on the situation and the reactions it sparked, which would create a more complete picture.\n\n- **Consistency (5)**: The summary accurately reflects the main points from the source text, including Katter's accusation and Gordon's situation, without introducing any factual errors or inconsistencies.\n\n- **Fluency (5)**: The text is grammatically correct and easy to read, with well-structured sentences that facilitate understanding.\n\n- **Relevance (4)**: While the summary captures the essential elements of the controversy involving Katter and Gordon, it omits some details, such as the social media reactions and Katter's defense of his statement, which could provide a fuller understanding of the situation.\n\nAspects like creativity, empathy, and humor are not applicable here, as the task focuses on factual summarization rather than emotional or creative"
        },
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Outspoken MP Bob Katter has risked being drawn into a race row after accusing the Queensland Government of believing embattled MP Billy Gordon was 'a little black puppet on a string'. In a bizarre statement on his party's website, Mr Katter said the Government believes Mr Gordon would quit after he was recently accused of domestic violence - and later admitted to his criminal past. Mr Katter, federal MP for Kennedy, is of Lebanese descent. He said he's been personally subjected to racist attitudes from both major parties in the past, and that he sympathised with Mr Gordon's position. Federal MP Bob Katter, who released a statement today accusing the Queensland Government of assuming Billy Gordon would quit after he was recently accused of domestic violence Mr Gordon was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history from Premier Annastacia Palaszczuk . He announced on Wednesday he will remain in the Queensland parliament as an independent, throwing the Government, which holds a razor-thin majority, into chaos. 'In the last two cases where this occurred in the federal parliament, there was never any question raised about the member resigning. So why was it assumed here that Billy Gordon would resign?' Mr Katter's statement reads. 'I can't help but think that the Queensland Government thought they had a little black puppet on a string. Mr Gordon, who \u00a0was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history The statement has sparked criticism on social media, with users questioning his terminology. 'Irrespective of his weird use of the term 'black puppet' it's kinda odd how Bob Katter is speaking on Billy Gordon's behalf' wrote Benjamin James. Not that John wrote: 'Wow, Bob Katter calls Billy Gordon, 'black puppet on a string' and the lefty loons stay silent. What if Abbott had said it?' The statement has sparked criticism on social media, with some users questioning his terminology Mr Katter has since defended the statement, saying he used the term black puppet in light of his personal experiences Mr Katter has since defended the statement, saying he used the term black puppet 'because I'm dark and I come from Cloncurry.' 'I can understand and I can identify with the attitudes that these people have and I'm going to call it for what it is,' he said. 'I deliberately put that comment in with a forethought, because I think they deserve a belting for the racial overtones.' Mr Katter used federal MPs Tony Windsor and Rob Oakeshott as examples of independents becoming puppets for the government and sacrificing their independence, but he vowed Mr Gordon would not make that mistake. output: bob katter has risked being drawn into a race row after accusing the queensland government of believing embattled mp billy gordon was ' a little black puppet on a string ' . mr katter said he 's been personally subjected to racist attitudes from both major parties in the past , and that he sympathised with mr gordon 's position . mr gordon was forced to quit after he was recently accused of domestic violence .",
            'answer': "{\"coherence\": 5, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary presents a clear sequence of events, focusing on Bob Katter's controversial statement and its implications. However, it could benefit from smoother transitions to enhance the narrative flow.\n\n- **Consistency (5)**: The summary accurately reflects the key points from the source text, including Katter's accusation and the context surrounding Billy Gordon, ensuring factual consistency.\n\n- **Fluency (5)**: The text is grammatically correct and easy to read, with well-structured sentences that facilitate understanding.\n\n- **Relevance (4)**: The summary captures the main issues involving Katter's statement and Gordon's situation but omits some contextual details, such as the public reaction and Katter's defense, which could provide a fuller picture.\n\nAspects like creativity, empathy, and humor are not applicable here, as the task focuses on summarizing factual content rather than emotional or creative expression. Depth and inclusivity were not selected, as they are more relevant to analytical"
        },
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Outspoken MP Bob Katter has risked being drawn into a race row after accusing the Queensland Government of believing embattled MP Billy Gordon was 'a little black puppet on a string'. In a bizarre statement on his party's website, Mr Katter said the Government believes Mr Gordon would quit after he was recently accused of domestic violence - and later admitted to his criminal past. Mr Katter, federal MP for Kennedy, is of Lebanese descent. He said he's been personally subjected to racist attitudes from both major parties in the past, and that he sympathised with Mr Gordon's position. Federal MP Bob Katter, who released a statement today accusing the Queensland Government of assuming Billy Gordon would quit after he was recently accused of domestic violence Mr Gordon was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history from Premier Annastacia Palaszczuk . He announced on Wednesday he will remain in the Queensland parliament as an independent, throwing the Government, which holds a razor-thin majority, into chaos. 'In the last two cases where this occurred in the federal parliament, there was never any question raised about the member resigning. So why was it assumed here that Billy Gordon would resign?' Mr Katter's statement reads. 'I can't help but think that the Queensland Government thought they had a little black puppet on a string. Mr Gordon, who \u00a0was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history The statement has sparked criticism on social media, with users questioning his terminology. 'Irrespective of his weird use of the term 'black puppet' it's kinda odd how Bob Katter is speaking on Billy Gordon's behalf' wrote Benjamin James. Not that John wrote: 'Wow, Bob Katter calls Billy Gordon, 'black puppet on a string' and the lefty loons stay silent. What if Abbott had said it?' The statement has sparked criticism on social media, with some users questioning his terminology Mr Katter has since defended the statement, saying he used the term black puppet in light of his personal experiences Mr Katter has since defended the statement, saying he used the term black puppet 'because I'm dark and I come from Cloncurry.' 'I can understand and I can identify with the attitudes that these people have and I'm going to call it for what it is,' he said. 'I deliberately put that comment in with a forethought, because I think they deserve a belting for the racial overtones.' Mr Katter used federal MPs Tony Windsor and Rob Oakeshott as examples of independents becoming puppets for the government and sacrificing their independence, but he vowed Mr Gordon would not make that mistake. output: bob katter has risked being drawn into a race row after accusing the queensland government of believing embattled mp billy gordon was ' a little black puppet on a string ' . mr katter said he 's been personally subjected to racist attitudes from both major parties in the past , and that he sympathised with mr gordon 's position . mr gordon was forced to quit after he was recently accused of domestic violence .",
            'answer': "{\"coherence\": 5, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary effectively captures the main points of the source text, focusing on Bob Katter's controversial statement and its implications. However, it could improve by including more context about the situation's progression, such as the political ramifications of Mr. Gordon's resignation.\n\n- **Consistency (5)**: The summary accurately reflects the key facts from the source text, maintaining alignment with the original information without introducing discrepancies or errors.\n\n- **Fluency (5)**: The text is grammatically correct and well-structured, allowing for smooth reading and comprehension without any awkward phrasing or errors.\n\n- **Relevance (4)**: The summary includes the crucial information about Katter's statement and Gordon's situation but omits some details, such as the public reaction and Katter's defense of his statement, which could provide a more comprehensive understanding.\n\nAspects like creativity, empathy, and humor were not selected as they are not pertinent to the task of factual summarization."
        },
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Outspoken MP Bob Katter has risked being drawn into a race row after accusing the Queensland Government of believing embattled MP Billy Gordon was 'a little black puppet on a string'. In a bizarre statement on his party's website, Mr Katter said the Government believes Mr Gordon would quit after he was recently accused of domestic violence - and later admitted to his criminal past. Mr Katter, federal MP for Kennedy, is of Lebanese descent. He said he's been personally subjected to racist attitudes from both major parties in the past, and that he sympathised with Mr Gordon's position. Federal MP Bob Katter, who released a statement today accusing the Queensland Government of assuming Billy Gordon would quit after he was recently accused of domestic violence Mr Gordon was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history from Premier Annastacia Palaszczuk . He announced on Wednesday he will remain in the Queensland parliament as an independent, throwing the Government, which holds a razor-thin majority, into chaos. 'In the last two cases where this occurred in the federal parliament, there was never any question raised about the member resigning. So why was it assumed here that Billy Gordon would resign?' Mr Katter's statement reads. 'I can't help but think that the Queensland Government thought they had a little black puppet on a string. Mr Gordon, who \u00a0was forced to quit the Queensland Labor Party last week in the wake of being accused of domestic violence and withholding his criminal history The statement has sparked criticism on social media, with users questioning his terminology. 'Irrespective of his weird use of the term 'black puppet' it's kinda odd how Bob Katter is speaking on Billy Gordon's behalf' wrote Benjamin James. Not that John wrote: 'Wow, Bob Katter calls Billy Gordon, 'black puppet on a string' and the lefty loons stay silent. What if Abbott had said it?' The statement has sparked criticism on social media, with some users questioning his terminology Mr Katter has since defended the statement, saying he used the term black puppet in light of his personal experiences Mr Katter has since defended the statement, saying he used the term black puppet 'because I'm dark and I come from Cloncurry.' 'I can understand and I can identify with the attitudes that these people have and I'm going to call it for what it is,' he said. 'I deliberately put that comment in with a forethought, because I think they deserve a belting for the racial overtones.' Mr Katter used federal MPs Tony Windsor and Rob Oakeshott as examples of independents becoming puppets for the government and sacrificing their independence, but he vowed Mr Gordon would not make that mistake. output: bob katter has risked being drawn into a race row after accusing the queensland government of believing embattled mp billy gordon was ' a little black puppet on a string ' . mr katter said he 's been personally subjected to racist attitudes from both major parties in the past , and that he sympathised with mr gordon 's position . mr gordon was forced to quit after he was recently accused of domestic violence .",
            'answer': "{\"coherence\": 5, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary presents a clear sequence of events, focusing on Bob Katter's accusation and its implications. However, it could benefit from more context regarding the reactions and consequences to enhance the narrative flow.\n\n- **Consistency (5)**: The summary accurately reflects the key facts from the source text, including Katter's accusations and Gordon's situation, ensuring factual alignment without introducing discrepancies.\n\n- **Fluency (5)**: The text is grammatically correct and well-structured, making it easy to read and understand. The language is polished, contributing to overall fluency.\n\n- **Relevance (4)**: While the summary captures the main points about Katter's accusation and Gordon's resignation, it omits some details about the broader political implications and public reactions, which could provide a more comprehensive understanding.\n\nAspects like creativity, empathy, and humor were not selected as they are not pertinent to the task of summarizing factual content. Depth and cultural sensitivity were"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: i know brady can occasionally blow it but he did n't for me . i 'm a chiefs fan ...   even as a chiefs fan , do you think that tom brady is the greatest quarterback of all time , has he is commonly referred to .   he 's one of the best for sure . i go back to johnny unitas as a kid so that 's a tough call . all kinds of variables like kellan moore being left - handed .   apparently , kellan moore is the only left - handed quarterback in the nfl , i wonder if the defensive plays factor in that fact and are adjusted .   probably , maybe its like baseball and pitchers and batters play r / l - handedness . speaking of brady , i did n't know he played at u of michigan .   yeah , could be to the team 's advantage . yeah , i did n't know that , did brady play at umich on a scholarship ?   probably ... he was a 6th round draft pick in 2000 . i do like boston but i 'm not sure i 've ever been to foxborough where the pat 's play .   me neither but the gillette stadium in foxborough seems like a giant stadium . have you ever seen a live professional football game ?   yes , the falcons and the chiefs . i like college better , like georgia tech . they could use a brady qb however ...   haha , i agree , i really enjoy going to college football games . although i never knew that the huddle was created by a deaf quarterback so other teams could not read his hand signals .   did n't know that either ... i thought it was because they could n't hear . qb 's have to scream to be heard in most stadiums . some qbs are smart like alex smith who got a bachelors in 2 years .   that 's surprising , it seems like the focus on athletes is n't about school but rather heavy emphasis on sports so to see him go on to get his master 's degree , is very surprising to me .   i agree but some are both jocks and academics . i wonder if brady was ? i do admire him he 's seen hard times like to 41 - 33 loss to the eagles .   i definitely think that his leadership and tenacity and the so called \" patriot way \" can really shape people and lead them to their championships but his wife makes more than double brady 's salary ! i thought she was retired !    Corresponding Fact: bill belichick is the winningest coach in nfl postseason football with 21 wins . 20 with the new england patriots and 1 with the cleveland browns .  output: i think he has a great record , i think he is a great player , he is a great coach and also a great player , he has a great record",
            'answer': "{\"naturalness\": 4, \"coherence\": 2, \"engagingness\": 2, \"groundedness\": 1}",
            'knowledge': "- **Naturalness (2)**: The response is repetitive and awkward, lacking the fluidity expected in a natural conversation. The repeated use of \"great\" without variation makes it sound unnatural and forced.\n\n- **Coherence (2)**: The response does not connect well with the previous conversation. It mentions a \"great player\" and \"great coach\" without specifying who is being referred to, creating confusion and disrupting the flow of the dialogue.\n\n- **Engagingness (1)**: The repetitive nature of the response makes it unengaging and unlikely to prompt further discussion. It fails to introduce new information or provoke interest.\n\n- **Groundedness (1)**: The response does not effectively incorporate the provided fact about Bill Belichick. It lacks specific details or context, making it feel disconnected from the conversation.\n\nOther aspects like clarity, creativity, and accuracy were not selected because the response does not provide clear or creative insights, nor does it accurately address the given fact. The focus"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: i know brady can occasionally blow it but he did n't for me . i 'm a chiefs fan ...   even as a chiefs fan , do you think that tom brady is the greatest quarterback of all time , has he is commonly referred to .   he 's one of the best for sure . i go back to johnny unitas as a kid so that 's a tough call . all kinds of variables like kellan moore being left - handed .   apparently , kellan moore is the only left - handed quarterback in the nfl , i wonder if the defensive plays factor in that fact and are adjusted .   probably , maybe its like baseball and pitchers and batters play r / l - handedness . speaking of brady , i did n't know he played at u of michigan .   yeah , could be to the team 's advantage . yeah , i did n't know that , did brady play at umich on a scholarship ?   probably ... he was a 6th round draft pick in 2000 . i do like boston but i 'm not sure i 've ever been to foxborough where the pat 's play .   me neither but the gillette stadium in foxborough seems like a giant stadium . have you ever seen a live professional football game ?   yes , the falcons and the chiefs . i like college better , like georgia tech . they could use a brady qb however ...   haha , i agree , i really enjoy going to college football games . although i never knew that the huddle was created by a deaf quarterback so other teams could not read his hand signals .   did n't know that either ... i thought it was because they could n't hear . qb 's have to scream to be heard in most stadiums . some qbs are smart like alex smith who got a bachelors in 2 years .   that 's surprising , it seems like the focus on athletes is n't about school but rather heavy emphasis on sports so to see him go on to get his master 's degree , is very surprising to me .   i agree but some are both jocks and academics . i wonder if brady was ? i do admire him he 's seen hard times like to 41 - 33 loss to the eagles .   i definitely think that his leadership and tenacity and the so called \" patriot way \" can really shape people and lead them to their championships but his wife makes more than double brady 's salary ! i thought she was retired !    Corresponding Fact: bill belichick is the winningest coach in nfl postseason football with 21 wins . 20 with the new england patriots and 1 with the cleveland browns .  output: i think he has a great record , i think he is a great player , he is a great coach and also a great player , he has a great record",
            'answer': "{\"naturalness\": 4, \"coherence\": 2, \"engagingness\": 2, \"groundedness\": 1}",
            'knowledge': "- **Naturalness (2)**: The response lacks natural flow due to repetitive phrasing (\"great record,\" \"great player,\" \"great coach\"), making it awkward and difficult to read smoothly.\n- **Coherence (2)**: The response fails to coherently connect with the previous conversation, which focused on Tom Brady and his career, rather than Bill Belichick's coaching record.\n- **Engagingness (1)**: The repetitive nature of the response diminishes its ability to engage the reader or invite further discussion, as it doesn't introduce new or interesting information.\n- **Groundedness (1)**: The response does not accurately incorporate the provided fact about Bill Belichick's coaching record, instead repeating vague statements without context or relevance to the conversation.\n- **Clarity (2)**: The response is unclear due to its repetitive structure, making it difficult for the reader to discern the intended message or connection to the conversation.\n\nOther aspects such as empathy, creativity, and depth were"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: i know brady can occasionally blow it but he did n't for me . i 'm a chiefs fan ...   even as a chiefs fan , do you think that tom brady is the greatest quarterback of all time , has he is commonly referred to .   he 's one of the best for sure . i go back to johnny unitas as a kid so that 's a tough call . all kinds of variables like kellan moore being left - handed .   apparently , kellan moore is the only left - handed quarterback in the nfl , i wonder if the defensive plays factor in that fact and are adjusted .   probably , maybe its like baseball and pitchers and batters play r / l - handedness . speaking of brady , i did n't know he played at u of michigan .   yeah , could be to the team 's advantage . yeah , i did n't know that , did brady play at umich on a scholarship ?   probably ... he was a 6th round draft pick in 2000 . i do like boston but i 'm not sure i 've ever been to foxborough where the pat 's play .   me neither but the gillette stadium in foxborough seems like a giant stadium . have you ever seen a live professional football game ?   yes , the falcons and the chiefs . i like college better , like georgia tech . they could use a brady qb however ...   haha , i agree , i really enjoy going to college football games . although i never knew that the huddle was created by a deaf quarterback so other teams could not read his hand signals .   did n't know that either ... i thought it was because they could n't hear . qb 's have to scream to be heard in most stadiums . some qbs are smart like alex smith who got a bachelors in 2 years .   that 's surprising , it seems like the focus on athletes is n't about school but rather heavy emphasis on sports so to see him go on to get his master 's degree , is very surprising to me .   i agree but some are both jocks and academics . i wonder if brady was ? i do admire him he 's seen hard times like to 41 - 33 loss to the eagles .   i definitely think that his leadership and tenacity and the so called \" patriot way \" can really shape people and lead them to their championships but his wife makes more than double brady 's salary ! i thought she was retired !    Corresponding Fact: bill belichick is the winningest coach in nfl postseason football with 21 wins . 20 with the new england patriots and 1 with the cleveland browns .  output: i think he has a great record , i think he is a great player , he is a great coach and also a great player , he has a great record",
            'answer': "{\"naturalness\": 4, \"coherence\": 2, \"engagingness\": 2, \"groundedness\": 1}",
            'knowledge': "- **Naturalness (2)**: The response lacks natural flow due to repetitive phrasing, making it awkward and difficult to read. The repetition of \"great\" without variation or elaboration disrupts the conversational tone.\n  \n- **Coherence (2)**: The response fails to connect meaningfully with the previous conversation about Tom Brady and his achievements. It mentions a \"great record\" but does not specify who it refers to, creating confusion.\n\n- **Engagingness (1)**: The repetitive nature of the response makes it unengaging and unlikely to provoke further discussion. It does not introduce new information or insights that could capture interest.\n\n- **Groundedness (1)**: The response does not incorporate the provided fact about Bill Belichick's record. It remains vague and lacks the specific grounding needed to enhance the conversation with relevant information.\n\nOther aspects such as empathy, creativity, and depth were not selected because the response does not explore emotional connections, offer original ideas, or provide"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: i know brady can occasionally blow it but he did n't for me . i 'm a chiefs fan ...   even as a chiefs fan , do you think that tom brady is the greatest quarterback of all time , has he is commonly referred to .   he 's one of the best for sure . i go back to johnny unitas as a kid so that 's a tough call . all kinds of variables like kellan moore being left - handed .   apparently , kellan moore is the only left - handed quarterback in the nfl , i wonder if the defensive plays factor in that fact and are adjusted .   probably , maybe its like baseball and pitchers and batters play r / l - handedness . speaking of brady , i did n't know he played at u of michigan .   yeah , could be to the team 's advantage . yeah , i did n't know that , did brady play at umich on a scholarship ?   probably ... he was a 6th round draft pick in 2000 . i do like boston but i 'm not sure i 've ever been to foxborough where the pat 's play .   me neither but the gillette stadium in foxborough seems like a giant stadium . have you ever seen a live professional football game ?   yes , the falcons and the chiefs . i like college better , like georgia tech . they could use a brady qb however ...   haha , i agree , i really enjoy going to college football games . although i never knew that the huddle was created by a deaf quarterback so other teams could not read his hand signals .   did n't know that either ... i thought it was because they could n't hear . qb 's have to scream to be heard in most stadiums . some qbs are smart like alex smith who got a bachelors in 2 years .   that 's surprising , it seems like the focus on athletes is n't about school but rather heavy emphasis on sports so to see him go on to get his master 's degree , is very surprising to me .   i agree but some are both jocks and academics . i wonder if brady was ? i do admire him he 's seen hard times like to 41 - 33 loss to the eagles .   i definitely think that his leadership and tenacity and the so called \" patriot way \" can really shape people and lead them to their championships but his wife makes more than double brady 's salary ! i thought she was retired !    Corresponding Fact: bill belichick is the winningest coach in nfl postseason football with 21 wins . 20 with the new england patriots and 1 with the cleveland browns .  output: i think he has a great record , i think he is a great player , he is a great coach and also a great player , he has a great record",
            'answer': "{\"naturalness\": 4, \"coherence\": 2, \"engagingness\": 2, \"groundedness\": 1}",
            'knowledge': "- **Naturalness (3):** The response uses conversational language but feels somewhat repetitive and lacks a smooth flow, making it less natural to read.\n\n- **Coherence (2):** The response does not connect well with the prior conversation, which focused on women in the NFL and the strategic decisions of coaches like Bill Belichick. The statement about Belichick being the best coach is disconnected from the previous dialogue.\n\n- **Relevance (2):** The response introduces a broad claim about Belichick's status without tying it to the specific context of the conversation or the provided fact about his playoff history with the Browns.\n\n- **Groundedness (2):** The response fails to incorporate the provided fact about Belichick's history with the Browns, missing an opportunity to ground the conversation in specific historical context.\n\nOther aspects like creativity, empathy, and depth were not selected as the response does not demonstrate originality, emotional engagement, or detailed exploration of the topic. The focus here is"
        }
    ]
    eval_sample_data = [
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Everton manager Roberto Martinez was forced to defend another penalty fiasco at the club after Ross Barkley missed from the spot in their 1-0 win against Burnley at Goodison Park. The untried Barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by David Jones on Aaron Lennon \u2013 rather than Leighton Baines, who has scored 15 penalties from 16 attempts in the Premier League. Although there was no dispute between the team-mates this time, it brought back memories of Everton's match against West Brom in January when Kevin Mirallas grabbed the ball from Baines to take a penalty - and missed. Ross Barkley steps up to take a 10th minute penalty despite the presence of Leighton Baines on the pitch Barkley's effort is saved by\u00a0Burnley goalkeeper Tom Heaton at Goodison Park Martinez insisted Barkley was within his rights to request penalty-taking duties on Saturday. 'If Romelu Lukaku had been on the pitch, he would have taken it. Otherwise, I am happy to have three or four players who can take penalties and let it depend on how they feel at that moment,' argued the Everton manager. Baines (left)\u00a0has scored 15 penalties from 16 attempts in the Premier League 'Ross showed incredible responsibility to take it. I love seeing players take control of the big moments and Leighton was happy to given him that responsibility.' Barkley's penalty was well-struck but wasn't put in the corner and Burnley goalkeeper Tom Heaton dived to his right to save. Fortunately for the young England player, it didn't prove costly as Mirallas went on to score the only goal of the game after 29 minutes. Everton boss Roberto Martinez issues instructions to his players during a break in play against Burnley output: everton manager roberto martinez was forced to defend another penalty fiasco at the club after ross barkley missed from the spot in their 1 - 0 win against burnley at goodison park . the untried barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by david jones on aaron lennon \u2013 rather than leighton baines , who has scored 15 penalties from 16 attempts in the premier league . martinez insisted barkley was within his rights to request penalty - taking duties on saturday .",
            'answer': "{\"coherence\": 4, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary presents a clear narrative about the penalty incident involving Ross Barkley and Roberto Martinez's defense of it. However, it could improve by integrating more context about the match outcome and previous incidents for a fuller picture.\n\n- **Consistency (5)**: The summary accurately reflects the key facts from the source text, such as the missed penalty by Barkley and Martinez's comments, ensuring factual alignment without discrepancies.\n\n- **Fluency (5)**: The text is grammatically correct and flows well, making it easy to read and understand. There are no awkward constructions or errors that would hinder comprehension.\n\n- **Relevance (4)**: The summary includes crucial information about the penalty incident and Martinez's response but omits details about the match's final score and Barkley's performance, which could provide a more comprehensive understanding.\n\nAspects like creativity, empathy, and humor are not relevant to the task of summarizing factual information. Depth and inclusivity were not chosen"
        },
        {
            'task': 'summeval', 'task_ix': 0,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given one summary written for a news article(the following Source Text). Your task is to rate the summary on appropriate metrics. Source Text: Everton manager Roberto Martinez was forced to defend another penalty fiasco at the club after Ross Barkley missed from the spot in their 1-0 win against Burnley at Goodison Park. The untried Barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by David Jones on Aaron Lennon \u2013 rather than Leighton Baines, who has scored 15 penalties from 16 attempts in the Premier League. Although there was no dispute between the team-mates this time, it brought back memories of Everton's match against West Brom in January when Kevin Mirallas grabbed the ball from Baines to take a penalty - and missed. Ross Barkley steps up to take a 10th minute penalty despite the presence of Leighton Baines on the pitch Barkley's effort is saved by\u00a0Burnley goalkeeper Tom Heaton at Goodison Park Martinez insisted Barkley was within his rights to request penalty-taking duties on Saturday. 'If Romelu Lukaku had been on the pitch, he would have taken it. Otherwise, I am happy to have three or four players who can take penalties and let it depend on how they feel at that moment,' argued the Everton manager. Baines (left)\u00a0has scored 15 penalties from 16 attempts in the Premier League 'Ross showed incredible responsibility to take it. I love seeing players take control of the big moments and Leighton was happy to given him that responsibility.' Barkley's penalty was well-struck but wasn't put in the corner and Burnley goalkeeper Tom Heaton dived to his right to save. Fortunately for the young England player, it didn't prove costly as Mirallas went on to score the only goal of the game after 29 minutes. Everton boss Roberto Martinez issues instructions to his players during a break in play against Burnley output: everton manager roberto martinez was forced to defend another penalty fiasco at the club after ross barkley missed from the spot in their 1 - 0 win against burnley at goodison park . the untried barkley inexplicably took the 10th minute kick \u2013 awarded for a foul by david jones on aaron lennon \u2013 rather than leighton baines , who has scored 15 penalties from 16 attempts in the premier league . martinez insisted barkley was within his rights to request penalty - taking duties on saturday .",
            'answer': "{\"coherence\": 4, \"consistency\": 5, \"fluency\": 5, \"relevance\": 5}",
            'knowledge': "- **Coherence (4)**: The summary presents a clear sequence of events, focusing on the penalty incident involving Ross Barkley. However, it could benefit from smoother transitions between sentences to enhance the narrative flow.\n\n- **Consistency (5)**: The summary accurately reflects the facts from the source text, including the key details about the penalty incident and Roberto Martinez's defense of Barkley, ensuring factual alignment.\n\n- **Fluency (5)**: The text is grammatically correct and well-structured, making it easy to read and understand without any awkward phrasing or errors.\n\n- **Relevance (4)**: The summary captures the main incident and the context of the penalty decision but omits some details, such as the outcome of the match and the previous penalty incident with Kevin Mirallas, which could provide a more comprehensive understanding.\n\nAspects like creativity, empathy, and humor were not selected as they are not pertinent to the task of summarizing factual content. Depth and inclusivity were"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: hello , did you know that more than 80 % of the earths forests have been destroyed ?   that is sad . i hope we can reverse that statistic .   yeah that is really sad . i recently read that earth will be helium free by the end of the 21st century .   i also think it is interesting that when earth was formed , a day was only 5.5 hours long . i wonder why that changed ?   i do n't know , maybe it slowed its rotation eventually ? that 's pretty weird .    Corresponding Fact: in september of 2010 , the united nations appointed official ambassador to extraterrestrials in they case they would ever make contact with earth  output: wow . the un appointed an official ambassador to aliens ! maybe we can ask them for help if we run out of helium on earth .",
            'answer': "{\"naturalness\": 4, \"coherence\": 4, \"engagingness\": 5, \"groundedness\": 5}",
            'knowledge': "- **Naturalness (4)**: The response maintains a conversational tone and is easy to read, but the abrupt transition to extraterrestrials feels slightly forced, affecting the overall flow.\n- **Coherence (3)**: While the response connects to the conversation by mentioning the U.N. ambassador, it introduces a new topic without a clear link to the ongoing discussion about Earth's rotation and resources, resulting in a disjointed narrative.\n- **Engagingness (3)**: The mention of an ambassador to aliens is intriguing and could spark interest, but the connection to the previous conversation is weak, limiting its engagement potential.\n- **Groundedness (3)**: The response includes the fact about the U.N. ambassador, but it is not well integrated into the conversation, lacking a direct link to the previous topics discussed.\n- **Relevance (3)**: The response is partially relevant as it introduces a new fact, but it does not directly relate to the prior discussion about Earth's natural phenomena"
        },
        {
            'task': 'topical-chat', 'task_ix': 1,
            'question': "For the following task, if the following output is obtained, choose appropriate evaluation aspects from the following aspects list to assess the quality of the output and assign scores(1-5) to those aspects as well. aspects list: [\"coherence\", \"consistency\", \"fluency\", \"relevance\", \"naturalness\", \"engagingness\", \"groundedness\", \"clarity\", \"creativity\", \"empathy\", \"adaptability\", \"depth\", \"accuracy\", \"inclusivity\", \"persuasiveness\", \"formatting\", \"cultural sensitivity\", \"humor or emotional appeal\", \"interactivity\", \"robustness\"] task: You will be given a conversation between two individuals(the following Conversation History). You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well. Your task is to rate the responses on appropriate metrics. Conversation History: hello , did you know that more than 80 % of the earths forests have been destroyed ?   that is sad . i hope we can reverse that statistic .   yeah that is really sad . i recently read that earth will be helium free by the end of the 21st century .   i also think it is interesting that when earth was formed , a day was only 5.5 hours long . i wonder why that changed ?   i do n't know , maybe it slowed its rotation eventually ? that 's pretty weird .    Corresponding Fact: in september of 2010 , the united nations appointed official ambassador to extraterrestrials in they case they would ever make contact with earth  output: wow . the un appointed an official ambassador to aliens ! maybe we can ask them for help if we run out of helium on earth .",
            'answer': "{\"naturalness\": 4, \"coherence\": 4, \"engagingness\": 5, \"groundedness\": 5}",
            'knowledge': "- **Naturalness (4)**: The response is mostly natural and conversational, with a playful tone that fits the context. However, the sudden shift to aliens may seem slightly out of place in the discussion about Earth's resources.\n\n- **Coherence (3)**: While the response acknowledges the provided fact about the U.N. ambassador, it doesn't logically connect with the prior conversation about Earth's environmental issues, leading to a somewhat disjointed narrative.\n\n- **Engagingness (3)**: The mention of an ambassador to aliens is intriguing and could capture interest, but the lack of a direct connection to the ongoing conversation might limit engagement.\n\n- **Groundedness (3)**: The response incorporates the fact about the U.N. ambassador, but the connection to helium and Earth's issues is tenuous, reducing the relevance and depth of integration.\n\nAspects like empathy, depth, and accuracy were not selected because the response focuses more on introducing an interesting fact rather than deeply exploring the topic or expressing emotional depth"
        }
    ]
    train_dataloader = DataLoader(train_sample_data, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_sample_data, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    print(f"Batch size: {args.batch_size}, len(train_dataloader): {len(train_dataloader)}")

    # Initializing tokenizer
    print(f'Initializing tokenizer ...')
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, padding_side="left")  # padding_sideは追加
    tokenizer.max_question_len = args.max_question_len
    tokenizer.max_answer_len = args.max_answer_len
    tokenizer.max_knowledge_len = args.max_knowledge_len
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize models and optimizer
    print(f'Initializing models ...')
    # HuggingFace にアクセスするためのトークン設定
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # もしtokenが設定されていなければエラーを出すか、警告を出す
    if not token:
        print("Warning: Hugging Face token not set.")

    # 量子化、LoRAの流れの全般知識について: https://huggingface.co/docs/peft/developer_guides/quantization
    # 特に参考にしたQLoRAについての記述: https://huggingface.co/docs/peft/accelerate/deepspeed#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus
    # 以下はQLoRA用に追加 configの設定
    bnb_config = None
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        # https://docs.python.org/3/library/functions.html#getattr より 今回の場合、quant_storage_dtypeにはtorch.bfloat16が代入される
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)  

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:  # 今回の設定(compute_dtype=bfloat16)ではここは実行されない
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:  # 今回の設定では実行されない
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    # Load model
    # https://docs.pytorch.org/docs/stable/tensor_attributes.html よりis_floating_pointはtorch.dtypeがfloating point data typeかどうかみるproperty(真偽値)
    torch_dtype = (
        quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch_dtype, token=token
    )

    peft_config = None
    # chat_template = None  # chat_template も tokenizer に自動で設定されるため、明示的に設定する必要はほぼない
    if args.use_peft_lora and not args.use_unsloth:  # ここは今回の設定で実行される
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )
    
    # 量子化されたモデルに対して推奨される前処理を実行
    # これにより、勾配チェックポイントの有効化や、LayerNormの学習などが適切に設定される
    ## https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model#peft.prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)  # 今回の設定でgradient_checkpointing はTrueになっている
    # https://github.com/huggingface/trl/issues/835 も参考にしつつ gradient_checkpointing_kwargs={"use_reentrant": False}に
    gradient_checkpointing_kwargs = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
    )

    if peft_config:
        model = get_peft_model(model, peft_config)

    # gradient ckpt
    # ここまでの設定ではgradient_checkpointingはTrueになっている
    model.config.use_cache = not args.gradient_checkpointing
    # ここまでの設定でuse_unslothはFalseになっている
    args.gradient_checkpointing = args.gradient_checkpointing and not args.use_unsloth
    # if args.gradient_checkpointing:  # ここは今回の設定で実行される
    #     args.gradient_checkpointing_kwargs = {"use_reentrant": args.use_reentrant}  # prepare_model_for_kbit_training()の前に明示的にFalseに設定したのでここはコメントアウト

    model.eval()
    # args.learning_rate が `get_args()` で定義されていることを確認
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    
    print(f"len(train_dataloader) after accelerator.prepare: {len(train_dataloader)}")

    # set for test
    init_step=0
    eval_losses=[]
    eval_accs=[]

    # Set up trainer
    trainer = Trainer(
        args=args,
        accelerator=accelerator,  # accelerator を事前に引数 gradient_accumulation_steps 設定の上使いたかったので追加
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        init_step=init_step,
        eval_losses=eval_losses,
        eval_accs=eval_accs,
    )

    # Train
    steps = list(range(init_step, args.total_steps + 1))
    steps = tqdm(steps) if accelerator.is_main_process else steps
    for step in steps:
        trainer.train(step)


if __name__ == '__main__':
    main()

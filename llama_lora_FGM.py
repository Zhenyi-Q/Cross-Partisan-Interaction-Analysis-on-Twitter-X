from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import numpy as np
import random
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType


data = load_dataset("cardiffnlp/sentiment")
data["val"] = data["validation"]
del data["validation"]



llama_path = "/root/emotion_classification/Meta-Llama-3.1-8B"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token
def preprocessing_function(examples):
    examples['label'] = [int(i) for i in examples['label']]
    return llama_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)

tokenized_data = data.map(preprocessing_function, batched=True, remove_columns= ["text"])
tokenized_data.set_format("torch")
# 创建 DataCollatorWithPadding 实例
data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)




pretrain_model = AutoModelForSequenceClassification.from_pretrained(llama_path, 
                                                                 num_labels=3,
                                                                device_map="auto",
                                                                offload_folder="offload",
                                                                trust_remote_code=True)
pretrain_model.config.pad_token_id = llama_tokenizer.pad_token_id
# llama_model.config.use_cache = False
# llama_model.config.pretraining_tp = 1



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # eval_pred 是模型返回的预测值和实际值元组
    predictions = np.argmax(logits, axis=-1)
    
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    print(f"precision: {precision}, recall: {recall}, f1-score: {f1}, accuracy: {accuracy}")
    # 返回包含所有指标的字典
    return {"precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy}



# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(seed)


def evaluate(model, dataloader):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_logits, all_labels

# 执行评估并计算指标
# logits, labels = evaluate(model, val_dataloader)
# metrics = compute_metrics((logits, labels))

# # 输出结果
# print("Evaluation Results:")
# for key, value in metrics.items():
#     print(f"{key}: {value:.4f}")

test_dataloader = DataLoader(
    tokenized_data["test"],
    batch_size=16,  # 根据硬件资源调整 batch_size
    shuffle=False,
    collate_fn=data_collator
)



class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 对抗训练，只在embedding层添加扰动
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 恢复embedding层的参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

from transformers import Trainer

class CustomTrainer(Trainer):
    def __init__(self, *args, fgm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm = fgm

    def training_step(self, model, inputs):
        # 正常的前向传递和损失计算
        loss = super().training_step(model, inputs)
        
        if self.fgm is not None:
            # 使用FGM生成对抗样本
            self.fgm.attack()  # 在原始输入上加扰动
            loss_adv = super().training_step(model, inputs)  # 再次计算损失
            loss = (loss + loss_adv) / 2  # 将原始损失与对抗损失结合
            self.fgm.restore()  # 恢复模型参数
        
        return loss


batch_size = 32
num_epochs = 5
lr_pars = [2e-04, 1e-05]
lora_ranks = [16, 8]
lora_alphas = [32]

for lr in lr_pars:
    for lora_rank in lora_ranks:
        for lora_alpha in lora_alphas:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=0.05, 
                bias="none",
                target_modules=[
                    "q_proj", "v_proj"
                ],
            )

            llama_model = get_peft_model(pretrain_model, lora_config)
            llama_model.print_trainable_parameters()

            # 初始化FGM
            fgm = FGM(llama_model)

            training_args = TrainingArguments(
                output_dir=f"./results/lr={lr}_lora_alpha={lora_alpha}_lora_r={lora_rank}",
                learning_rate=lr,
                lr_scheduler_type="constant",
                warmup_ratio=0.1,
                max_grad_norm=0.3,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=0.001,
                eval_strategy="epoch",
                save_strategy="epoch",
                metric_for_best_model="eval_f1-score",
                greater_is_better=True,
                save_total_limit=3, 
                load_best_model_at_end=True,
                fp16=True,
                gradient_checkpointing=True,
                # disable_tqdm=True,  # 禁用进度条
                report_to=["none"],  # 禁用 wandb 报告
            )

            # 使用自定义的Trainer进行训练
            trainer = CustomTrainer(
                model=llama_model,
                args=training_args,
                train_dataset=tokenized_data['train'],
                eval_dataset=tokenized_data['val'],
                tokenizer=llama_tokenizer,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                fgm=fgm  # 传入FGM实例
            )

            # 开始训练
            print(f"------lr={lr}_lora_rank={lora_rank}_lora_alpha={lora_alpha}")
            trainer.train()
            trainer.save_model(f"./results/lr={lr}_lora_alpha={lora_alpha}_lora_r={lora_rank}/best_model")

            logits, labels = evaluate(llama_model, test_dataloader)
            metrics = compute_metrics((logits, labels))

            # 输出结果
            print("test Results:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

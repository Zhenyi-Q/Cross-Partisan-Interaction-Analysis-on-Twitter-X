import pandas as pd
from datasets import Dataset
import numpy as np

reply_to_keep = ['id','conversation_id','referenced_tweets.replied_to.id','author_id','formatted_text',"theta", "accounts_followed","sentiment"]
poster_to_keep = ['id','author_id','formatted_text','topic','context_annotations','event',"theta", "accounts_followed","sentiment"]
dtypes = {'id':str,'conversation_id':str,'referenced_tweets.replied_to.id' : str,"theta": np.float64}

# 读取 CSV 文件
print("Start load files")
replies = pd.read_csv('filtered_replies.csv', usecols=lambda column: column in reply_to_keep, dtype=dtypes)
posters = pd.read_csv("all_posters.csv", usecols=lambda column: column in poster_to_keep)
print("load files successful")

posters_text = posters["formatted_text"]
replies_text = replies["formatted_text"]

from datasets import DatasetDict
posters_dict = {"text": posters_text.tolist()}
replies_dict = {"text": replies_text.tolist()}

data = DatasetDict({
    "posters" : Dataset.from_dict(posters_dict),
    "replies": Dataset.from_dict(replies_dict)
})

from transformers import AutoTokenizer, DataCollatorWithPadding
llama_path = "./Meta-Llama-3.1-8B"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token
def preprocessing_function(examples):
    return llama_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_data = data.map(preprocessing_function, batched=True, remove_columns= ["text"])
tokenized_data.set_format("torch")
# 创建 DataCollatorWithPadding 实例
data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)

from peft import PeftModel

from transformers import AutoModelForSequenceClassification
import torch

pretrain_model = AutoModelForSequenceClassification.from_pretrained(llama_path, 
                                                                 num_labels=3,
                                                                device_map="auto",
                                                                offload_folder="offload",
                                                                trust_remote_code=True)
pretrain_model.config.pad_token_id = llama_tokenizer.pad_token_id
# llama_model.config.use_cache = False
# llama_model.config.pretraining_tp = 1

# 加载微调后的权重
lora_weights_path = "/root/emotion_classification/results/lr=0.0002_lora_alpha=32_lora_r=8/best_model"  # 这里填写你LoRA微调后的权重路径
model = PeftModel.from_pretrained(pretrain_model, lora_weights_path)

from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
posters_dataloader = DataLoader(
    tokenized_data["posters"],
    batch_size=16,  # 根据硬件资源调整 batch_size
    shuffle=False,
    collate_fn=data_collator
)
replies_dataloader = DataLoader(
    tokenized_data["replies"],
    batch_size=16,  # 根据硬件资源调整 batch_size
    shuffle=False,
    collate_fn=data_collator
)

from torch.amp import autocast
def evaluate(model, dataloader):
    model.eval()
    predictions = []
    
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        with torch.no_grad():
            with autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(prediction)
   
    return predictions
posters_output = evaluate(model, posters_dataloader)
replies_output = evaluate(model, replies_dataloader)

posters['sentiment'] = posters_output
replies['sentiment'] = replies_output

posters.to_csv('posters_with_sentiment.csv', index=False)
replies.to_csv('replies_with_sentiment.csv', index=False)

print("Finish!!!!!")
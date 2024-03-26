# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM,AutoTokenizer,BertTokenizer,NezhaModel,ErnieForMaskedLM,NezhaForMaskedLM,GPT2ForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math

model_name = "E:\项目\COLING2022-MedicalDialog_data and code\GPTAUTODL\pretrained\GPT2"
# 训练集
train_file = "train.txt"
# 验证集
eval_file = "valid.txt"
max_seq_length = 512
out_model_path = "pretain"
train_epoches = 50
batch_size = 256

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
model = NezhaForMaskedLM.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=128,
)

training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        save_total_limit=4,
        learning_rate=1e-4,
        prediction_loss_only=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
trainer.train()

trainer.save_model(out_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
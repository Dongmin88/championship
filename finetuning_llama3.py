import ollama
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, EvalPrediction
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# IMDB 데이터셋 로드
dataset = load_dataset("imdb")

# 모델 및 토크나이저 로드
model = ollama.AutoModel.from_pretrained("llama3")
tokenizer = ollama.AutoTokenizer.from_pretrained("llama3")

# 데이터셋 전처리
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 데이터 로더 준비
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # 샘플링 데이터
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

# F1 스코어 계산 함수
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    report = classification_report(p.label_ids, preds, output_dict=True)
    f1 = report["weighted avg"]["f1-score"]
    return {"f1": f1}

# 파인튜닝 설정 및 실행
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 모델 저장
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 모델 평가
eval_results = trainer.evaluate()
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

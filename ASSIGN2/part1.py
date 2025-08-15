import torch
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from datasets import load_dataset, ClassLabel
from tqdm import tqdm

class Config:
    bottleneck_dimensions = [8, 32, 64, 128]
    bert_model = "bert-base-uncased"
    num_labels = 150
    noise_std = 0.1 
    max_sentence_length = 136 # max sent length from train dataset
    batch_size = 32
    bottleneck_lr = 1e-3
    bert_lr = 2e-5 # I will use this for fine-tuning bert layers
    epochs = 5
    dropout_prob = 0.1 # For trying

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset():
    dataset = load_dataset("Deeppavlov/clinc150")

    print("Sample example:", dataset["train"][0])

    print(f"Train size before filtering: {len(dataset['train'])}")
    print(f"Validation size before filtering: {len(dataset['validation'])}")
    print(f"Test size before filtering: {len(dataset['test'])}")

    # As I saw some label were of NoneType, so I removed them
    def filter_none_labels(example):
        return example["label"] is not None

    dataset = dataset.filter(
        filter_none_labels,
        batched=False
    )

    print(f"Train size after filtering: {len(dataset['train'])}")
    print(f"Validation size after filtering: {len(dataset['validation'])}")
    print(f"Test size after filtering: {len(dataset['test'])}")

    # max_sentence_length = 0
    # for example in dataset['train']:
    #     if len(example['utterance']) > max_sentence_length:
    #         max_sentence_length = len(example['utterance'])
    # print(f"Max sentence length in dataset['train']: {max_sentence_length}")


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["utterance"],
            padding="max_length",
            truncation=True,
            max_length=Config.max_sentence_length,
            return_tensors="pt"
        )

    tokenized_datasets = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["utterance"]
    )

    tokenized_datasets.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        tokenized_datasets["test"]
    )

train_data, validation_data, test_data = prepare_dataset()

def create_dataloader(dataset, batch_size = 32):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2
    )



class BertBottleneckClassifier(nn.Module):
    def __init__(self, bottleneck_dimension):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.bert_model)
        self.bottleneck = nn.Linear(self.bert.config.hidden_size, bottleneck_dimension)
        self.classifier = nn.Linear(bottleneck_dimension, Config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids = None):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        cls_embedding = bert_out.last_hidden_state[:, 0, :]
        compressed = self.bottleneck(cls_embedding)
        return self.classifier(compressed)


def train_model(bottleneck_dimension):
    model = BertBottleneckClassifier(bottleneck_dimension).to(device)

    optimizer = AdamW([
        {"params": model.bottleneck.parameters(), "lr": Config.bottleneck_lr},
        {"params": model.classifier.parameters(), "lr": Config.bottleneck_lr}
    ])

    train_loader = create_dataloader(train_data, Config.batch_size)
    val_loader = create_dataloader(validation_data, Config.batch_size)


    best_val_acc = 0
    best_model = None

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc = f"Training X = {bottleneck_dimension} ----> Epoch = {epoch + 1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            classification_loss = nn.CrossEntropyLoss()(outputs, labels)
            classification_loss.backward()
            optimizer.step()

            total_loss += classification_loss.item()

        # Model evaluation on validation dataset
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                labels = batch["label"].to(device)

                outputs = model(**inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch = {epoch+1}: Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    # Load best model
    model.load_state_dict(best_model)
    return model

def test_model(model):
    test_loader = create_dataloader(test_data, Config.batch_size)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)

            outputs = model(**inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

results = {}

for dim in Config.bottleneck_dimensions:
    print(f"\n{'-'*50} Bottleneck dimension: {dim} {'-'*50}\n")

    model = train_model(dim)
    test_acc = test_model(model)
    results[dim] = test_acc
    print(f"Test Accuracy for X={dim}: {test_acc:.4f}")


dims = sorted(results.keys())
part1_accs = [results[d] for d in dims]
print(part1_accs)

plt.figure(figsize=(10, 6))
plt.plot(dims, part1_accs, 'bo-')
plt.xscale('log', base=2)
plt.xticks(dims, labels = dims)
plt.xlabel('Bottleneck Dimension (X)')
plt.ylabel('Test Accuracy')
plt.title('Classification Accuracy vs Bottleneck Width (Bottleneck without Reconstruction)')
plt.grid(True)
info_text = f"Epochs: {Config.epochs}\nBatch Size: {Config.batch_size}\nTraining Pairs: {len(train_data)}\nValidation Pairs: {len(validation_data)}\nTest Pairs: {len(test_data)}"
plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()

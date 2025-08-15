import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

def parse_input_file(file_path):
    # Initialize lists
    anchor = ""
    pros = []
    cons = []

    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Discussion Title:"):
            anchor = line.split("Discussion Title:", 1)[1].strip()

        elif "Pro:" in line:
            clean_line = re.sub(r"\[.*?\]\(.*?\)", "", line.split("Pro:", 1)[1].strip())
            pros.append(clean_line)

        elif "Con:" in line:
            clean_line = re.sub(r"\[.*?\]\(.*?\)", "", line.split("Con:", 1)[1].strip())
            cons.append(clean_line)

    return anchor, pros, cons


# Example usage
file_path = "./Data/File1.txt"
anchor, pros, cons = parse_input_file(file_path)
# print(anchor)
# print(pros)
# print(cons)

pairs = []
for pro in pros:
    pairs.append({
        "first": anchor,
        "second": pro,
        "label": 1
    })

for pro in pros:
    for con in cons:
        pairs.append({
            "first": pro,
            "second": con,
            "label": 0
        })


class SiameseNetwork(nn.Module):
    def __init__(self, base_model_name="all-mpnet-base-v2"):
        super(SiameseNetwork, self).__init__()
        self.encoder = SentenceTransformer(base_model_name)

    def forward(self, input1, input2):
        # Both inputs using the same encoder
        embedding1 = self.encoder.encode(input1, convert_to_tensor=True)
        embedding2 = self.encoder.encode(input2, convert_to_tensor=True)
        return embedding1, embedding2

siamese_net = SiameseNetwork()

class SiameseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['first'], item['second'], float(item['label'])

dataset = SiameseDataset(pairs)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.CosineEmbeddingLoss()

optimizer = optim.Adam(siamese_net.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    siamese_net.train()
    running_loss = 0.0

    for batch in train_loader:
        input1, input2, labels = batch
        optimizer.zero_grad()
        embedding1, embedding2 = siamese_net(input1, input2)
        loss = criterion(embedding1, embedding2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

print("Finished Training")

siamese_net.eval()
val_embeddings_first = []
val_embeddings_second = []
val_labels = []

with torch.no_grad():
    for input1, input2, label in train_loader:
        embedding1, embedding2 = siamese_net(input1, input2)
        val_embeddings_first.append(embedding1)
        val_embeddings_second.append(embedding2)
        val_labels.append(label)

similarities = [
    cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    for emb1, emb2 in zip(val_embeddings_first, val_embeddings_second)
]

similar_pairs = [sim for sim, label in zip(similarities, val_labels) if label == 1]
dissimilar_pairs = [sim for sim, label in zip(similarities, val_labels) if label == 0]

import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(similar_pairs, label="Similar Pairs", fill=True)
sns.kdeplot(dissimilar_pairs, label="Dissimilar Pairs", fill=True)
plt.xlabel("Cosine Similarity")
plt.title("Similarity Distribution")
plt.legend()
plt.show()

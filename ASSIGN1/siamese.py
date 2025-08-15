import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as contrastive_loss

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
file_path = "./Data/should-prisons-exist-18590.txt"
anchor, pros, cons = parse_input_file(file_path)

# print(anchor)
# print(pros)
# print(cons)

pairs = []
for pro in pros:
    for con in cons:
        pairs.append({
            "anchor": anchor,
            "pro": pro,
            "con": con,
        })

print(len(pairs))
print(pairs[0:5])



# train_examples = [
#     InputExample(texts=[pair["first"], pair["second"]], label=pair["label"])
#     for pair in pairs
# ]
# train_loader = DataLoader(train_examples, batch_size=1)

# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# loss = losses.CosineSimilarityLoss(model=model)
# model.fit(train_objectives=[(train_loader, loss)], epochs=10)

# anchor_embedding = model.encode(anchor)
# pro_embeddings = model.encode(pros)
# con_embeddings = model.encode(cons)

# pro_similarities = contrastive_loss([anchor_embedding], pro_embeddings).flatten()
# con_similarities = contrastive_loss([anchor_embedding], con_embeddings).flatten()

# sns.kdeplot(pro_similarities, label="Pro Arguments", fill=True)
# sns.kdeplot(con_similarities, label="Con Arguments", fill=True)
# plt.xlabel("Cosine Similarity")
# plt.title("Similarity to Anchor Claim")
# plt.legend()
# plt.show()
import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as contrastive_loss
import os
import random

def parse_input_file(file_path):
    anchor, pros, cons = "", [], []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Discussion Title:"):
            anchor = line.split("Discussion Title:", 1)[1].strip()
        elif "Pro:" in line:
            clean_line = re.sub(r"\[.*?\]\(.*?\)", "", line.split("Pro:", 1)[1].strip())
            pros.append(clean_line)
        elif "Con:" in line:
            clean_line = re.sub(r"\[.*?\]\(.*?\)", "", line.split("Con:", 1)[1].strip())
            cons.append(clean_line)
    return anchor, pros, cons

folder_path = "./Data"

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
txt_files = random.sample(txt_files, min(10, len(txt_files)))
triplet_pairs = []

for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    anchor, pros, cons = parse_input_file(file_path)
    for pro in pros:
        for con in cons:
            triplet_pairs.append({
                "pro": pro,
                "con": con,
                "anchor": anchor
            })
print(len(triplet_pairs))
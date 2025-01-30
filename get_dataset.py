import kagglehub
import os
import pandas as pd

def download_prepare_dataset():
    path = kagglehub.dataset_download("haithemhermessi/sanad-dataset")
    categories = []
    texts = []
    for category in os.listdir(path):
        for text in os.listdir(os.path.join(path, category)):
            categories.append(category)
            f = open(os.path.join(path, category, text), "r")
            texts.append(f.read())
    data = pd.DataFrame({"text": texts, "category": categories})
    return data
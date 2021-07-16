import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os, json

def generate_clean_df(all_files):
    cleaned_files = []
    for file in tqdm(all_files):
        features = [
            file['ID'],
            file['metadata']['Symptom']
        ]

        cleaned_files.append(features)

    col_names = ['ID', 'Symptom']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()

    return clean_df


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files

data_path = "C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/Symptom/"
files = load_files(data_path)
print("Loaded {} files".format(len(files)))
df = generate_clean_df(files)
df.head()
# dataset = pd.read_csv('C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/FrequencySymptoms.csv')
# dataset.head()

# X = dataset.iloc[:, [0]].values

tfidf_vectorizer = TfidfVectorizer( max_df=10, min_df=0.005, sublinear_tf=True)
data = df
print(data)
tfidf_matrix = tfidf_vectorizer.fit_transform(data.text)
dist = 1 - cosine_similarity(tfidf_matrix)
dist = dist - dist.min() # get rid of some pesky floating point errors that give neg. distance
linkage_matrix = ward(dist) # replace with complete, single, or other scipy.cluster.hierarchical algorithms


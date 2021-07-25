import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from argparse import ArgumentParser
import copy

def label_f0(f0_path, km_path, output_dir, n_clusters=12):
    with open(f0_path, "rb") as f1:
        f0 = pickle.load(f1)

    km_model = joblib.load(km_path)
    discrete_f0 = copy.deepcopy(f0)

    for i in tqdm(range(len(f0))):
        discrete_f0[i] = km_model.predict(np.array(f0[i]).reshape(-1, 1))
    
    for i in tqdm(range(len(f0))):
        for j in range(len(f0[i])):
            if f0[i][j] == 0:         
                discrete_f0[i][j] = n_clusters
    
    
    with open(os.path.join(output_dir, "pred_f0_label.km"), 'w') as f:
        for units in discrete_f0: 
            for i, unit in enumerate(units):
                f.write(str(int(unit)))
                if i < len(units)-1:
                    f.write(" ")
            f.write("\n")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--f0_path", required=True)
    PARSER.add_argument("-k", "--km_path", required=True)
    PARSER.add_argument("-o", "--output_dir", help="the directory to save trained kmeans model", required=True)
    PARSER.add_argument("--n_clusters", type=int, default=12, help="number of clusters for kmeans")
    label_f0(**vars(PARSER.parse_args()))
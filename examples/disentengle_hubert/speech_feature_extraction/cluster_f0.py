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

def learn_f0_kmeans(f0_path, voiced_flag_path, split, output_dir, visualize=False, n_clusters=12, batch_size=10000, max_no_improvement=100, max_iter=300, n_init=20):
    # with open(f0_path, "rb") as f1:
    #     f0 = pickle.load(f1)
    f0 = np.load(f0_path)
    voiced_flag = np.load(voiced_flag_path)
    assert len(f0) == len(voiced_flag), "the number of records is mismatch between f0 and voiced_flag"
    print(f" the number of records is {len(f0)}")
    
    tot_voiced_f0 = []
    for i in tqdm(range(len(f0))):
        for j in range(len(f0[i])):
            if voiced_flag[i][j]: 
                tot_voiced_f0.append(f0[i][j])

    km_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter, n_init=n_init)
    tot_voiced_f0 = np.array(tot_voiced_f0).reshape(-1, 1)
    print(tot_voiced_f0.shape)
    print("[INFO] start running kmeans")
    kmeans = km_model.fit(tot_voiced_f0)
    print("[INFO] finish running kmeans")
    print(sorted(km_model.cluster_centers_))
    # save kmeans model 
    km_path = os.path.join(output_dir, f"{split}_kmeans_f0.pkl")
    joblib.dump(km_model, km_path)

    # visualize f0 histogram
    if visualize: 
        plt.hist(tot_voiced_f0, bins=50)
        plt.savefig(os.path.join(output_dir, "kmeans_f0_result.jpg"))

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--f0_path", required=True)
    PARSER.add_argument("-f", "--voiced_flag_path", required=True)
    PARSER.add_argument("-s", "--split", help="dataset split name", required=True)
    PARSER.add_argument("-o", "--output_dir", help="the directory to save trained kmeans model", required=True)
    PARSER.add_argument("--visualize", action='store_true', help="whether to visualize kmeans result by tsne visualization")
    PARSER.add_argument("--n_clusters", type=int, default=12, help="number of clusters for kmeans")
    learn_f0_kmeans(**vars(PARSER.parse_args()))
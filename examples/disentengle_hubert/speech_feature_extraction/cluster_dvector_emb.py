import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

def learn_spk_kmeans(spk_emb_path, split, output_dir, visualize=False, n_clusters=12, batch_size=10000, max_no_improvement=100, max_iter=300, n_init=20):
    x_train = np.load(spk_emb_path)

    km_model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter,
                batch_size=batch_size, max_no_improvement=max_no_improvement, n_init=n_init)

    print("[INFO] start running kmeans")
    kmeans = km_model.fit(x_train)
    print("[INFO] finish running kmeans")
    # save kmeans model 
    km_path = os.path.join(output_dir, f"{split}_kmeans_spk.pkl")
    joblib.dump(km_model, km_path)

    if visualize: 
        tsne = TSNE(n_components=2, verbose=1)
        transformed = tsne.fit_transform(x_train)
        data = {
                "dim-1": transformed[:, 0],
                "dim-2": transformed[:, 1],
                "label": km_model.labels_,
            }
        sns.scatterplot(
            x="dim-1",
            y="dim-2",
            hue="label",
            palette=sns.color_palette(n_colors=n_clusters),
            data=data,
        )
        plt.legend(bbox_to_anchor=(1.04,0), loc="upper left", borderaxespad=0)
        plt.title(f'kmean clustering c = {n_clusters}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kmeans_spk_result.jpg"), bbox_inches="tight")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--spk_emb_path", required=True)
    PARSER.add_argument("-s", "--split", help="dataset split name", required=True)
    PARSER.add_argument("-o", "--output_dir", help="the directory to save trained kmeans model", required=True)
    PARSER.add_argument("--visualize", action='store_true', help="whether to visualize kmeans result by tsne visualization")
    PARSER.add_argument("--n_clusters", type=int, default=12, help="number of clusters for kmeans")
    PARSER.add_argument("--batch_size", default=10000, help="bsz of MiniBatchKMeans")
    learn_spk_kmeans(**vars(PARSER.parse_args()))
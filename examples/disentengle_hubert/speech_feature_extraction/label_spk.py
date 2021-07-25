from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from load_audio import load_audio
import torch
import torchaudio

def label_spk(spk_emb_path, km_path, output_dir, n_PCA=None):
    # load pretrained model
    km_model = joblib.load(km_path)
    spk_emb = np.load(spk_emb_path)
    spk_emb = np.transpose(spk_emb) # (D, num_file) -> (num_file, D)
    
    if n_PCA: 
    # do PCA first then kmeans
        pca = PCA(n_components=n_PCA)
        spk_emb = pca.fit_transform(spk_emb)
    preds = km_model.predict(spk_emb)
    with open(os.path.join(output_dir, "pred_spk_label.km"), 'w') as f:
        for pred in preds: 
            f.write(str(pred))
            f.write("\n")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-k", "--km_path", help="pretrained kmean model path", required=True)
    PARSER.add_argument("-s", "--spk_emb_path", help="pretrained kmean model path", required=True)
    PARSER.add_argument("-o", "--output_dir", help="save label as .km file", required=True)
    PARSER.add_argument("--n_PCA", type=int, default=128, help="PCA first then do kmeans")
    label_spk(**vars(PARSER.parse_args()))


from sklearn.cluster import MiniBatchKMeans
import numpy as np
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from load_audio import load_audio
import torch
import torchaudio

def label_spk(spk_emb_path, km_path, split, output_dir, n_PCA=None):
    # load pretrained model
    km_model = joblib.load(km_path)
    spk_emb = np.load(spk_emb_path)
    
    preds = km_model.predict(spk_emb)
    with open(os.path.join(output_dir, f"{split}_spk.km"), 'w') as f:
        for pred in preds: 
            f.write(str(pred))
            f.write("\n")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-k", "--km_path", help="pretrained kmean model path", required=True)
    PARSER.add_argument("-s", "--spk_emb_path", help="pretrained kmean model path", required=True)
    PARSER.add_argument("-s", "--split", help="dataset split name", required=True)
    PARSER.add_argument("-o", "--output_dir", help="save label as .km file", required=True)
    label_spk(**vars(PARSER.parse_args()))


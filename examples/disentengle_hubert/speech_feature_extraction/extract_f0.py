from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display
from presets import Preset
import librosa as _librosa
import os
from tqdm import tqdm 
from load_audio import load_audio
import pickle
import math

def extract_f0(manifest_path, output_dir, sr=16000, fmin=80, fmax=500):
    """
    sampling rate 16000 Hz, frame duration = 20 ms in Hubert
    to make f0 correspond to the length of Hubert representation
    """
    # reset librosa default sampling rate
    librosa = Preset(_librosa)
    librosa["sr"] = sr

    root, names, inds, tot, sizes = load_audio(manifest_path)
    # given file index to load every file in file_list
    f0_tot, voiced_flag_tot = [], []

    with open(os.path.join(output_dir, "f0.txt"), "wb") as f1:        
        with open(os.path.join(output_dir, "voiced_flag.txt"), "wb") as f2:        
            for ind in tqdm(range(tot), desc="extracting f0"):
                filename = os.path.join(root, names[ind])
                y, sr = librosa.load(filename)
                duration = len(y) / sr
                tgt_len = math.floor(duration / 0.020005)
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin, fmax, fill_na = 0.0, switch_prob=0.1, frame_length=320)
                f0_reduced = [f0[4 * i] for i in range(tgt_len)]
                voiced_flag = [voiced_flag[4 * i] for i in range(tgt_len)]
                f0_tot.append(f0)
                voiced_flag_tot.append(voiced_flag)
            pickle.dump(f0_tot, f1)
            pickle.dump(voiced_flag_tot, f2)

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--manifest_path", help=".tsv file that is created by wav2vec_manifest.py format", required=True)
    PARSER.add_argument("-o", "--output_dir", help="save extracted f0 and unvoiced to txt file", required=True)
    extract_f0(**vars(PARSER.parse_args()))   


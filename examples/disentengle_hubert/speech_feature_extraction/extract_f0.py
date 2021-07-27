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
from npy_append_array import NpyAppendArray

def extract_f0(manifest_path, split, output_dir, sr=16000, fmin=80, fmax=500):
    # reset librosa default sampling rate
    librosa = Preset(_librosa)
    librosa["sr"] = sr

    root, names, inds, tot, sizes = load_audio(manifest_path)

    # please make sure .npy file haven't existed already
    f0_path = os.path.join(output_dir, f"{split}_f0.npy")
    f0_f = NpyAppendArray(f0_path)
    voiced_flag_path = os.path.join(output_dir, f"{split}_voiced_flag.npy")
    voiced_flag_f = NpyAppendArray(voiced_flag_path)


    for ind in tqdm(range(tot), desc="extracting f0"):
        filename = os.path.join(root, names[ind])
        y, sr = librosa.load(filename)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin, fmax, fill_na = 0.0, switch_prob=0.1, frame_length=320)
        f0_f.append(f0)
        voiced_flag_f.append(voiced_flag)

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--manifest_path", help=".tsv file that is created by wav2vec_manifest.py format", required=True)
    PARSER.add_argument("-s", "--split", help="dataset split name", required=True)
    PARSER.add_argument("-o", "--output_dir", help="save extracted f0 and voice tag to npy file", required=True)
    extract_f0(**vars(PARSER.parse_args()))   


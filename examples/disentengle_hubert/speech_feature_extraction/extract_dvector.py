import torch
import torchaudio
from argparse import ArgumentParser
import os
from tqdm import tqdm 
import numpy as np
from load_audio import load_audio

def extract_dvector(manifest_path, wav2mel_path, checkpoint_path, output_path):
    """
    load pretrained d-vector model and extract d-vector for audio in manifest_path;
    in the end, save extracted speaker embedding to output_path.
    """
    root, names, inds, tot, sizes = load_audio(manifest_path)
    # load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav2mel = torch.jit.load(wav2mel_path)
    dvector = torch.jit.load(checkpoint_path).eval().to(device)
    # given file index to load every file in file_list
    for ind in tqdm(range(tot)):
        filename = os.path.join(root, names[ind])
        with torch.no_grad():
            wav_tensor, sample_rate = torchaudio.load(filename)
            mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
            emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)
            emb_tensor = emb_tensor.cpu().numpy()
            emb_tensor = np.expand_dims(emb_tensor, axis=1)
            if ind == 0: 
                output = emb_tensor
            else: 
                output = np.concatenate((output, emb_tensor), axis=1)

    np.save(output_path, output) # (256, num_file)

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-p", "--manifest_path", help=".tsv file that is created by wav2vec_manifest.py format", required=True)
    PARSER.add_argument("-w", "--wav2mel_path", required=True)
    PARSER.add_argument("-c", "--checkpoint_path", required=True)
    PARSER.add_argument("-o", "--output_path", help="save extracted speaker embedding as .npy file", required=True)
    extract_dvector(**vars(PARSER.parse_args()))

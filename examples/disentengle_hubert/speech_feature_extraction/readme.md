# Disentangled speech feature extraction 
dataset is in `.tsv` file, same as file format in wav2vec.

`*.tsv` files contains a list of audio, where each line is the root, and
following lines are the subpath for each audio:
```
<root-dir>
<audio-path-1>
<audio-path-2>
...
```
## Speaker information
### Speaker embedding Extraction
- d-vector 
    `extract_dvector.py` use pretrained model `dvector.pt` to extract d-vector.
    #### input argument
        - `-p`: dataset path in `.tsv`
        - `-w`: wav2mel model path in `.pt`
        - `-c`: dvector checkpoint path in `.pt`
        - `-o`: output path to save extracted speaker embedding in `.npy`

- i-vector (TODO)
### Speaker embedding clustering
    (TODO)
## Pitch information 

### Pitch Extraction 
`extract_f0.py` use `librosa.pyin` function to extract f0.
#### input argument
    - `-p`: dataset path in `.tsv`
    - `-o`: output directory to save extracted f0 and voice tag to `.txt` file
## Pitch clustering
    (TODO)


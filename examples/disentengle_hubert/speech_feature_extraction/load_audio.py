def load_audio(manifest_path):
    """ 
    description: similar loading function as in fairseq, to load tsv file 

    intput: wav2vec manifest format tsv file
    output: 
        root: root directory
        names: filename
        inds: 
        tot: number of files
        sizes: audio length
    """
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[1])
            names.append(items[0])
            inds.append(ind)
            sizes.append(sz)
    tot = ind + 1
    print(
            f"loaded {len(names)} audio files "
        )
    return root, names, inds, tot, sizes

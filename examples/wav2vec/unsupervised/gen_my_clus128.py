import torch
import numpy as np
import argparse
import _pickle as pk
import shutil
import os


"""
Metric definitions from: An improved speech segmentation quality measure: The R-value
http://users.spa.aalto.fi/orasanen/papers/IS09_r_value.pdf
"""

def read_pickle(path):
    return pk.load(open(path,'rb'))

def generate_one_new_kmeans_cluster_seq(origin_kmeans_cluster_seq, orc_utt):
    total_mfcc_num = orc_utt[-1]+1
    ratio = total_mfcc_num/len(origin_kmeans_cluster_seq)
    
    transformed_orc_utt_seg = [int(orc_seg/ratio) for orc_seg in orc_utt]
    monotonic_transformed_orc_utt_seg = [transformed_orc_utt_seg[0]]
    prev = transformed_orc_utt_seg[0]
    for i in range(1, len(transformed_orc_utt_seg)):
        if transformed_orc_utt_seg[i] > prev:
            monotonic_transformed_orc_utt_seg.append(transformed_orc_utt_seg[i])
            prev = transformed_orc_utt_seg[i]
    monotonic_transformed_orc_utt_seg[-1] = len(origin_kmeans_cluster_seq)

    new_kmeans_cluster_seq = [0]*len(origin_kmeans_cluster_seq)
    
    fill_in_int = 0
    for i in range(1, len(monotonic_transformed_orc_utt_seg)):
        for j in range(monotonic_transformed_orc_utt_seg[i-1], monotonic_transformed_orc_utt_seg[i]):
            new_kmeans_cluster_seq[j] = fill_in_int
        fill_in_int = (fill_in_int+1)%3

    return new_kmeans_cluster_seq
    
def generate_new_src_file(origin_src_path, origin_tsv_path, new_src_path, orc_utt_id2idx, orc):
    all_kmeans_cluster_lines = open(origin_src_path, 'r').read().splitlines()
    all_tsv_lines = open(origin_tsv_path, 'r').read().splitlines()
    all_kmeans_utt_id = [tsv_line.split('.wav')[0].split('/')[-1].lower() for tsv_line in all_tsv_lines[1:]]
    assert len(all_kmeans_utt_id) == len(all_kmeans_cluster_lines)

    all_new_kmeans_cluster_lines = []
    for i in range(len(all_kmeans_cluster_lines)):
        kmeans_utt_id = all_kmeans_utt_id[i]
        kmeans_cluster_seq = [int(x) for x in all_kmeans_cluster_lines[i].split(' ')]
        new_kmeans_cluster_seq = generate_one_new_kmeans_cluster_seq(kmeans_cluster_seq, orc[orc_utt_id2idx[kmeans_utt_id]])
        all_new_kmeans_cluster_lines.append(' '.join([str(x) for x in new_kmeans_cluster_seq]))
    with open(new_src_path, 'w') as f:
        for line in all_new_kmeans_cluster_lines:
            f.write(line)
            f.write('\n')

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_o', '--train_orc_boundary_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/nonmatch-data/timit_for_GAN/audio/timit-train-orc1-bnd.pkl', help='')
    parser.add_argument('-train_m', '--train_meta_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/nonmatch-data/timit_for_GAN/audio/timit-train-meta.pkl', help='')
    parser.add_argument('-test_o', '--test_orc_boundary_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/nonmatch-data/timit_for_GAN/audio/timit-test-orc1-bnd.pkl', help='')
    parser.add_argument('-test_m', '--test_meta_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/nonmatch-data/timit_for_GAN/audio/timit-test-meta.pkl', help='')
    parser.add_argument('-o', '--origin_clus128_dir', type=str, default='', help='')
    parser.add_argument('-n', '--new_clus128_dir', type=str, default='', help='')

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    os.makedirs(args.new_clus128_dir, exist_ok=True)

    orc = read_pickle(args.train_orc_boundary_path)
    meta = read_pickle(args.train_meta_path)
    assert len(orc) == len(meta['prefix'])

    orc_utt_id2idx = {}
    for i in range(len(orc)):
        utt_id = meta['prefix'][i]
        utt_id = utt_id.split('_')[1] + '_'+utt_id.split('_')[2]
        orc_utt_id2idx[utt_id] = i
    
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'train.tsv'),
        os.path.join(args.new_clus128_dir, 'train.tsv')
    )
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'train.phn'),
        os.path.join(args.new_clus128_dir, 'train.phn')
    )
    generate_new_src_file(
        os.path.join(args.origin_clus128_dir, 'train.src'), 
        os.path.join(args.origin_clus128_dir, 'train.tsv'), 
        os.path.join(args.new_clus128_dir, 'train.src'), 
        orc_utt_id2idx, orc
    )


    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'valid.tsv'),
        os.path.join(args.new_clus128_dir, 'valid.tsv')
    )
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'valid.phn'),
        os.path.join(args.new_clus128_dir, 'valid.phn')
    )
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'valid.src'),
        os.path.join(args.new_clus128_dir, 'valid.src')
    )

    orc = read_pickle(args.test_orc_boundary_path)
    meta = read_pickle(args.test_meta_path)
    assert len(orc) == len(meta['prefix'])

    orc_utt_id2idx = {}
    for i in range(len(orc)):
        utt_id = meta['prefix'][i]
        utt_id = utt_id.split('_')[1] + '_'+utt_id.split('_')[2]
        orc_utt_id2idx[utt_id] = i
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'test.tsv'),
        os.path.join(args.new_clus128_dir, 'test.tsv')
    )
    shutil.copyfile(
        os.path.join(args.origin_clus128_dir, 'test.phn'),
        os.path.join(args.new_clus128_dir, 'test.phn')
    )
    generate_new_src_file(
        os.path.join(args.origin_clus128_dir, 'test.src'), 
        os.path.join(args.origin_clus128_dir, 'test.tsv'), 
        os.path.join(args.new_clus128_dir, 'test.src'), 
        orc_utt_id2idx, orc
    )
import torch
import numpy as np
import argparse
import _pickle as pk

"""
Metric definitions from: An improved speech segmentation quality measure: The R-value
http://users.spa.aalto.fi/orasanen/papers/IS09_r_value.pdf
"""


def boundary2location(boundaries):
    maxlen = max([max(boundary) for boundary in boundaries]) + 1
    locations = torch.zeros(len(boundaries), maxlen).long()
    for bsxid, boundary in enumerate(boundaries):
        locations[bsxid, boundary] = 1
    return locations


def compute_alignment(predicts_loc, reals_loc, one2one=True, T=2):
    assert (len(predicts_loc) == len(reals_loc))
    hits = torch.zeros(len(predicts_loc)).long()
    detecteds = []
    for bsxid, (predict_loc, real_loc) in enumerate(zip(predicts_loc, reals_loc)):
        detected = []
        real_positions = real_loc.nonzero().squeeze()
        for idx in range(len(real_positions)):
            now_pos = real_positions[idx]
            start = max(now_pos - T, 0)
            end = now_pos + (T+1)
            if one2one:
                if idx > 0 and real_positions[idx - 1] + (T+2) >= start:
                    start = (real_positions[idx] + real_positions[idx - 1]) // 2
                if idx < len(real_positions) - 1 and real_positions[idx + 1] - (T+1) <= end:
                    end = (real_positions[idx] + real_positions[idx + 1]) // 2
            if predict_loc[start:end].sum() > 0:
                hits[bsxid] += 1
                detected.append(now_pos.item())
        detecteds.append(detected)
    return hits, detecteds


def compare_boundaries(predicts, reals, one2one=True, T=2):
    predicts_loc = boundary2location(predicts)
    reals_loc = boundary2location(reals)
    hits, detecteds = compute_alignment(predicts_loc, reals_loc, one2one)
    NC = hits.sum().float()
    NT = predicts_loc.sum().float()
    NG = reals_loc.sum().float()
    return torch.tensor([NC, NT, NG])


def compute_R(tensor):
    NC, NT, NG = tensor
    HR = (NC / NG) * 100.0
    OS = ((NT / NG) - 1) * 100.0
    r1 = np.sqrt(np.power((100.0 - HR), 2) + np.power(OS, 2))
    r2 = (-OS + HR - 100.0) / np.sqrt(2)
    R = 1 - (np.abs(r1) + np.abs(r2)) / 200.0
    return R.item()


def compute_F(tensor):
    NC, NT, NG = tensor
    PRC = NC / NT
    RCL = NC / NG
    F = (2 * PRC * RCL) / (PRC + RCL)
    return [F.item(), PRC.item(), RCL.item()]


def read_pickle(path):
    return pk.load(open(path,'rb'))

def gen_one_kmeans_seg(kmeans_cluster_seq, orc_utt, first_stage_merge_ratio=2):
    total_mfcc_num = orc_utt[-1]+1
    ratio = total_mfcc_num/len(kmeans_cluster_seq)
    
    kmeans_change_position = []
    prev_cluster=None
    for i, current_cluster in enumerate(kmeans_cluster_seq):
        if current_cluster != prev_cluster:
            kmeans_change_position.append(i)
            prev_cluster = current_cluster
    
    kmeans_seg = [int(position*ratio) for position in kmeans_change_position[::first_stage_merge_ratio]]
    if kmeans_seg[-1] > total_mfcc_num-1:
        kmeans_seg[-1] = total_mfcc_num-1
    elif kmeans_seg[-1] < total_mfcc_num-1:
        kmeans_seg.append(total_mfcc_num-1)
    
    return kmeans_seg
    

def addParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--orc_boundary_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/new-data/timit_for_GAN/audio/timit-train-orc1-bnd.pkl', help='')
    parser.add_argument('-m', '--meta_path', type=str, default='/home/darong/frequent_data/GAN_Harmonized_with_HMMs/new-data/timit_for_GAN/audio/timit-train-meta.pkl', help='')
    parser.add_argument('-k', '--kmeans_src_path', type=str, default='/home/darong/frequent_data/wav2vecu/timit_processed/matched/feat/CLUS128/train.src', help='')
    parser.add_argument('-t', '--kmeans_tsv_path', type=str, default='/home/darong/frequent_data/wav2vecu/timit_processed/matched/feat/CLUS128/train.tsv', help='')

    return parser

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()

    orc = read_pickle(args.orc_boundary_path)
    meta = read_pickle(args.meta_path)

    all_kmeans_cluster_lines = open(args.kmeans_src_path, 'r').read().splitlines()
    all_tsv_lines = open(args.kmeans_tsv_path, 'r').read().splitlines()

    utt_id2idx = {}
    for i, tsv_line in enumerate(all_tsv_lines[1:]):
        utt_id = tsv_line.split('.wav')[0].split('/')[-1].lower()
        utt_id2idx[utt_id] = i
    assert len(utt_id2idx) == len(all_kmeans_cluster_lines)

    all_kmeans_seg = []
    for i, utt in enumerate(orc):
        utt_id = meta['prefix'][i]
        utt_id = utt_id.split('_')[1] + '_'+utt_id.split('_')[2]
        kmeans_idx = utt_id2idx[utt_id]
        kmeans_cluster_seq = [int(x) for x in all_kmeans_cluster_lines[kmeans_idx].split(' ')]
        kmeans_seg = gen_one_kmeans_seg(kmeans_cluster_seq, utt)
        all_kmeans_seg.append(kmeans_seg)
    
    all_kmeans_seg = np.array(all_kmeans_seg, dtype=object)

    stats = compare_boundaries(orc, all_kmeans_seg)
    #stats = compare_boundaries(all_kmeans_seg, orc)
    R = compute_R(stats)
    F, precision, recall = compute_F(stats)

    print(f'uns: R: {R}, F: {F}, PRC: {precision}, RCL: {recall}')


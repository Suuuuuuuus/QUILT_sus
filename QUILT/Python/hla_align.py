import io
import os
import re
import sys
import csv
import gzip
import time
import json
import pickle
import math
import secrets
import multiprocessing
import subprocess
import resource
import pandas as pd
import numpy as np
import pywfa
sys.path.append('/well/band/users/rbx225/software/lcwgsus/')
import lcwgsus
from lcwgsus.variables import *
from hla_align_functions import *

def hla_aligner(gene, bam, db, hla_gene_information, 
                reads_apart_max=1000,
                temperature=100,
                n_mismatches=5,
                assumed_bq=0.001):
    reads1 = get_chr6_reads(gene, bam, hla_gene_information, reads_apart_max)
    reads2 = get_hla_reads(gene, bam, reads_apart_max)

    if reads1.empty:
        reads1 = reads2.iloc[:2, :] if not reads2.empty else pd.DataFrame()
    elif reads2.empty:
        reads2 = reads1.iloc[:2, :]
    else:
        pass
    
    rl = reads1['sequence'].str.len().mode().values[0]

    likemat1 = calculate_loglikelihood(reads1, db)
    likemat2 = calculate_loglikelihood(reads2, db)
    min_valid_prob = np.log(math.comb(rl, n_mismatches) * (assumed_bq**n_mismatches) * ((1 - assumed_bq)**(rl - n_mismatches)))

    valid_indices1 = np.any(likemat1 >= min_valid_prob, axis=1)
    valid_indices2 = np.any(likemat2 >= min_valid_prob, axis=1)
    likemat1, reads1 = likemat1[valid_indices1], reads1[valid_indices1]
    likemat2, reads2 = likemat2[valid_indices2], reads2[valid_indices2]
    
    likemat_all = np.vstack((likemat1, likemat2))

    id1, id2 = reads1.iloc[:, 0].to_numpy(), reads2.iloc[:, 0].to_numpy()
    
    readind = (reads1.iloc[:, 1].astype(int) // 64) % 4
    readind2 = (reads2.iloc[:, 1].astype(int) // 64) % 4
    mate_indicator = np.concatenate((readind, readind2))

    ids_all = np.concatenate((id1, id2))
    unique_ids = np.unique(ids_all)
    likemat_mate = np.zeros((len(unique_ids), likemat_all.shape[1]))

    for i, uid in enumerate(unique_ids):
        t1 = likemat_all[ids_all == uid, :]
        t2 = mate_indicator[ids_all == uid]
        if len(t2) > 0:
            likemat_mate[i, :] = np.sum(t1[t2 > 0], axis=0)

    valid_mask = likemat_mate.max(axis=1) >= min_valid_prob
    likemat_mate = likemat_mate[valid_mask]
    likemat_norm = 0.5 * np.exp(likemat_mate - likemat_mate.max(axis=1, keepdims=True)) + 1e-100

    likemat_paired = likemat_norm.T @ likemat_norm
    likemat_paired = pd.DataFrame(likemat_paired, index=db.columns, columns=db.columns)
    
    likemat_mate = pd.DataFrame(likemat_mate, index = unique_ids[valid_mask], columns=db.columns)
    return reads1, reads2, likemat_mate, likemat_paired

def main(gene, bam, db, hla_gene_information, outdir):
    r1, r2, mate, pair = hla_aligner(gene, bam, db, hla_gene_information)
    
    r1.to_csv(outdir + '/reads1.csv', header = False, index = False)
    r2.to_csv(outdir + '/reads2.csv', header = False, index = False)
    
    pd.set_option('display.float_format', '{:.6e}'.format)
    mate.to_csv(outdir + '/mate_likelihood_matrix.ssv', index=True, header=True, sep = ' ')
    pair.to_csv(outdir + '/pair_likelihood_matrix.ssv', index=True, header=True, sep = ' ')
    
if __name__ == "__main__":
    gene = sys.argv[1]
    bam = sys.argv[2]
    outdir = sys.argv[3]
    
    hla_gene_information = pd.read_csv('/well/band/users/rbx225/software/QUILT_sus/hla_ancillary_files/hla_gene_information.tsv', sep = ' ')
    db = pd.read_csv(f'/well/band/users/rbx225/recyclable_files/hla_reference_files/v3570_aligners/{gene}.ssv', sep = ' ')
    main(gene, bam, db, hla_gene_information, outdir)
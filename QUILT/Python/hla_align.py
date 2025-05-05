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
home_dir = os.environ.get("home_dir")
sys.path.append(f'{home_dir}software/lcwgsus/')
sys.path.append(f'{home_dir}software/QUILT_sus/QUILT/Python/')
import lcwgsus
from lcwgsus.variables import *
from hla_align_functions import *
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

def main(gene, bam, db_dir, as_dir, hla_gene_information, outdir, reads_df_outdir):
    r1, r2, mate, pair = hla_aligner(gene, bam, db_dir, as_dir, hla_gene_information)
    r1.to_csv(outdir + '/reads1.csv', header = False, index = False)
    r2.to_csv(outdir + '/reads2.csv', header = False, index = False)
    mate.to_csv(outdir + '/mate_likelihood_matrix.ssv',  index=True, header=True, sep = ' ')
    pair.to_csv(outdir + '/pair_likelihood_matrix.ssv',  index=True, header=True, sep = ' ')
    
    sample = bam.split('/')[-1].split('.')[0]

    bestalleles = get_best_alleles(pair)
    bestalleles.columns = ['bestallele1', 'bestallele2', 'post_prob', 'sums']
    bestalleles['sample_number'] = 1
    bestalleles['sample_name'] = sample
    bestalleles = bestalleles[['sample_number', 'sample_name', 'bestallele1', 'bestallele2', 'post_prob', 'sums']]

    bestalleles.to_csv(f'{reads_df_outdir}/{sample}/{gene}/quilt.hla.output.onlyreads.all.txt', index = False, header = True, sep = '\t')
    bestalleles.iloc[[0], :].to_csv(f'{reads_df_outdir}/{sample}/{gene}/quilt.hla.output.onlyreads.topresult.txt', index = False, header = True, sep = '\t')

if __name__ == "__main__":
    gene = sys.argv[1]
    bam = sys.argv[2]
    outdir = sys.argv[3]
    reads_df_outdir = sys.argv[4]

    if ('3570' in reads_df_outdir) or ('optimal' in reads_df_outdir):
        version = '3570'
    else:
        version = '3390'
    
    hla_gene_information = pd.read_csv(HLA_GENE_INFORMATION_FILE, sep = ' ')
    db_dir = f'/well/band/users/rbx225/recyclable_files/hla_reference_files/v{version}_merged_only/'
    as_dir = f'/well/band/users/rbx225/GAMCC/results/hla/imputation/WFA_alignments/v{version}/'
    main(gene, bam, db_dir, as_dir, hla_gene_information, outdir, reads_df_outdir)

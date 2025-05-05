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
from IPython.display import display_html
sys.path.append('/well/band/users/rbx225/software/lcwgsus/')
import lcwgsus
from lcwgsus.variables import *
from collections import Counter
from itertools import combinations_with_replacement
from sklearn.cluster import KMeans
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.preprocessing import minmax_scale

def hla_aligner(gene, bam, db_dir, as_dir, hla_gene_information, 
                reads_apart_max = 1000,
                reads_extend_max = 1000,
                n_mismatches = 5,
                assumed_bq = 0.001,
                score_diff_in_alignment_genes = 8,
                n_unique_onefield = None,
                phase = False):
    reads1 = get_chr6_reads(gene, bam, hla_gene_information, strict = True,
                            reads_apart_max = reads_apart_max, 
                            reads_extend_max = reads_extend_max)
    
    rl = reads1['sequence'].str.len().mode().values[0]
    sample = bam.split('/')[-1].split('.')[0]

    reads2 = get_hla_reads(gene, bam, reads_extend_max = reads_extend_max)

    if reads1.empty:
        reads1 = reads2.iloc[:2, :] if not reads2.empty else pd.DataFrame()
    elif reads2.empty:
        reads2 = reads1.iloc[:2, :]
    else:
        pass

    likemat = pd.read_csv(f"{as_dir}{sample}/{gene}/AS_matrix.ssv", sep = ' ', index_col = 0)
    likemat1 = likemat.values
    aligned_genes = likemat.columns.str.split('*').str.get(0).unique().tolist()
    
    columns = []
    for g in aligned_genes:
        db = pd.read_csv(f'{db_dir}{g}.ssv', sep = ' ')
        columns = columns + db.columns.tolist()
    columns = np.array(columns)
    columns_to_keep = np.where(np.char.startswith(columns, gene + '*'))[0]
    target_alleles = columns[columns_to_keep]
    
    min_valid_prob = n_mismatches*np.log(assumed_bq) + (rl - n_mismatches)*np.log(1 - assumed_bq)
#     min_valid_prob = -20
    valid_indices1 = np.any(likemat1 >= min_valid_prob, axis=1)
    likemat1, reads1 = likemat1[valid_indices1], reads1[valid_indices1]
    likemat_all = likemat1
    id1 = reads1.iloc[:, 0].to_numpy()

    unique_ids = np.unique(id1)
    likemat_mate = -600*np.ones((len(unique_ids), likemat_all.shape[1]))

    for i, uid in enumerate(unique_ids):
        tmp = likemat_all[id1 == uid, :]

        keep = False
        unique = True
        for j in range(tmp.shape[0]):
            best_indices = np.where(tmp[j,:] == tmp[j,:].max())[0]
            best_alleles = columns[best_indices]
            aligned_genes = np.unique(np.array([s.split('*')[0] for s in best_alleles]))
            keep = keep or ((len(aligned_genes) == 1) and (aligned_genes[0] == gene))
            unique = unique and (gene in aligned_genes)

        if (keep and unique):
            likemat_mate[i, :] = np.sum(tmp, axis=0)

   # Filtering out based on alignment scores
    scoring_df = pd.DataFrame({'ID': unique_ids, 
                        'Target': likemat_mate[:,columns_to_keep].max(axis = 1), 
                        'Others': likemat_mate[:,~np.where(np.char.startswith(columns, gene + '*'))[0]].max(axis = 1)
                       })
    scoring_df['diff'] = scoring_df['Target'] - scoring_df['Others']
    valid_mask = ((scoring_df['diff'] > score_diff_in_alignment_genes) & (scoring_df['Target'] >= min_valid_prob)).tolist()
    
    if len(np.where(valid_mask)[0]) == 0:
        likemat_mate = np.zeros((len(valid_mask), len(target_alleles)))
        likemat_mate_df = pd.DataFrame(likemat_mate, index = unique_ids, columns=target_alleles)
        likemat_paired_df=pd.DataFrame(0, index=target_alleles, columns=target_alleles)                       
        return reads1, reads2, likemat_mate_df, likemat_paired_df

    likemat_mate = likemat_mate[valid_mask, :][:, columns_to_keep]
    unique_ids = unique_ids[valid_mask]
    reads1 = reads1[reads1['ID'].isin(unique_ids)]
    likemat_mate_df = pd.DataFrame(likemat_mate, index = unique_ids, columns=target_alleles)
    
    if n_unique_onefield is not None:
        df = likemat_mate_df.copy()
        best_alignments = df.max(axis=1)
        matching_cols = df.eq(best_alignments, axis=0).apply(lambda row: likemat_mate_df.columns[row].tolist(), axis=1)
        one_field = matching_cols.apply(lambda cols: list(set(col.split(':')[0] for col in cols)))
        counts = one_field.apply(len)
        rows_to_remove = counts[counts >= n_unique_onefield].index
        rows_to_keep = df.index.difference(rows_to_remove)

        if len(rows_to_remove) == len(df):
            id1 = reads1.iloc[:, 0].to_numpy()
            unique_ids = np.unique(id1)
            likemat_mate = np.zeros((len(unique_ids), len(target_alleles)))
            likemat_mate_df = pd.DataFrame(likemat_mate, index = unique_ids, columns=target_alleles)
            likemat_pair=pd.DataFrame(0, index=target_alleles, columns=target_alleles)                       
            return reads1, reads2, likemat_mate_df, likemat_paired_df  
                                
        likemat_mate_df = likemat_mate_df.loc[rows_to_keep]
        reads1 = reads1[reads1['ID'].isin(rows_to_keep)]
        likemat_mate = likemat_mate_df.values
    
    if phase and (likemat_mate.shape[0] > 2):
        likemat_mate_normalised = normalize(np.exp(likemat_mate - likemat_mate.max(axis = 1, keepdims = True)), axis=1, norm='l1')
        gmm = GaussianMixture(n_components=2, tol = 1e-10, max_iter = 1000, n_init = 50)
        gmm.fit(likemat_mate_normalised)
        labels = gmm.predict(likemat_mate_normalised)

        group1 = np.where(labels == 0)[0]
        group2 = np.where(labels == 1)[0]

        l1 = likemat_mate[group1, :].sum(axis=0)
        l2 = likemat_mate[group2, :].sum(axis=0)

        likemat_mate = np.vstack([l1, l2])

#     likemat_paired_df = compute_pair_likelihoods(likemat_mate_df)
    n_reads, n_alleles = likemat_mate.shape
    likemat_pair = pd.DataFrame(0, index=target_alleles, columns=target_alleles, dtype=float)

    for i in range(n_reads):
        likemat_norm = likemat_mate[i, :] - logsumexp(likemat_mate[i, :])
        likelihood_probs = np.exp(likemat_norm)
        m1 = np.add.outer(likelihood_probs, likelihood_probs) / 2 
        likemat_pair += np.log(m1)
        
    likemat_paired_df = pd.DataFrame(likemat_pair, index=target_alleles, columns=target_alleles)
#     n = likemat_paired_df.shape[0]
#     mat = np.full((n, n), np.log(2))
#     np.fill_diagonal(mat, 0)
#     likemat_paired_df = likemat_paired_df - mat
    
    likemat_paired_df = deresolute_pair_max(likemat_paired_df)
    return reads1, reads2, likemat_mate_df, likemat_paired_df

def compute_pair_likelihoods(mate):
    scores = mate.values
    n_reads, n_alleles = scores.shape
    alleles = np.arange(n_alleles)
    matrix = np.zeros((n_alleles, n_alleles))

    for i, j in combinations_with_replacement(alleles, 2):
        a1 = scores[:, i]
        a2 = scores[:, j]
        
        max_a = np.maximum(a1, a2)
        ll = max_a + np.log(0.5 * np.exp(a1 - max_a) + 0.5 * np.exp(a2 - max_a))
        
        if i == j:
            matrix[i, j] = ll.sum()
        else:
            matrix[i, j] = ll.sum()
            matrix[j, i] = ll.sum()
        
    return pd.DataFrame(matrix, index = mate.columns, columns = mate.columns)

def get_chr6_reads(gene, bam, hla_gene_information, strict = True, reads_apart_max = 1000, reads_extend_max = 1000):
    regstart = hla_gene_information[hla_gene_information['Name'] == f'HLA-{gene}']['Start'].values[0]
    regend = hla_gene_information[hla_gene_information['Name'] == f'HLA-{gene}']['End'].values[0]

    command = f"samtools view {bam} chr6:{regstart-reads_extend_max}-{regend+reads_extend_max}"
    reads = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')

    reads = [i.split('\t') for i in reads if '##' not in i]
    reads = pd.DataFrame(reads)
    reads.columns = [
        "ID", "flag", "chr", "pos", "map_quality", "CIGAR", "chr_alt", "pos_alt", "insert_size", "sequence", "base_quality"
    ] + [f'col{i}' for i in range(11, reads.shape[1])]

    reads['pos'] = reads['pos'].astype(int)
    reads['pos_alt'] = reads['pos_alt'].astype(int)
    
    mode = reads['sequence'].str.len().mode().values[0]

    reads = reads[
        (reads['chr'] == 'chr6') &
        (reads['chr_alt'].isin(['chr6', '='])) &
        (reads['pos_alt'] + reads['sequence'].str.len() >= np.ones(len(reads))*(regstart - reads_extend_max)) & 
        (reads['pos_alt'] <= np.ones(len(reads))*(regend + reads_extend_max)) &
        (reads['sequence'].str.len() == mode)
    ]
    
    def check_alternative_mapping(r):
        alt_map_str = r['col11']
        invalid_chrs = [f'chr{i}' for i in list(range(1,6)) + list(range(7,23)) + ['X', 'Y', 'MT']]
        if 'chr' not in alt_map_str:
            return r
        else:
            alt_map_str = alt_map_str.split(':')[-1]
            for mapping in alt_map_str.split(';'):
                if mapping != '':
                    components = mapping.split(',')
                    if components[0] in invalid_chrs:
                        r['col11'] = 'REMOVE'
                        return r
                    pos = abs(int(components[1]))
                    if (pos > regend + reads_extend_max) or (pos < regstart - reads_extend_max):
                        r['col11'] = 'REMOVE'
                        return r
            return r
    if strict:
        reads = reads.apply(check_alternative_mapping, axis = 1)
    if reads_apart_max is not None:
        reads = reads[abs(reads['pos'] - reads['pos_alt']) <= reads_apart_max]
    reads = reads[reads['col11'] != 'REMOVE'].reset_index(drop = True)
    
    return reads

def get_hla_reads(gene, bam, strict = True, reads_extend_max = 1000):
    command = f"samtools view -H {bam} | grep HLA-{gene}"
    header = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')
    if header[0] == '':
        return pd.DataFrame()
    else:
        header = [i.split('\t') for i in header]
        header = pd.DataFrame(header)
        contigs = ' '.join(header[1].str.split('SN:').str.get(1).tolist())

        command = f"samtools view {bam} {contigs}"
        reads = subprocess.run(command, shell = True, capture_output = True, text = True).stdout[:-1].split('\n')

        reads = [i.split('\t') for i in reads if '##' not in i]
        reads = pd.DataFrame(reads)
        reads.columns = [
            "ID", "flag", "chr", "pos", "map_quality", "CIGAR", "chr_alt", "pos_alt", "insert_size", "sequence", "base_quality"
        ] + [f'col{i}' for i in range(11, reads.shape[1])]

        reads['pos'] = reads['pos'].astype(int)
        reads['pos_alt'] = reads['pos_alt'].astype(int)
        valid_contigs = contigs.split(' ') + ['chr6', '=']

        mode = reads['sequence'].str.len().mode().values[0]

        reads = reads[
            (reads['chr'].isin(valid_contigs)) &
            (reads['chr_alt'].isin(valid_contigs)) &
            (reads['pos_alt'] + reads['sequence'].str.len() >= np.ones(len(reads))*(regstart - reads_extend_max)) & 
            (reads['pos_alt'] <= np.ones(len(reads))*(regend + reads_extend_max)) &
            (reads['sequence'].str.len() == mode)
        ]
        
        def check_alternative_mapping(r):
            alt_map_str = r['col11']
            invalid_chrs = [f'chr{i}' for i in list(range(1,6)) + list(range(7,23)) + ['X', 'Y', 'MT']]
            if 'chr' not in alt_map_str:
                return r
            else:
                alt_map_str = alt_map_str.split(':')[-1]
                for mapping in alt_map_str.split(';'):
                    if mapping != '':
                        components = mapping.split(',')
                        if components[0] in invalid_chrs:
                            r['col11'] = 'REMOVE'
                            return r
                        pos = abs(int(components[1]))
                        if (pos > regend + reads_extend_max) or (pos < regstart - reads_extend_max):
                            r['col11'] = 'REMOVE'
                            return r
                return r
        if strict:
            reads = reads.apply(check_alternative_mapping, axis = 1)
        reads = reads[reads['col11'] != 'REMOVE'].reset_index(drop = True)

        return reads

def get_alignment_scores(hla_gene_information_file, gene, bam, db_dir, ofile, strict = True,
                         reads_apart_max = 1000, reads_extend_max = 1000):
    hla_gene_information = pd.read_csv(hla_gene_information_file, sep = ' ')
    
    reads1 = get_chr6_reads(gene, bam, hla_gene_information, 
                    strict = strict,
                    reads_apart_max = reads_apart_max, 
                    reads_extend_max = reads_extend_max)

    rl = reads1['sequence'].str.len().mode().values[0]
    ncores = max(2*(len(os.sched_getaffinity(0))) - 1, 1)

    db_dict = {}
    for g in HLA_GENES_ALL_EXPANDED:
        db = pd.read_csv(f'{db_dir}{g}.ssv', sep = ' ')
        for c in db.columns:
            refseq = ''.join(db[c].tolist()).replace('.', '').lstrip('*').rstrip('*')
            db_dict[c] = refseq

    columns = np.array(list(db_dict.keys()))
    likemat1 = multi_calculate_loglikelihood_per_allele(reads1, db_dict, ncores)

    alignment_scores_raw = pd.DataFrame(likemat1, index=reads1['ID'].tolist(), columns=columns)
    alignment_scores_raw.to_csv(ofile, float_format='%.6f', sep = ' ', header = True, index = True)
    return None
    
def multi_calculate_loglikelihood_per_allele(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], len(db)))
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_allele_alignment,
            [(ix, a, refseq, reads) for ix, (a, refseq) in enumerate(db.items())]
        )

    for res in results:
        ix, score = res
        scores_mat[:, ix] = score
        
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    return scores_mat 

def per_allele_alignment(ix, a, refseq, reads, min_aligned_bases = 10):
    ref = pywfa.WavefrontAligner(refseq)
    
    scores_ary = []
    rev_scores_ary = []
    
    for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
        result = ref(seq, clip_cigar=True, min_aligned_bases_left=min_aligned_bases, min_aligned_bases_right=min_aligned_bases)
        if result.pattern_start > result.pattern_end:
            cigar_lst = None
            refstart, refend, readstart, readend = 0,0,0,0
        else:
            cigar_lst = result.cigartuples
            refstart, refend, readstart, readend = result.pattern_start, result.pattern_end, result.text_start, result.text_end
        
        newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
        likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
        scores_ary.append(likelihood_per_read_per_allele)
            
    for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
        result = ref(seq, clip_cigar=True, min_aligned_bases_left=min_aligned_bases, min_aligned_bases_right=min_aligned_bases)
        if result.pattern_start > result.pattern_end:
            cigar_lst = None
            refstart, refend, readstart, readend = 0,0,0,0
        else:
            cigar_lst = result.cigartuples
            refstart, refend, readstart, readend = result.pattern_start, result.pattern_end, result.text_start, result.text_end
    
        newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
        likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
        rev_scores_ary.append(likelihood_per_read_per_allele)
        
    scores_ary_final = np.maximum(np.array(scores_ary), np.array(rev_scores_ary))

    return ix, scores_ary_final

def calculate_score_per_alignment(seq, refseq, bq, minscore = -600, cg = -6, cm = 0, cx = -4, ce = -2):
    seq = list(seq)
    refseq = list(refseq)
    bq = phred_to_scores(bq)
    
    score = 0
    start_gap = True
    
    for i, (b1, b2, q) in enumerate(zip(seq, refseq, bq)):
        if (b1 == '.') and (b2 == '.'):
            pass
        elif (b1 == '.') or (b2 == '.'):
            if start_gap:
                score += cg
                start_gap = False
            else:
                score += ce
        else:
            start_gap = True
            if (b2 == '*') or (b1 == 'N'):
                p_match = 0.25
                score += cm*p_match + cx*(1-p_match)
            elif b1 == b2:
                p_match = (1-10**(-0.1*q))
                score += cm*p_match + cx*(1-p_match)
            elif b1 != b2:
                p_match = 10**(-0.1*q)/3
                score += cm*p_match + cx*(1-p_match)
            else:
                pass
    if minscore is not None:
        return max(minscore, score) 
    else:
        return score

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def phred_to_scores(bq):
    return [ord(char) - 33 for char in bq]

def find_best_allele_per_clade_two_field(db):
    colnames = [':'.join(col.split(':')[:2]) for col in db.columns]
    res = {}
    for i, c in enumerate(colnames):
        if c in res.keys():
            res[c].append(i)
        else:
            res[c] = [i]
    result = {}

    for group, indices in res.items():
        sub_df = db.iloc[:, indices]
        colref = sub_df.isin(['*']).sum().idxmin()
        result[group] = colref
    return result

def recode_sequences(refseq, seq, bq, cigars_lst, refstart = 0, refend = 0, readstart = 0, readend = 0, rl = 151):
    if cigars_lst is None:
        return refseq[:rl], seq[:rl], bq[:rl]
    n_bases = 0
    for code, length in cigars_lst:
        if code != 2:
            n_bases += length
    if n_bases == rl:
        if cigars_lst[0][0] == 4:
            refstart = max(refstart - cigars_lst[0][1], 0)
            readstart = max(readstart - cigars_lst[0][1], 0)
        if cigars_lst[-1][0] == 4:
            refend = min(refend + cigars_lst[-1][1], len(refseq))
            readend = min(readend + cigars_lst[-1][1], rl)
        refseq = refseq[refstart:refend]
        seq = seq[readstart:readend]
    else:
        trim_start = 0
        trim_end = 0

        if cigars_lst[0][0] == 2:
            trim_start = cigars_lst[0][1]
            cigars_lst = cigars_lst[1:]
        if cigars_lst[-1][0] == 2:
            trim_end = cigars_lst[-1][1]
            cigars_lst = cigars_lst[:-1]

        refseq = refseq[trim_start:(len(refseq) - trim_end)]
    newseq = ''
    newrefseq = ''
    newbq = ''
    refseq_index = 0
    seq_index = 0
    for code, length in cigars_lst:
        if code == 1:
            newrefseq += '.' * length
            newseq += seq[seq_index:(seq_index + length)]
            newbq += bq[seq_index:(seq_index + length)]
            seq_index += length
        elif code == 2:
            newseq += '.' * length
            newbq += 'f' * length
            newrefseq += refseq[refseq_index:(refseq_index + length)]
            refseq_index += length
        else:
            newseq += seq[seq_index:(seq_index + length)]
            newbq += bq[seq_index:(seq_index + length)]
            newrefseq += refseq[refseq_index:(refseq_index + length)]
            seq_index += length
            refseq_index += length
    return newrefseq, newseq, newbq
    
def find_best_pair_alignment(df):
    logllmax = df.max().max()
    row_index, col_index = np.where(df == logllmax)
    a1, a2 = df.index[row_index], df.columns[col_index]
    return a1, a2

def find_best_hap_alignment(gene, db, l):
    alleles_g_code, _ = find_best_allele_per_g_group(gene, db)
    data = pd.DataFrame({'Allele': alleles_g_code, 'Score': l})
    data = data[data['Allele'] != 'rare']
    data['4Digit'] = data['Allele'].apply(lambda x: ':'.join(x.split(':')[:2])) 
    grouped = data.groupby('4Digit', as_index=False).agg({'Score': 'max'})
    max_score = grouped['Score'].max()
    best_alleles = grouped[grouped['Score'] == max_score]['4Digit'].tolist()

    if len(best_alleles) == 1:
        result = best_alleles[0]
    else:
        two_digit_groups = {}
        for allele in best_alleles:
            two_digit = allele.split(':')[0]
            if two_digit not in two_digit_groups:
                two_digit_groups[two_digit] = []
            two_digit_groups[two_digit].append(allele)

        if len(two_digit_groups) == 1:
            result = '/'.join(best_alleles)
        else:
            result = ' '.join(best_alleles)
    return result.split(' ')[0].split('/')[0] # A hack: if there is a tie take the first

def softmax_1d(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def softmax_2d(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def process_db_genfile(gene, 
                            ipd_gen_file_dir, 
                            hla_gene_information,
                            impute = False):
    strand = hla_gene_information[hla_gene_information['Name'] == ('HLA-' + gene)]['Strand'].iloc[0]
    nucleotides = ['A', 'T', 'C', 'G']

    ipd_gen_file = ipd_gen_file_dir + gene + '_gen.txt'
    with open(ipd_gen_file, "r") as file:
        lines = file.readlines()
    gDNA_idx = []
    names = []
    i = 0
    while len(gDNA_idx) < 2:
        l = lines[i]
        if 'gDNA' in l:
            gDNA_idx.append(i)
        elif l.lstrip(' ').split(' ')[0].startswith(gene + '*'):
            name = l.lstrip(' ').split(' ')[0]
            names.append(name)
        i += 1

    first_base = int(lines[gDNA_idx[0]].split(' ')[-1].split('\n')[0])
    n_alleles = gDNA_idx[1] - gDNA_idx[0] - 3

    alleles_dict = {k:'' for k in names}
    for i, s in enumerate(lines):
        r = s.lstrip(' ')
        if r.startswith(f'{gene}*'):
            r = r.rstrip(' \n')
            name = r.split(' ')[0]
            sequence = r.split(' ')[1:]
            sequence = ''.join(sequence)
            alleles_dict[name] = alleles_dict[name] + sequence

    db = pd.DataFrame({key: list(value) for key, value in alleles_dict.items()}).T
    db = db.drop(columns=db.columns[db.eq('|').all()])
    db.columns = range(db.shape[1])

    db = db.apply(lambda c: c.replace('-', c[0]) ,axis = 0)

    if strand != 1:
        db.replace({'A': 't', 'C': 'g', 'G': 'c', 'T': 'a'}, inplace=True)
        db.replace({'a': 'A', 'c': 'C', 'g': 'G', 't': 'T'}, inplace=True)
        db = db.iloc[:, ::-1]

    db.columns = range(db.shape[1])
    db = db.T
    
    if impute:
        low_missing = ((db == '*').mean(axis=1) < 0.1)

        for i, a in enumerate(db.columns):
            db_undetermined = ((db[a] == '*') & (low_missing))
            subseted_db =  db[~db_undetermined] 
            allele = subseted_db[a].to_numpy()
            _, others = subseted_db[a].align(subseted_db.drop(columns=a), axis=0)
            counts = (others != allele[:, np.newaxis]).sum(axis=0).sort_values()
            closest_allele_idx = 0
            substituted_nucleotides = np.array(['']*np.sum(db_undetermined))
            substituted_indices = db_undetermined[db_undetermined == True].index

            while '' in substituted_nucleotides:
                closest_allele = counts.index[closest_allele_idx]

                target_haplotype = db.loc[substituted_indices, closest_allele]
                substitutable = target_haplotype[target_haplotype.values != '*']
                substituted_nucleotides[np.where(np.in1d(substituted_indices, substitutable.index))[0]] = substitutable.values
                closest_allele_idx += 1

            db.loc[substituted_indices, a] = substituted_nucleotides
    return db

def get_best_alleles(df, thresh: float = 0.99):
    alleles = df.index
    n = len(alleles)
    
    upper_indices = np.triu_indices(n)
    log_probs = df.values[upper_indices]

    max_log = np.max(log_probs)
    probs = np.exp(log_probs - max_log)
    probs /= probs.sum()

    allele1 = [alleles[i] for i in upper_indices[0]]
    allele2 = [alleles[j] for j in upper_indices[1]]
    result = pd.DataFrame({
        "allele1": allele1,
        "allele2": allele2,
        "prob": probs
    })

    result = result.sort_values("prob", ascending=False).reset_index(drop=True)
    result["cum"] = result["prob"].cumsum()
    row2 = np.min(np.where(result["cum"].values >= thresh)) + 1
    result = result.iloc[:row2,:]
    return result

def deresolute_pair(pair, two_field_ary = None):
    if two_field_ary is None:
        two_field_ary = lcwgsus.extract_unique_twofield(pair.columns)
        
    tt = pair.index.tolist()
    colons = np.array([name.count(":") for name in tt])
    keep = np.array(tt)[colons >= 1]
    overall2 = pair.loc[keep, keep].copy()

    overall2 = overall2 - overall2.max().max()
    overall2 = np.exp(overall2)
    
    keep2 = np.array([":".join(name.split(":")[:2]) for name in keep])
    vv = pd.Series(keep2).value_counts()

    weights = np.array([vv[k] for k in keep2])
    overall2 = overall2.div(weights, axis=0)
    overall2 = overall2.div(weights, axis=1)
    overall2 /= overall2.sum().sum()
    
    readscaledlikelihoodmat = overall2

    fourdigitsseen = keep2
    cond = np.isin(fourdigitsseen, two_field_ary)
    intersectreadscaledlikelihoodmat = readscaledlikelihoodmat.loc[cond, cond]
    intersectreadscaledlikelihoodmat /= intersectreadscaledlikelihoodmat.sum().sum()
    
    keep3 = keep2[cond]
    vv2 = pd.Series(keep3).value_counts()
    
    fourdigitreadscaledlikelihoodmat = pd.DataFrame(0, index=vv.index, columns=vv.index)
    rows = [vv.index.get_loc(k) for k in keep2]
    cols = rows.copy()
    
    for i in range(len(keep2)):
        for j in range(len(keep2)):
            fourdigitreadscaledlikelihoodmat.iloc[rows[i], cols[j]] += readscaledlikelihoodmat.iloc[i, j]
    fourdigitreadscaledlikelihoodmat = fourdigitreadscaledlikelihoodmat.loc[two_field_ary, two_field_ary]
    return fourdigitreadscaledlikelihoodmat

def deresolute_pair_max(df):
    columns = df.columns
    two_field_ary = lcwgsus.extract_unique_twofield(columns)
    result = pd.DataFrame(0, index = two_field_ary, columns = two_field_ary)
    for a1 in two_field_ary:
        for a2 in two_field_ary:
            ridx = np.where(columns.str.startswith(a1))[0]
            cidx = np.where(columns.str.startswith(a2))[0]
            tmp = df.iloc[ridx, cidx]
            score = tmp.max().max()
            result.loc[a1, a2] = score
    return result

def filter_raw_as_matrix(mate, reads1, columns, gene, n_mismatches = 5, assumed_bq = 0.001, rl = 151, score_diff_in_alignment_genes = 0):
    mate = mate.copy()
    reads1 = reads1.copy()
    
    columns_to_keep = np.where(np.char.startswith(columns, gene + '*'))[0]
    target_alleles = columns[columns_to_keep]
    min_valid_prob = n_mismatches*np.log(assumed_bq) + (rl - n_mismatches)*np.log(1 - assumed_bq)

    valid_indices1 = np.any(mate >= min_valid_prob, axis=1)
    mate, reads1 = mate[valid_indices1], reads1[valid_indices1]
    id1 = reads1.iloc[:, 0].to_numpy()

    unique_ids = np.unique(id1)
    likemat_mate = -600*np.ones((len(unique_ids), mate.shape[1]))

    for i, uid in enumerate(unique_ids):
        tmp = mate[id1 == uid, :]

        keep = False
        unique = True
        for j in range(tmp.shape[0]):
            best_indices = np.where(tmp[j,:] == tmp[j,:].max())[0]
            best_alleles = columns[best_indices]
            aligned_genes = np.unique(np.array([s.split('*')[0] for s in best_alleles]))
            keep = keep or ((len(aligned_genes) == 1) and (aligned_genes[0] == gene))
            unique = unique and (gene in aligned_genes)

        if (keep and unique):
            likemat_mate[i, :] = np.sum(tmp, axis=0)
            
    scoring_df = pd.DataFrame({'ID': unique_ids, 
                        'Target': likemat_mate[:,columns_to_keep].max(axis = 1), 
                        'Others': likemat_mate[:,~np.where(np.char.startswith(columns, gene + '*'))[0]].max(axis = 1)
                       })
    scoring_df['diff'] = scoring_df['Target'] - scoring_df['Others']
    valid_mask = ((scoring_df['diff'] > score_diff_in_alignment_genes) & (scoring_df['Target'] >= min_valid_prob)).tolist()
            
    likemat_mate = likemat_mate[valid_mask,:][:, columns_to_keep]
    unique_ids = unique_ids[valid_mask]
    reads1 = reads1[reads1['ID'].isin(unique_ids)]
    likemat_mate_df = pd.DataFrame(likemat_mate, index = unique_ids, columns=target_alleles)
    return likemat_mate_df, reads1

def get_n_aligned_mates_method(g, s, as_dir, columns):
    bam = f'data/bams/{s}.bam'
    hla_gene_information = pd.read_csv(HLA_GENE_INFORMATION_FILE, sep = ' ')
    reads1 = get_chr6_reads(g, bam, hla_gene_information, reads_apart_max = 1000)
    mate = pd.read_csv(f"{as_dir}{s}/{g}/AS_matrix.ssv", sep = ' ', index_col = 0).values
    mate, reads1 = filter_raw_as_matrix(mate, reads1, columns, g)
    return mate

def get_n_aligned_mates_quilt(g, s, indir, columns):
    ifile = f'{indir}{s}/{g}/extracted.hla{g}.RData'
    data = pyreadr.read_r(ifile)
    if len(data) != 0:
        qmate = data['qmate']
        qmate = qmate.loc[(qmate.index != ''), :]

        columns_to_keep = np.where(np.char.startswith(columns, g + '*'))[0]
        target_alleles = columns[columns_to_keep]
        qmate.columns = target_alleles
    else:
        return pd.DataFrame()
    return qmate

def count_correct_aligned_reads(mate, a1, a2):
    if len(mate) == 0:
        return 0
    else:
        row_max = mate.max(axis=1)
        result = mate.eq(row_max, axis=0)
        indices = [lcwgsus.extract_unique_twofield(list(row[row].index)) for _, row in result.iterrows()]
        n_true = np.array([(a1 in i) or (a2 in i) for i in indices]).sum()
        return n_true

def get_columns_from_indir(db_dir, genes):
    columns = []
    for g in genes:
        db = pd.read_csv(f'{db_dir}{g}.ssv', sep = ' ')
        columns = columns + db.columns.tolist()
    return np.array(columns)
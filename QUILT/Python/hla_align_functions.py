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

# For now I'm assuming reads come to same lengths 
# Or maybe we could calculate per-alignment score divided by read length
# Otherwise unfair for longer reads because they are more prone to errors

def get_chr6_reads(gene, bam, hla_gene_information, reads_apart_max = 1000):
    regstart = hla_gene_information[hla_gene_information['Name'] == f'HLA-{gene}']['Start'].values[0]
    regend = hla_gene_information[hla_gene_information['Name'] == f'HLA-{gene}']['End'].values[0]

    command = f"samtools view {bam} chr6:{regstart-reads_apart_max}-{regend+reads_apart_max}"
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
        (reads['pos_alt'] + reads['sequence'].str.len() >= np.ones(len(reads))*(regstart - reads_apart_max)) & 
        (reads['pos_alt'] <= np.ones(len(reads))*(regend + reads_apart_max)) &
        (reads['sequence'].str.len() == mode)
    ]

    id_counts = reads['ID'].value_counts()
    valid_ids_ary = id_counts[id_counts == 2].index.tolist()
    reads = reads[reads['ID'].isin(valid_ids_ary)].sort_values(by = 'ID').reset_index(drop = True)
    return reads

def get_hla_reads(gene, bam, reads_apart_max = 1000):
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
            (reads['pos_alt'] + reads['sequence'].str.len() >= np.ones(len(reads))*(regstart - reads_apart_max)) & 
            (reads['pos_alt'] <= np.ones(len(reads))*(regend + reads_apart_max)) &
            (reads['sequence'].str.len() == mode)
        ]

        id_counts = reads['ID'].value_counts()
        valid_ids_ary = id_counts[id_counts == 2].index.tolist()
        reads = reads[reads['ID'].isin(valid_ids_ary)].sort_values(by = 'ID').reset_index(drop = True)
        return reads 
    
def per_allele(j, a, db, reads, q):
    refseq = (''.join(db[a].tolist())).replace('.', '')
    ref = pywfa.WavefrontAligner(refseq)
    scores_mat_ary1 = []
    scores_mat_ary2 = []
    for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
        result = ref(seq)
        refseq_aligned = refseq[result.pattern_start:result.pattern_end]
        seq_aligned = seq[result.text_start:result.text_end]
        likelihood_per_read_per_allele1 = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
        scores_mat_ary1.append(likelihood_per_read_per_allele1)
    for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
        result = ref(seq)
        refseq_aligned = refseq[result.pattern_start:result.pattern_end]
        seq_aligned = seq[result.text_start:result.text_end]
        likelihood_per_read_per_allele2 = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
        scores_mat_ary2.append(likelihood_per_read_per_allele2)
    
    q.put([j, np.array(scores_mat_ary1), np.array(scores_mat_ary2)])
    
def multi_calculate_loglikelihood_per_allele(reads, db, temperature=1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    manager = multiprocessing.Manager()
    q = manager.Queue()
    processes = []
    
    for j, a in enumerate(db.columns):
        tmp = multiprocessing.Process(target=per_allele,
                                          args=(j, a, db, reads, q))
        tmp.start()
        processes.append(tmp)

    for process in processes:
        process.join()
    res_lst = []
    while not q.empty():
        res_lst.append(q.get())
    else:
        for res in res_lst:
            j = res[0]
            scores_mat[:, j] = res[1]
            rev_scores_mat[:, j] = res[2]

    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    likelihood_mat = np.exp(scores_mat/temperature)/np.sum(np.exp(scores_mat/temperature), axis = 1, keepdims = True)
    loglikelihood_mat = np.log(likelihood_mat)
    return loglikelihood_mat   

def process_db_genfile(gene, 
                            ipd_gen_file_dir, 
                            hla_gene_information):
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
        if r.startswith(gene):
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

def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))

def phred_to_scores(bq):
    return [ord(char) - 33 for char in bq]

'''
def calculate_loglikelihood(reads, db, temperature = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    for j, a in enumerate(db.columns):
        refseq = (''.join(db[a].tolist())).replace('.', '')
        ref = pywfa.WavefrontAligner(refseq)
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            result = ref(seq)
            refseq_aligned = refseq[result.pattern_start:result.pattern_end]
            seq_aligned = seq[result.text_start:result.text_end]
#             cigars = result.cigartuples
#             score = result.score
#             cigars, score = adjust_align_score(cigars, score)
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            result = ref(seq)
            refseq_aligned = refseq[result.pattern_start:result.pattern_end]
            seq_aligned = seq[result.text_start:result.text_end]
#             cigars = result.cigartuples
#             score = result.score
#             cigars, score = adjust_align_score(cigars, score)
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele

    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    likelihood_mat = np.exp(scores_mat/temperature)/np.sum(np.exp(scores_mat/temperature), axis = 1, keepdims = True)
    loglikelihood_mat = np.log(likelihood_mat)
    return loglikelihood_mat   

def adjust_align_score(cigartuples, score, gap_opening = 6, gap_extension = 2):
    trim_start_indicator = 0
    trim_end_indicator = 0
    trim_start = 0
    trim_end = 0
    
    if cigartuples[0][0] == 2:
        trim_start = cigartuples[0][1]
        trim_start_indicator = 1
        cigartuples = cigartuples[1:]
    if cigartuples[-1][0] == 2:
        trim_end = cigartuples[-1][1]
        trim_end_indicator = 1
        cigartuples = cigartuples[:-1]
    newscore = score + gap_opening*(trim_start_indicator + trim_end_indicator) + gap_extension*(trim_start + trim_end)
    return cigartuples, newscore

def recode_sequences(refseq, seq, cigars_lst):
    newseq = []
    newrefseq = []
    index = 0
    for code, length in cigars_lst:
        if code == 1:
            newrefseq += ['-' * length]
            newseq += [seq[index:index + length]]
        elif code == 2:
            newseq += ['-' * length]
            newrefseq += [refseq[index:index + length]]
        else:
            newseq += [seq[index:index + length]]
            newrefseq += [refseq[index:index + length]]
        index += length
    return "".join(newrefseq), "".join(newseq)
'''

def calculate_score_per_alignment(seq, refseq, bq, cg = -6, cm = 0, cx = -4, ce = -2):
    seq = list(seq)
    refseq = list(refseq)
    bq = phred_to_scores(bq)
    
    score = 0
    gap_ix = -1
    
    for i, (b1, b2, q) in enumerate(zip(seq, refseq, bq)):
        if (b1 == '-') or (b2 == '-'):
            if gap_ix != i:
                score += cg
            gap_ix = i+1
            score += ce
        elif b2 == '*':
            pass
        elif b1 == b2:
            p_match = (1-10**(-0.1*q))
            score += cm*p_match + cx*(1-p_match)
        elif b1 != b2:
            p_match = 10**(-0.1*q)/3
            score += cm*p_match + cx*(1-p_match)
        else:
            pass
    return score  
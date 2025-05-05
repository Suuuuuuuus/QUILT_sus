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
sys.path.append('/well/band/users/rbx225/software/QUILT_test/QUILT/Python/')
import lcwgsus
from lcwgsus.variables import *
from hla_align_functions import *
from scipy.special import logsumexp

def deresolute_db_alleles(df, n_mismatch = 5):
    colnames = [col.split(':')[0] + ':' + col.split(':')[1] for col in df.columns]
    res = {}
    for i, c in enumerate(colnames):
        if c in res.keys():
            res[c].append(i)
        else:
            res[c] = [i]
    result = {}

    for group, indices in res.items():
        sub_df = df.iloc[:, indices]
        if sub_df.shape[1] == 1:
            pass
        else:
            colref = sub_df.isin(['*']).sum().idxmin()
            colrefidx = np.where(df.columns == colref)[0][0]
            for i in indices:
                col = df.columns[i]
                differences = ((sub_df[col] != sub_df[colref]) & (sub_df[col] != '*') & (sub_df[colref] != '*'))
                if (differences.sum() <= n_mismatch) and col != colref:
                    result[col] = colrefidx
    return result

def calculate_loglikelihood(reads, db, temperature = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    deresolute_alleles = deresolute_db_alleles(db)
    
    for j, a in enumerate(db.columns):
        if a not in deresolute_alleles.keys():
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
    for j, a in enumerate(db.columns):
        if a in deresolute_alleles.keys():
            scores_mat[:, j] = scores_mat[:, deresolute_alleles[a]]
            rev_scores_mat[:, j] = rev_scores_mat[:, deresolute_alleles[a]]
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
        elif (b2 == '*') or (b1 == 'N'):
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
    return score  


def multi_calculate_loglikelihood_per_allele(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))

    deresolute_alleles = deresolute_db_alleles(db)

    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_allele,
            [(j, a, db, reads) for j, a in enumerate(db.columns) if a not in deresolute_alleles.keys()]
        )

    for res in filter(None, results):
        j, scores, rev_scores = res
        scores_mat[:, j] = scores
        rev_scores_mat[:, j] = rev_scores

    for j, a in enumerate(db.columns):
        if a in deresolute_alleles.keys():
            scores_mat[:, j] = scores_mat[:, deresolute_alleles[a]]
            rev_scores_mat[:, j] = rev_scores_mat[:, deresolute_alleles[a]]

    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    likelihood_mat = np.exp(scores_mat)/np.sum(np.exp(scores_mat), axis = 1, keepdims = True)
    loglikelihood_mat = np.log(likelihood_mat)
    return loglikelihood_mat

def multi_calculate_loglikelihood_msa(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade(db)
    plus_positions = {}
    minus_positions = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, pos_df, rev_pos_df = res
        plus_positions[g] = pos_df
        minus_positions[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = a.split(':')[0]
        refseq = (''.join(db[a].tolist())).replace('.', '')
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            refstart, refend, readstart, readend = plus_positions[group].loc[i,:]
            refseq_aligned = refseq[refstart:refend]
            seq_aligned = seq[readstart:readend]
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            refstart, refend, readstart, readend = minus_positions[group].loc[i,:]
            refseq_aligned = refseq[refstart:refend]
            seq_aligned = seq[readstart:readend]
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    likelihood_mat = np.exp(scores_mat)/np.sum(np.exp(scores_mat), axis = 1, keepdims = True)
    loglikelihood_mat = np.log(likelihood_mat)
    return loglikelihood_mat 

def per_allele(j, a, db, reads, q = None):
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

    res = [j, np.array(scores_mat_ary1), np.array(scores_mat_ary2)]
    if q is None:
        return res
    else:
        q.put(res)

def per_group_alignment(g, a, db, reads):
    refseq = (''.join(db[a].tolist())).replace('.', '')
    ref = pywfa.WavefrontAligner(refseq)
    pos_df = pd.DataFrame(columns = ['refstart', 'refend', 'readstart', 'readend'])
    for seq in reads['sequence']:
        result = ref(seq)
        pos_df.loc[len(pos_df)] = [result.pattern_start, result.pattern_end, result.text_start, result.text_end]
        
    rev_pos_df = pd.DataFrame(columns = ['refstart', 'refend', 'readstart', 'readend'])
    for seq in reads['rev_seq']:
        result = ref(seq)
        rev_pos_df.loc[len(rev_pos_df)] = [result.pattern_start, result.pattern_end, result.text_start, result.text_end]
    return [g, pos_df, rev_pos_df]

def multi_calculate_loglikelihood_msa1(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade(db)
    plus_positions = {}
    minus_positions = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, pos_df, rev_pos_df = res
        plus_positions[g] = pos_df
        minus_positions[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = a.split(':')[0]
        refseq = (''.join(db[a].tolist())).replace('.', '')
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            refstart, refend, readstart, readend = plus_positions[group].loc[i,:]
            refseq_aligned = refseq[refstart:refend]
            seq_aligned = seq[readstart:readend]
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            refstart, refend, readstart, readend = minus_positions[group].loc[i,:]
            refseq_aligned = refseq[refstart:refend]
            seq_aligned = seq[readstart:readend]
            likelihood_per_read_per_allele = calculate_score_per_alignment(seq_aligned, refseq_aligned, bq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat 

###
def per_group_alignment(g, a, db, reads):
    refseq = (''.join(db[a].tolist()))
    ref = pywfa.WavefrontAligner(refseq)
    plus_tuple_lsts = []
    minus_tuple_lsts = []
    for seq in reads['sequence']:
        result = ref(seq)
        plus_tuple_lsts.append(result.cigartuples)
        
    for seq in reads['rev_seq']:
        result = ref(seq)
        minus_tuple_lsts.append(result.cigartuples)
    return [g, plus_tuple_lsts, minus_tuple_lsts]
def multi_calculate_loglikelihood_msa(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade(db)
    plus_positions = {}
    minus_positions = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, plus_tuple_lsts, minus_tuple_lsts = res
        plus_positions[g] = plus_tuple_lsts
        minus_positions[g] = minus_tuple_lsts
    
    for j, a in enumerate(db.columns):
        group = a.split(':')[0]
        refseq = (''.join(db[a].tolist()))
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            cigar_lst = plus_positions[group][i]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            cigar_lst = minus_positions[group][i]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat 

def recode_sequences(refseq, seq, bq, cigars_lst):
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
###

###
def adjust_clipping(score, cigars_lst, cg = -6, ce = -2, cx = -4):
    if (cigars_lst[0][0] == 4) or (cigars_lst[0][0] == 5):
        length = cigars_lst[0][1]
        score = score + cg + ce*length
    if (cigars_lst[-1][0] == 4) or (cigars_lst[-1][0] == 5):
        length = cigars_lst[-1][1]
        score = score + cg + ce*length
    return score   
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
        elif code == 0:
            newseq += seq[seq_index:(seq_index + length)]
            newbq += bq[seq_index:(seq_index + length)]
            newrefseq += refseq[refseq_index:(refseq_index + length)]
            seq_index += length
            refseq_index += length
        else:
            pass
    return newrefseq, newseq, newbq
###

def hla_aligner(gene, bam, db, hla_gene_information, 
                reads_extend_max=0,
                n_mismatches=5,
                assumed_bq=0.001):
    reads1 = get_chr6_reads(gene, bam, hla_gene_information, reads_apart_max = 1000, reads_extend_max = 0)
    reads2 = get_hla_reads(gene, bam, reads_extend_max = 0)

    if reads1.empty:
        reads1 = reads2.iloc[:2, :] if not reads2.empty else pd.DataFrame()
    elif reads2.empty:
        reads2 = reads1.iloc[:2, :]
    else:
        pass
    
    rl = reads1['sequence'].str.len().mode().values[0]
    ncores = 2*(len(os.sched_getaffinity(0))) - 1

    likemat1 = multi_calculate_loglikelihood_msa(reads1, db, ncores)
    likemat2 = multi_calculate_loglikelihood_msa(reads2, db, ncores)
    
#     min_valid_prob = np.log(math.comb(rl, n_mismatches)) + n_mismatches*np.log(assumed_bq) + (rl - n_mismatches)*np.log(1 - assumed_bq)
#     min_valid_prob = n_mismatches*np.log(assumed_bq/3)
    min_valid_prob = n_mismatches*np.log(assumed_bq) + (rl - n_mismatches)*np.log(1 - assumed_bq)

    valid_indices1 = np.any(likemat1 >= min_valid_prob, axis=1)
#     valid_indices2 = np.any(likemat2 >= min_valid_prob, axis=1)
    likemat1, reads1 = likemat1[valid_indices1], reads1[valid_indices1]
#     likemat2, reads2 = likemat2[valid_indices2], reads2[valid_indices2]
    
#     likemat_all = np.vstack((likemat1, likemat2))
    likemat_all = likemat1

#     id1, id2 = reads1.iloc[:, 0].to_numpy(), reads2.iloc[:, 0].to_numpy()
    id1 = reads1.iloc[:, 0].to_numpy()
    
    readind = (reads1.iloc[:, 1].astype(int) // 64) % 4
#     readind2 = (reads2.iloc[:, 1].astype(int) // 64) % 4
#     mate_indicator = np.concatenate((readind, readind2))
    mate_indicator = readind

#     ids_all = np.concatenate((id1, id2))
    ids_all = id1
    unique_ids = np.unique(ids_all)
    likemat_mate = np.zeros((len(unique_ids), likemat_all.shape[1]))

    for i, uid in enumerate(unique_ids):
        t1 = likemat_all[ids_all == uid, :]
        t2 = mate_indicator[ids_all == uid]
        if len(t2) > 0:
            likemat_mate[i, :] = np.sum(t1[t2 > 0], axis=0)

    valid_mask = likemat_mate.max(axis=1) >= min_valid_prob
    likemat_mate = likemat_mate[valid_mask,:]
    likemat_mate_df = pd.DataFrame(likemat_mate, index = unique_ids[valid_mask], columns=db.columns)

    if likemat_mate.shape[0] > 2:
        likemat_mate_normalised = normalize(likemat_mate, axis=1, norm='l1')
        gmm = GaussianMixture(n_components=2, tol = 1e-10, max_iter = 10000, n_init = 50)
        gmm.fit(likemat_mate_normalised)
        labels = gmm.predict(likemat_mate_normalised)
        
        group1 = np.where(labels == 0)[0]
        group2 = np.where(labels == 1)[0]

        l1 = likemat_mate[group1, :].mean(axis=0)
        l2 = likemat_mate[group2, :].mean(axis=0)

        likemat_pair = np.ones((len(db.columns),len(db.columns)))
        likemat_pair *= softmax_1d(l1)[None,:]
        likemat_pair = (likemat_pair.T * softmax_1d(l2)).T
        likemat_pair = np.log(0.5 * (likemat_pair + likemat_pair.T) + 1e-100)
    
    else:
        likemat_norm=likemat_mate-likemat_mate.max(axis = 1, keepdims = True)
        likemat_norm=0.5*np.exp(likemat_norm)+1e-100

        likemat_pair=pd.DataFrame(0, index=db.columns, columns=db.columns)
        qq=likemat_pair*0
        for i in range(likemat_mate.shape[0]):
            qq=qq*0
            m1=qq+likemat_norm[i,:]
            m1=m1+m1.T

            likemat_pair=likemat_pair+np.log(m1)
        
    likemat_paired_df = pd.DataFrame(likemat_pair, index=db.columns, columns=db.columns)
    return reads1, reads2, likemat_mate_df, likemat_paired_df

def main(gene, bam, db, hla_gene_information, outdir, reads_df_outdir):
    print('+++Begin HLA aligner+++')
    r1, r2, mate, pair = hla_aligner(gene, bam, db, hla_gene_information)
    print('+++Saving+++')
    r1.to_csv(outdir + '/reads1.csv', header = False, index = False)
    r2.to_csv(outdir + '/reads2.csv', header = False, index = False)
    
    pd.set_option('display.float_format', '{:.6e}'.format)
    mate.to_csv(outdir + '/mate_likelihood_matrix.ssv', index=True, header=True, sep = ' ')
    pair.to_csv(outdir + '/pair_likelihood_matrix.ssv', index=True, header=True, sep = ' ')
    print('+++End HLA aligner+++')
    
    sample = bam.split('/')[-1].split('.')[0]
    logllmax = pair.max().max()
    row_index, col_index = np.where(pair == logllmax)
    a1, a2 = pair.index[row_index[0]], pair.columns[col_index[0]]

    if a1.count(':') > 1:
        a1 = ':'.join(a1.split(':')[:2])
    if a2.count(':') > 1:
        a2 = ':'.join(a2.split(':')[:2])

    result = pd.DataFrame({'sample_number': 1,
                          'sample_name': sample,
                          'bestallele1': a1,
                          'bestallele2': a2,
                          'post_prob': logllmax,
                          'sums': 0}, index = ['0'])
    result.to_csv(f'{reads_df_outdir}/{sample}/{gene}/quilt.hla.output.onlyreads.topresult.txt', index = False, header = True, sep = '\t')

## G group
    
def multi_calculate_loglikelihood_msa(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    gene = db.columns[0].split('*')[0]
    
    alleles_g_code_ary, best_alleles_g_dict = find_best_allele_per_g_group(gene, db) # Can further optimise this by preparing in advance
    plus_tuple_dict = {}
    minus_tuple_dict = {}
    plus_position_dict = {}
    minus_position_dict = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles_g_dict.items()]
        )
    
    for res in results:
        g, plus_tuple_lsts, pos_df, minus_tuple_lsts, rev_pos_df = res
        plus_tuple_dict[g] = plus_tuple_lsts
        minus_tuple_dict[g] = minus_tuple_lsts
        plus_position_dict[g] = pos_df
        minus_position_dict[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = alleles_g_code_ary[j]
        if group != 'rare':
            refseq = (''.join(db[a].tolist()))
            for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
                cigar_lst = plus_tuple_dict[group][i]
                refstart, refend, readstart, readend = plus_position_dict[group].loc[i,:]
                newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
                likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
                scores_mat[i, j] = likelihood_per_read_per_allele
            for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
                cigar_lst = minus_tuple_dict[group][i]
                refstart, refend, readstart, readend = minus_position_dict[group].loc[i,:]
                newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
                likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
                rev_scores_mat[i, j] = likelihood_per_read_per_allele
        else:
            scores_mat[:, j] = -600
            rev_scores_mat[:, j] = -600
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat     
    
def find_best_allele_per_g_group(gene, db):
    g_code = pd.read_csv(AMBIGUOUS_G_CODE_FILE, sep = '\t')
    alleles = db.columns.str.split('*').str.get(1).tolist()
    two_field = [":".join(a.split(':', 2)[:2]) for a in alleles]
    
    tmp = g_code[g_code['Locus'] == gene]
    res_dict = {}
    for g, t in zip(tmp['G code'], tmp['Two field']):
        alleles_ary = t.split('/')
        for a in alleles_ary:
            res_dict[a] = g

    alleles_g_code = []
    
    for a in two_field:
        if a in res_dict.keys():
            alleles_g_code.append(res_dict[a])
        else:
            alleles_g_code.append('rare')

    best_alleles_g = {}
    for g in np.unique(np.array(alleles_g_code)):
        indices = np.where(np.array(alleles_g_code) == g)[0]
        sub_df = db.iloc[:, indices]
        colref = sub_df.isin(['*']).sum().idxmin()
        best_alleles_g[g] = colref
    return alleles_g_code, best_alleles_g    

## Current best

def find_best_allele_per_clade(db):
    colnames = [col.split(':')[0] for col in db.columns]
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

def multi_calculate_loglikelihood_msa(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade(db)
    plus_tuple_dict = {}
    minus_tuple_dict = {}
    plus_position_dict = {}
    minus_position_dict = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, plus_tuple_lsts, pos_df, minus_tuple_lsts, rev_pos_df = res
        plus_tuple_dict[g] = plus_tuple_lsts
        minus_tuple_dict[g] = minus_tuple_lsts
        plus_position_dict[g] = pos_df
        minus_position_dict[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = a.split(':')[0]
        refseq = (''.join(db[a].tolist()))
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            cigar_lst = plus_tuple_dict[group][i]
            refstart, refend, readstart, readend = plus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            cigar_lst = minus_tuple_dict[group][i]
            refstart, refend, readstart, readend = minus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat 

sample = bam.split('/')[-1].split('.')[0]
logllmax = pair.max().max()
row_index, col_index = np.where(pair == logllmax)
a1, a2 = pair.index[row_index[0]], pair.columns[col_index[0]]

if a1.count(':') > 1:
    a1 = ':'.join(a1.split(':')[:2])
if a2.count(':') > 1:
    a2 = ':'.join(a2.split(':')[:2])

result = pd.DataFrame({'sample_number': 1,
                      'sample_name': sample,
                      'bestallele1': a1,
                      'bestallele2': a2,
                      'post_prob': logllmax,
                      'sums': 0}, index = ['0'])


def find_best_allele_per_clade_one_field(db):
    colnames = [col.split(':')[0] for col in db.columns]
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

def multi_calculate_loglikelihood_msa_one_field(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade_one_field(db)
    plus_tuple_dict = {}
    minus_tuple_dict = {}
    plus_position_dict = {}
    minus_position_dict = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, plus_tuple_lsts, pos_df, minus_tuple_lsts, rev_pos_df = res
        plus_tuple_dict[g] = plus_tuple_lsts
        minus_tuple_dict[g] = minus_tuple_lsts
        plus_position_dict[g] = pos_df
        minus_position_dict[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = a.split(':')[0]
        refseq = (''.join(db[a].tolist()))
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            cigar_lst = plus_tuple_dict[group][i]
            refstart, refend, readstart, readend = plus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            cigar_lst = minus_tuple_dict[group][i]
            refstart, refend, readstart, readend = minus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat 


def multi_calculate_loglikelihood_per_allele(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_allele_alignment,
            [(j, a, db, reads) for j, a in enumerate(db.columns)]
        )

    for res in results:
        j, scores_ary_final = res
        scores_mat[:,j] = scores_ary_final
        
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    return scores_mat 

def per_allele_alignment(j, a, db, reads, min_aligned_bases = 10):
    refseq = ''.join(db[a].tolist()).replace('.', '').lstrip('*').rstrip('*')
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

    return j, scores_ary_final

def multi_calculate_loglikelihood_msa_two_field(reads, db, ncores = 1):
    reads['rev_seq'] = reads['sequence'].apply(reverse_complement)
    reads['rev_bq'] = reads['base_quality'].apply(lambda bq: bq[::-1])

    scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    rev_scores_mat = np.zeros((reads.shape[0], db.shape[1]))
    
    best_alleles = find_best_allele_per_clade_two_field(db)
    plus_tuple_dict = {}
    minus_tuple_dict = {}
    plus_position_dict = {}
    minus_position_dict = {}
    
    with multiprocessing.Pool(processes=ncores) as pool:
        results = pool.starmap(
            per_group_alignment,
            [(g, a, db, reads) for g, a in best_alleles.items()]
        )
    
    for res in results:
        g, plus_tuple_lsts, pos_df, minus_tuple_lsts, rev_pos_df = res
        plus_tuple_dict[g] = plus_tuple_lsts
        minus_tuple_dict[g] = minus_tuple_lsts
        plus_position_dict[g] = pos_df
        minus_position_dict[g] = rev_pos_df
    
    for j, a in enumerate(db.columns):
        group = ':'.join(a.split(':')[:2])
        refseq = (''.join(db[a].tolist()))
        for i, (seq, bq) in enumerate(zip(reads['sequence'], reads['base_quality'])):
            cigar_lst = plus_tuple_dict[group][i]
            refstart, refend, readstart, readend = plus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            scores_mat[i, j] = likelihood_per_read_per_allele
        for i, (seq, bq) in enumerate(zip(reads['rev_seq'], reads['rev_bq'])):
            cigar_lst = minus_tuple_dict[group][i]
            refstart, refend, readstart, readend = minus_position_dict[group].loc[i,:]
            newrefseq, newseq, newbq = recode_sequences(refseq, seq, bq, cigar_lst, refstart, refend, readstart, readend)
            likelihood_per_read_per_allele = calculate_score_per_alignment(newseq, newrefseq, newbq)
            rev_scores_mat[i, j] = likelihood_per_read_per_allele
    reads = reads.drop(columns = ['rev_seq', 'rev_bq'])
    scores_mat = np.maximum(scores_mat, rev_scores_mat)
    return scores_mat 

def per_group_alignment(g, a, db, reads, min_aligned_bases = 10):
    refseq = (''.join(db[a].tolist()))
    ref = pywfa.WavefrontAligner(refseq)
    pos_df = pd.DataFrame(columns = ['refstart', 'refend', 'readstart', 'readend'])
    plus_tuple_lsts = []
    for seq in reads['sequence']:
        result = ref(seq, clip_cigar=True, min_aligned_bases_left=min_aligned_bases, min_aligned_bases_right=min_aligned_bases)
        if result.pattern_start > result.pattern_end:
            plus_tuple_lsts.append(None)
            pos_df.loc[len(pos_df)] = [0,0,0,0]
        else:
            plus_tuple_lsts.append(result.cigartuples)
            pos_df.loc[len(pos_df)] = [result.pattern_start, result.pattern_end, result.text_start, result.text_end]
        
    rev_pos_df = pd.DataFrame(columns = ['refstart', 'refend', 'readstart', 'readend'])
    minus_tuple_lsts = []
    for seq in reads['rev_seq']:
        result = ref(seq, clip_cigar=True, min_aligned_bases_left=min_aligned_bases, min_aligned_bases_right=min_aligned_bases)
        if result.pattern_start > result.pattern_end:
            minus_tuple_lsts.append(None)
            rev_pos_df.loc[len(rev_pos_df)] = [0,0,0,0]
        else:
            minus_tuple_lsts.append(result.cigartuples)
            rev_pos_df.loc[len(rev_pos_df)] = [result.pattern_start, result.pattern_end, result.text_start, result.text_end]
    return [g, plus_tuple_lsts, pos_df, minus_tuple_lsts, rev_pos_df]


def plot_first_step_phase(return_dict):
    oldphase1, oldphase2, phased1, phased2 = return_dict['initial_phase_res']
    plt.scatter(oldphase1, oldphase2, c=1 + phased1.astype(int) + 2 * phased2.astype(int))
    plt.title('Initial phasing')
    plt.colorbar()
    plt.xlim((-25, 200))
    plt.ylim((-25, 200))
    plt.show()
    return None

def plot_last_step_phase(return_dict):
    oldphase1, oldphase2, phased1, phased2 = return_dict['initial_phase_res']
    phase_df = return_dict['phase_df']
    phased1 = phase_df['final_step_phase1'].values
    phased2 = phase_df['final_step_phase2'].values
    
    plt.scatter(oldphase1, oldphase2, c=1 + phased1.astype(int) + 2 * phased2.astype(int))
    indices = phase_df.index[phase_df['allele1'] == 'N/A']
    plt.scatter(oldphase1[indices], oldphase2[indices], c = 'black', marker = 'x')
    plt.title('Final phasing')
    plt.colorbar()
    plt.xlim((-25, 200))
    plt.ylim((-25, 200))
    plt.show()
    return None

def phase_trios(r, gene, hlatypes):
    kid = r['Individual ID']
    pat = r['Paternal ID']
    mat = r['Maternal ID']
    
    lst_kid = hlatypes[hlatypes['Sample ID'] == kid][['HLA-' + gene + ' 1', 'HLA-' + gene + ' 2']].values[0]
    set_kid = set(lst_kid)
    
    pattypes = hlatypes[hlatypes['Sample ID'] == pat][['HLA-' + gene + ' 1', 'HLA-' + gene + ' 2']].values
    if len(pattypes) != 0:
        lst_pat = pattypes[0]
        set_pat = set(lst_pat)
    else:
        set_pat = {None}
        pattypes = None
    mattypes = hlatypes[hlatypes['Sample ID'] == mat][['HLA-' + gene + ' 1', 'HLA-' + gene + ' 2']].values
    if len(mattypes) != 0:
        lst_mat = mattypes[0]
        set_mat = set(lst_mat)
    else:
        set_mat = {None}
        mattypes = None
    
    if np.nan in set_kid:
        if len(set_kid) == 1:
            undetermined_trios.append(kid)
        elif len(set_kid) == 2:
            set_kid.discard(np.nan)
            kidtype = list(set_kid)[0]
            if np.nan not in (set_pat.union(set_mat)):
                if (kidtype in set_pat) and (kidtype not in set_mat):
                    r['allele1'] = kidtype
                elif (kidtype in set_mat) and (kidtype not in set_pat):
                    r['allele2'] = kidtype
                else:
                    pass
            else:
                if (np.nan not in set_mat) and (kidtype not in set_mat):
                    r['allele1'] = kidtype
                elif (np.nan not in set_pat) and (kidtype not in set_pat):
                    r['allele2'] = kidtype
                else:
                    undetermined_trios.append(kid)
    else:
        if len(set_kid) == 1:
            r['allele1'] = lst_kid[0]
            r['allele2'] = lst_kid[0]
        else:
            if (lst_kid[0] in set_pat) and (lst_kid[0] not in set_mat):
                if np.nan not in set_mat:
                    r['allele1'] = lst_kid[0]
                    r['allele2'] = lst_kid[1]
                else:
                    if lst_kid[1] not in set_pat:
                        r['allele1'] = lst_kid[0]
                        r['allele2'] = lst_kid[1]
                    else:
                        undetermined_trios.append(kid)
            elif (lst_kid[1] in set_pat) and (lst_kid[1] not in set_mat):
                if np.nan not in set_mat:
                    r['allele1'] = lst_kid[1]
                    r['allele2'] = lst_kid[0]
                else:
                    if lst_kid[0] not in set_pat:
                        r['allele1'] = lst_kid[1]
                        r['allele2'] = lst_kid[0]
                    else:
                        undetermined_trios.append(kid)  
            elif (lst_kid[0] in set_mat) and (lst_kid[0] not in set_pat):
                if np.nan not in set_pat:
                    r['allele1'] = lst_kid[1]
                    r['allele2'] = lst_kid[0]
                else:
                    if lst_kid[1] not in set_mat:
                        r['allele1'] = lst_kid[1]
                        r['allele2'] = lst_kid[0]
                    else:
                        undetermined_trios.append(kid)  
            elif (lst_kid[1] in set_mat) and (lst_kid[1] not in set_pat):
                if np.nan not in set_pat:
                    r['allele1'] = lst_kid[0]
                    r['allele2'] = lst_kid[1]
                else:
                    if lst_kid[0] not in set_mat:
                        r['allele1'] = lst_kid[0]
                        r['allele2'] = lst_kid[1]
                    else:
                        undetermined_trios.append(kid)  
    return r

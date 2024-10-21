import io
import os
import re
import sys
import csv
import gzip
import time
import json
import pickle
import secrets
import multiprocessing
import subprocess
import resource
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
# from plotnine import *
import numpy as np
import scipy as sp
import statsmodels.api as sm
import random
# from collections import Counter
# import seaborn as sns
# import matplotlib.colors as mcolors
# from matplotlib.ticker import FuncFormatter
import itertools
import collections
import pyreadr
import pywfa
from IPython.display import display_html
# import patchworklib as pw
sys.path.append('/well/band/users/rbx225/software/lcwgsus/')
import lcwgsus
from lcwgsus.variables import *
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.options.mode.chained_assignment = None

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
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in reversed(seq))

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

def phred_to_scores(bq):
    return [ord(char) - 33 for char in bq]

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
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

def main(hla_gene_information_file, gene, bam, db_dir, ofile, strict):
    get_alignment_scores(hla_gene_information_file, gene, bam, db_dir, ofile, strict)
    
if __name__ == "__main__":
    hla_gene_information_file = snakemake.params.hla_gene_information_file
    gene = snakemake.wildcards.gene
    bam = snakemake.input.bam
    db_dir = snakemake.params.db_dir
    ofile = snakemake.output.matrix
    strict = snakemake.params.strict
    
    main(hla_gene_information_file, gene, bam, db_dir, ofile, strict)


load(snakemake@input[['RData']])

qreads = all_results[[1]][[1]][[4]]
qmate = all_results[[1]][[1]][[5]]
qpair = all_results[[1]][[1]][[2]]
quilt = all_results[[1]][[11]]

save(qreads, qmate, qpair, quilt, file = snakemake@output[['extracted_RData']])


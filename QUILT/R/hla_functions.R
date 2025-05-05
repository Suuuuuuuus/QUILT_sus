quilt_hla_one_sample <- function(
    iSample,
    final_set_of_results,
    bamfiles,
    chr,
    region,
    regstart,
    regend,
    regmid,
    refseq_file,
    newkmers,
    lookup,
    revlookup,
    fullalleles,
    quilt_hla_haplotype_panelfile,
    quilt_seed,
    quilt_buffer,
    quilt_bqFilter,
    nGibbsSamples,
    hlahaptypes,
    summary_best_alleles_threshold,
    outputdir
) {

    n <- length(bamfiles)
    if (iSample %in% (match(1:10, ceiling(10 * (1:n / n))))) {
        print_message(paste0("Processing file ", iSample, " out of ", n))
    }
    
    bamfile <- bamfiles[iSample]
    python_output_dir <- tempdir()
    dir.create(python_output_dir, showWarnings = FALSE)

    parts <- unlist(strsplit(outputdir, "/"))
    basedir <- paste(parts[1:(length(parts) - 2)], collapse = "/")

    script <- "../software/QUILT_test/QUILT/Python/hla_align.py" # Modify this later to be relative path
    system(paste("python", script, region, bamfile, python_output_dir, basedir))

    readset1 <- read.csv(file.path(python_output_dir, "/reads1.csv"), header = FALSE)
    readset2 <- read.csv(file.path(python_output_dir, "/reads2.csv"), header = FALSE)
    readlikelihoodmat <- read.table(file.path(python_output_dir, "/pair_likelihood_matrix.ssv"), header = TRUE, row.names = 1, sep = " ", check.names = FALSE)
    pairedscores <- read.table(file.path(python_output_dir, "/mate_likelihood_matrix.ssv"), header = TRUE, row.names = 1, sep = " ", check.names = FALSE)
    readlikelihoodmat <- as.matrix(readlikelihoodmat)
    pairedscores <- as.matrix(pairedscores)
    overall <- readlikelihoodmat

    that <- readset1
    that2 <- readset2

    readscaledlikelihoodmat <- NULL
    fourdigitreadscaledlikelihoodmat <- NULL
    intersectfourdigitreadscaledlikelihoodmat <- NULL
    intersectquiltscaledlikelihoodmat <- NULL
    intersectreadscaledlikelihoodmat <- NULL
    combinedscaledlikelihoodmat <- NULL
    combinedresults <- NULL
    mappingonlyresults <- NULL
    
    unlink(python_output_dir)
    
    ##
    ## really strong Robbie hack because I don't know nature of how below works
    ## basically, do 1:20 are Gibbs samples
    ##
    i_gibbs_sample <- 1
    
    for(i_gibbs_sample in 1:(nGibbsSamples + 1)) {

        ## print_message(paste0("Re-shaping QUILT output ", i_gibbs_sample, " / ", nGibbsSamples))
        if (i_gibbs_sample == (nGibbsSamples + 1)) {
            use_final_phase <- TRUE
            use_averaging <- TRUE
        } else {
            use_final_phase <- FALSE
            use_averaging <- FALSE
        }
        
        ##resset is the results from QUILT
        ##can use this to make likelihood for different 4-digit codes, see below processing
        ##QUILT already takes into account allele frequency in the population
        ##there are some 4-digit codes that are non-overlapping, and even some weight on unknown alleles
        ##now take reads in our region and process
        resset <- reshape_QUILT_output(
            final_set_of_results = final_set_of_results,
            iSample = iSample,
            hlahaptypes = hlahaptypes,
            use_final_phase = use_final_phase,
            i_gibbs_sample = i_gibbs_sample
        )
        ##
        ## store a list of raw data things
        ##
        quiltprobs <- resset
        colnames(quiltprobs) <- c("Allele_1_best_prob","Allele_2_best_prob","Summed_means")
        raw_output <- list(
            quiltprobs = quiltprobs,
            readlikelihoodmat = readlikelihoodmat,
            readset1 = readset1,
            readset2 = readset2,
            pairedscores = pairedscores,
            ndistinctfragments = nrow(pairedscores)
        )
        
        out <- reshape_and_filter_resset(
            resset = resset,
            region = region,
            use_averaging = use_averaging
        )
        newresset <- out[["newresset"]]        
        newresset2 <- out[["newresset2"]]
        newquiltprobs <- out[["newquiltprobs"]]
    
        newphasedquiltprobs <- newresset2
        quiltscaledlikelihoodmat <- matrix(1,nrow=nrow(newresset),ncol=nrow(newresset))
        rownames(quiltscaledlikelihoodmat) <- rownames(newphasedquiltprobs)
        colnames(quiltscaledlikelihoodmat) <- rownames(newphasedquiltprobs)
        quiltscaledlikelihoodmat <- quiltscaledlikelihoodmat*newphasedquiltprobs[,1]
        quiltscaledlikelihoodmat <- t(t(quiltscaledlikelihoodmat)*newphasedquiltprobs[,2])
        quiltscaledlikelihoodmat <- 0.5*(quiltscaledlikelihoodmat+t(quiltscaledlikelihoodmat))


        ## make a list of processed output from reads
        ## now if we have some interesting results, process read based inferences
        if(length(that) | length(that2)){
            if (i_gibbs_sample == 1) {
                ## this should depend on quilt only through alleles which are constant
                output_fourdigitreadscaledlikelihoodmat <- get_fourdigitreadscaledlikelihoodmat(
                    overall = overall,
                    newphasedquiltprobs = newphasedquiltprobs
                )
            }
            fourdigitreadscaledlikelihoodmat <- output_fourdigitreadscaledlikelihoodmat[["fourdigitreadscaledlikelihoodmat"]]
            mappingonlyresults <- getbestalleles(fourdigitreadscaledlikelihoodmat, summary_best_alleles_threshold)
            vv2 <- output_fourdigitreadscaledlikelihoodmat[["vv2"]]
            readscaledlikelihoodmat <- output_fourdigitreadscaledlikelihoodmat[["readscaledlikelihoodmat"]]
            intersectreadscaledlikelihoodmat <- output_fourdigitreadscaledlikelihoodmat[["intersectreadscaledlikelihoodmat"]]
            ##
            ##intersection of this with quilt four digit inferences, summing the intersection, above (should sum to 1)
            ## 
            intersectfourdigitreadscaledlikelihoodmat=fourdigitreadscaledlikelihoodmat[names(vv2),names(vv2)]

            ## Sus
            intersectfourdigitreadscaledlikelihoodmat[lower.tri(intersectfourdigitreadscaledlikelihoodmat, diag = F)] <- 0
            ## Sus

            intersectfourdigitreadscaledlikelihoodmat=intersectfourdigitreadscaledlikelihoodmat/sum(intersectfourdigitreadscaledlikelihoodmat)
            
            ##
            ##intersection of quilt matrix with four digit inferences (rescaled to sum to one, ordered to match above)
            ## 
            intersectquiltscaledlikelihoodmat=quiltscaledlikelihoodmat[rownames(intersectfourdigitreadscaledlikelihoodmat),colnames(intersectfourdigitreadscaledlikelihoodmat)]

            ## Sus
            intersectquiltscaledlikelihoodmat[lower.tri(intersectquiltscaledlikelihoodmat, diag = F)] <- 0
            ## Sus

            intersectquiltscaledlikelihoodmat=intersectquiltscaledlikelihoodmat/sum(intersectquiltscaledlikelihoodmat)
            ##product of intersection matrices gives overall likelihood, rescaled to sum to one
            combinedscaledlikelihoodmat=intersectfourdigitreadscaledlikelihoodmat*intersectquiltscaledlikelihoodmat

            # ## Sus
            # w1 <- max(intersectfourdigitreadscaledlikelihoodmat)
            # w2 <- max(intersectquiltscaledlikelihoodmat)
            # w1 <- w1 / (w1 + w2)
            # w2 <- 1 - w1

            # combinedscaledlikelihoodmat <- (intersectfourdigitreadscaledlikelihoodmat**w1)*(intersectquiltscaledlikelihoodmat**w2)
            # ## Sus

            combinedscaledlikelihoodmat=combinedscaledlikelihoodmat/sum(combinedscaledlikelihoodmat)

            combinedresults <- get_best_alleles(combinedscaledlikelihoodmat, summary_best_alleles_threshold)
            ##for a matrix, make a little function to output allele pair probabilities, the answer
        }

        ## Sus
        newnames = sort(colnames(quiltscaledlikelihoodmat))
        tmp=quiltscaledlikelihoodmat[newnames,newnames]
        tmp[lower.tri(tmp, diag = F)] <- 0
        tmp = tmp/sum(tmp)
        quiltresults <- get_best_alleles(tmp, summary_best_alleles_threshold)
        ## Sus
        
        # quiltresults <- get_best_alleles(quiltscaledlikelihoodmat, summary_best_alleles_threshold)
        processed_output <- list(
            newquiltprobs = newquiltprobs,
            newphasedquiltprobs = newphasedquiltprobs,
            quiltscaledlikelihoodmat = quiltscaledlikelihoodmat,
            readscaledlikelihoodmat = readscaledlikelihoodmat,
            intersectreadscaledlikelihoodmat = intersectreadscaledlikelihoodmat,
            fourdigitreadscaledlikelihoodmat = fourdigitreadscaledlikelihoodmat,
            intersectfourdigitreadscaledlikelihoodmat = intersectfourdigitreadscaledlikelihoodmat,
            intersectquiltscaledlikelihoodmat = intersectquiltscaledlikelihoodmat,
            combinedscaledlikelihoodmat = combinedscaledlikelihoodmat
        )

        ## now per-gibbs sample, save
        ## need this
        if (i_gibbs_sample == 1) {
            joint_quiltscaledlikelihoodmat <- quiltscaledlikelihoodmat
            if(length(that) | length(that2)){
                joint_combinedscaledlikelihoodmat <- combinedscaledlikelihoodmat
            }
        } else if (i_gibbs_sample <= nGibbsSamples) {
            joint_quiltscaledlikelihoodmat <-
                joint_quiltscaledlikelihoodmat + 
                quiltscaledlikelihoodmat
            if(length(that) | length(that2)){
                joint_combinedscaledlikelihoodmat <-
                    joint_combinedscaledlikelihoodmat + 
                    combinedscaledlikelihoodmat
            }
        }

    }

    ##
    ## normalize joint version
    ##
    
    joint_quiltscaledlikelihoodmat <- joint_quiltscaledlikelihoodmat / nGibbsSamples

    ## Sus
    newnames = sort(colnames(joint_quiltscaledlikelihoodmat))
    joint_quiltscaledlikelihoodmat=joint_quiltscaledlikelihoodmat[newnames,newnames]
    joint_quiltscaledlikelihoodmat[lower.tri(joint_quiltscaledlikelihoodmat, diag = F)] <- 0
    joint_quiltscaledlikelihoodmat = joint_quiltscaledlikelihoodmat/sum(joint_quiltscaledlikelihoodmat)
    joint_quiltresults <- get_best_alleles(joint_quiltscaledlikelihoodmat, summary_best_alleles_threshold)
    ## Sus

    if(length(that) | length(that2)){
        # joint_combinedscaledlikelihoodmat <- joint_combinedscaledlikelihoodmat / nGibbsSamples

        cols = colnames(intersectfourdigitreadscaledlikelihoodmat)
        joint_quiltscaledlikelihoodmat = joint_quiltscaledlikelihoodmat[cols,cols]
        joint_quiltscaledlikelihoodmat[lower.tri(joint_quiltscaledlikelihoodmat, diag = F)] <- 0
        joint_quiltscaledlikelihoodmat = joint_quiltscaledlikelihoodmat/sum(joint_quiltscaledlikelihoodmat)

        # w1 <- max(intersectfourdigitreadscaledlikelihoodmat)
        # w2 <- max(joint_quiltscaledlikelihoodmat)
        # w1 <- w1 / (w1 + w2)
        # w2 <- 1 - w1
        # joint_combinedscaledlikelihoodmat <- (intersectfourdigitreadscaledlikelihoodmat**w1)*(tmp**w2)

        joint_combinedscaledlikelihoodmat <- intersectfourdigitreadscaledlikelihoodmat*joint_quiltscaledlikelihoodmat
        joint_combinedscaledlikelihoodmat = joint_combinedscaledlikelihoodmat/sum(joint_combinedscaledlikelihoodmat)

        joint_combinedresults <- get_best_alleles(joint_combinedscaledlikelihoodmat, summary_best_alleles_threshold)
    } else {
        joint_combinedresults <- joint_quiltresults
    }




    
    ##
    ## robbie hacks
    ## try and do proper unphased version
    ## 
    ## first, for QUILT, this is easy. take two highest posterior probabilities
    quilt_unphased_probs <- newresset[, 3]
    y <- quilt_unphased_probs[order(-quilt_unphased_probs)]
    unphased_summary_quilt_only <- c(y[1:2], conf = sum(y[1:2]))
    ## now, for Simon's thing
    ## first, get intersection
    if (!is.null(intersectfourdigitreadscaledlikelihoodmat)) {
        ## work on intersection, and make Simon's thing unphased
        a <- rownames(intersectfourdigitreadscaledlikelihoodmat)
        ## great
        simon_map_phased_probs <- intersectfourdigitreadscaledlikelihoodmat[a %in% names(y), a %in% names(y)]
        simon_map_unphased_probs <- rowSums(simon_map_phased_probs) ## I think
        ## OK, now merge
        joint <- intersect(names(quilt_unphased_probs), names(simon_map_unphased_probs))
        simon_unphased_joint <- simon_map_unphased_probs[match(joint, names(simon_map_unphased_probs))]
        quilt_unphased_joint <- quilt_unphased_probs[match(joint, names(quilt_unphased_probs))]
        both_unphased <- simon_unphased_joint * quilt_unphased_joint
        both_unphased <- both_unphased / sum(both_unphased)
        ##
        y <- both_unphased[order(-both_unphased)]
        unphased_summary_both <- c(y[1:2], conf = sum(y[1:2]))
    } else {
        unphased_summary_both <- NULL
    }
    ## simon_unphased_joint[c("A*02:01", "A*23:01")]
    ## unphased_summary_quilt_only
    ##unphased_summary_both

    
    ##various of above terms are empty if there is no additional informartion from  reads in a gene (at low coverage)
    ##below is a function to predict hla combinations from the above
    ## save in finaloutfile
    ##if that and that2 are empty, what happens, is one question
    ##should just have empty (null) containers for the other things I think

    hla_results <- list(
        raw_output = raw_output,
        processed_output = processed_output,
        region = region,
        quiltresults = quiltresults,
        combinedresults = combinedresults,
        mappingonlyresults = mappingonlyresults,
        unphased_summary_quilt_only = unphased_summary_quilt_only,
        unphased_summary_both = unphased_summary_both,
        joint_quiltresults = joint_quiltresults,
        joint_combinedresults = joint_combinedresults,
        quiltmat = joint_quiltscaledlikelihoodmat,
        readmat = intersectfourdigitreadscaledlikelihoodmat,
        combinedmat = joint_combinedscaledlikelihoodmat
    )

    return(hla_results)
    
}

check_samtools <- function() {
    ## /data/smew2/myers/1000G/samtools-1.10/
    v <- strsplit(system("samtools --version", intern = TRUE)[1], " ", fixed = TRUE)[[1]][2]
    vs <- as.numeric(strsplit(v, ".", fixed = TRUE)[[1]])
    ok <- vs[1] >= 2 | (vs[1] == 1 & vs[2] >= 10)
    if (!ok) {
        stop("Need samtools >=1.10 in PATH")
    }
    return(NULL)
}





reshape_QUILT_output <- function(
    final_set_of_results,
    iSample,
    hlahaptypes,
    use_final_phase,
    i_gibbs_sample
) {
    ## OK here's where Simons starting on this
    if (use_final_phase) {
        g1 <- final_set_of_results[[iSample]]$gamma1
        g2 <- final_set_of_results[[iSample]]$gamma2
    } else {
        g1 <- final_set_of_results[[iSample]]$list_of_gammas[[i_gibbs_sample]][[1]]
        g2 <- final_set_of_results[[iSample]]$list_of_gammas[[i_gibbs_sample]][[2]]
    }
    g3 <- final_set_of_results[[iSample]]$gamma_total    
    names(g1) <- hlahaptypes
    names(g2) <- hlahaptypes
    names(g3) <- hlahaptypes
    ##tabulate
    uniques=unique(names(g1))
    resset=matrix(0,nrow=length(uniques),ncol=3)
    rownames(resset)=uniques
    for(i in 1:length(g1)) resset[names(g1)[i],1]=as.double(resset[names(g1)[i],1])+g1[i]
    for(i in 1:length(g2)) resset[names(g2)[i],2]=as.double(resset[names(g2)[i],2])+g2[i]
    for(i in 1:length(g3)) resset[names(g3)[i],3]=as.double(resset[names(g3)[i],3])+g3[i]
    ## print(resset[order(resset[,1],decreasing=T),][1:10,])
    ## print(resset[order(resset[,2],decreasing=T),][1:10,])
    ## print(resset[order(resset[,3],decreasing=T),][1:10,])
    return(resset)
}

get_mode <- function(x) {
  u <- unique(x)
  tab <- tabulate(match(x, u))
  u[tab == max(tab)]
}


## robbie made function from Simon spaghetti code

reshape_and_filter_resset <- function(resset, region, use_averaging = TRUE) {
    newnames <- matrix(nrow=0,ncol=2)
    dd <- grep("/",rownames(resset))
    cc <- rownames(resset)
    ee <- cbind(cc,1:nrow(resset),rep(1,nrow(resset)))
    if (length(dd)){
        ee <- ee[-dd,]
	for(j in dd){
            check=cc[j]
            nn=unlist(strsplit(unlist(strsplit(check,":")),"/"))
            nn2=paste(nn[1],nn[2:length(nn)],sep=":")
            ee=rbind(ee,cbind(nn2,j,1/length(nn2)))
	}
    }
    newresset=matrix(0,nrow=length(unique(ee[,1])),ncol=3)
    rownames(newresset)=unique(ee[,1])
    for(i in 1:nrow(ee)) {
        newresset[ee[i,1],]=newresset[ee[i,1],]+as.double(ee[i,3])*resset[as.double(ee[i,2]),]
    }
    colnames(newresset)=colnames(resset)
    rownames(newresset)=paste(region,"*",rownames(newresset),sep="")
    newquiltprobs=newresset
    ## get some approximate phasing information for resset
    newresset2 <- newresset
    if (use_averaging) {
        for(i in 1:nrow(newresset2)){
            newresset2[i,1] <- newresset[i,1] / (newresset[i,1] + newresset[i,2]) * newresset[i,3] * 2
            newresset2[i,2] <- newresset[i,2] / (newresset[i,1] + newresset[i,2]) * newresset[i,3] * 2
        }
        newresset2[,1]=newresset2[,1]/sum(newresset2[,1])
        newresset2[,2]=newresset2[,2]/sum(newresset2[,2])
    }
    return(
        list(
            newresset = newresset,            
            newresset2 = newresset2,
            newquiltprobs = newquiltprobs
        )
    )
}






get_fourdigitreadscaledlikelihoodmat <- function(overall, newphasedquiltprobs) {
    ##first scale (by number of types for each four-digit code) and convert overall to likelihood scale, scaled to sum to 1
    ## only keep alleles typed to 4-digit accuracy or above
    tt=rownames(overall)
    ##count colons
    colons=1:length(tt)*0
    for(i in 1:length(tt)) colons[i]=sum(substring(tt[i],1:nchar(tt[i]),1:nchar(tt[i]))==":")
    ##keep alleles typed at four digit accuracy or above
    ##could go to six or even eight digit accuracy
    keep=tt[colons>=1]
    overall2=overall[keep,keep]
    overall2=overall2-max(overall2)
    overall2=exp(overall2)
    ##so this is raw likelihood for all 4-digit or above
    ##now extract 4-digit codes
    ##same total weight for each
    keep2=1:length(keep)
    for(i in 1:length(keep2)) {
        keep2[i]=paste(unlist(strsplit(keep[i],":"))[1:2],collapse=":")
    }
    vv=table(keep2)
    weights=1:nrow(overall2)*0
    for(i in 1:length(weights)) weights[i]=vv[keep2[i]]
    ##scale by rows
    overall2=overall2/weights
    ##scale by columns
    overall2=t(t(overall2)/weights)
    ##so weight on a particular allele is 1/number of alleles seen, corresponding to an equal likelihood of all four-digit HLA types (this is better for combining with quilt results which do weight by observed four-digit types)
    overall2=overall2/sum(overall2)
    readscaledlikelihoodmat=overall2
    ##intersection of this with quilt four digit inferences, scaled to sum to 1
    fourdigitsseen=keep2
    cond <- fourdigitsseen %in% rownames(newphasedquiltprobs)
    intersectreadscaledlikelihoodmat=readscaledlikelihoodmat[cond,cond]
    intersectreadscaledlikelihoodmat=intersectreadscaledlikelihoodmat/sum(intersectreadscaledlikelihoodmat)
    keep3=keep2[cond]
    vv2=table(keep3)
    ##
    ##
    ##
    ## four digit inferences, summing the above (should sum to 1)
    fourdigitreadscaledlikelihoodmat=matrix(0,nrow=length(vv),ncol=length(vv))
    rownames(fourdigitreadscaledlikelihoodmat)=names(vv)
    colnames(fourdigitreadscaledlikelihoodmat)=names(vv)
    ## fourdigit intersection with quilt (1000G) alleles, summing the above
    rows=match(keep2,names(vv))
    cols=rows
    for(i in 1:length(keep2)) {
        for(j in 1:length(keep2)) {
            fourdigitreadscaledlikelihoodmat[rows[i],cols[j]] <-
                fourdigitreadscaledlikelihoodmat[rows[i],cols[j]]+readscaledlikelihoodmat[i,j]
        }
    }
    return(
        list(
            fourdigitreadscaledlikelihoodmat = fourdigitreadscaledlikelihoodmat,
            vv2 = vv2,
            readscaledlikelihoodmat = readscaledlikelihoodmat,
            intersectreadscaledlikelihoodmat = intersectreadscaledlikelihoodmat
        )
    )
}


#######from below, is calling pipeline functions and then code, for the read-based calling of HLA type using only reads within each gene

##for a given region, now we have to read in the data

get_best_alleles <- function(df, thresh = 0.99) {
    # Expect df to be an upper triangular matrix
    df = df/sum(df)
    alleles <- rownames(df)
    upper_indices <- which(upper.tri(df, diag = TRUE), arr.ind = TRUE)
    
    lhoods <- df[upper_indices]
    result <- data.frame(
        bestallele1 = alleles[upper_indices[, 1]],
        bestallele2 = alleles[upper_indices[, 2]],
        lhoods = lhoods
    )
    result <- result[order(-result$lhoods), ]
    row.names(result) <- NULL
    result$sums <- cumsum(result$lhoods)
    row2 <- min(which(result$sums >= thresh))

    return(as.matrix(result[1:row2, , drop = FALSE]))
}

getbestalleles <- function(matrix,thresh=0.99){
    ##make diagonal
    for(i in 2:nrow(matrix)) {
        matrix[i,1:(i-1)]=0
    }
    diag(matrix)=diag(matrix)/2
    matrix=matrix/sum(matrix)
    bestallele1=rep(rownames(matrix),nrow(matrix))[order(matrix,decreasing=T)]
    bestallele2=rep(rownames(matrix),nrow(matrix))[order(t(matrix),decreasing=T)]
    lhoods=sort(matrix,decreasing=T)
    sums=cumsum(lhoods)
    results=cbind(bestallele1,bestallele2,lhoods,sums)
    row2=min(which(sums>=thresh))
    return(results[1:row2, , drop = FALSE])
}

summarize_all_results_and_write_to_disk <- function(
    all_results,
    sampleNames,
    what,
    outputdir,
    summary_prefix,
    summary_suffix,
    only_take_top_result = FALSE
) {
    result <- data.frame(rbindlist(lapply(1:length(all_results), function(iSample) {
        x <- all_results[[iSample]]
        y <- data.frame(
            sample_number = iSample,
            sample_name = sampleNames[iSample],
            x[[what]]
        )
        ## re-name some of the output
        colnames(y)[colnames(y) == "lhoods"] <- "post_prob"
        if (only_take_top_result) {
            y <- y[1, , drop = FALSE]
        }
        y
    })))
    file <- paste0(summary_prefix, ".", summary_suffix)    
    if (outputdir != "") {
        file <- file.path(outputdir, file)
    }
    write.table(
        result,
        file = file,
        row.names = FALSE,
        col.names = TRUE,
        sep = "\t",
        quote = FALSE
    )
    result
}
    

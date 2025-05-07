#!/usr/bin/env Rscript

#####if (!require("BiocManager", quietly = TRUE))
#####  install.packages("BiocManager")

pacman::p_load("BiocManager","edgeR")

input_message <- "Input in format: Rscript file.R {input_file} {output_file} {target}"
# get arguments from command line
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop(input_message, call. = FALSE)
} else{
  input_file <- args[1]
  output_file <- args[2]
  target <- args[3]
}

# read file as a matrix, index column is the first one, don't alter the name of columns from - to .
x <- as.matrix(read.csv(input_file, row.names = 1, check.names=FALSE))
# get the groupings (classes) of samples
groups <- x[, target]
# remove column with groupings
x <- x[, colnames(x) != target]
# transverse of matrix to genes in rows and samples i columns
x2 <- t(x)
# if target was not numeric, need to revert the matrix back to numeric
class(x2) <- "numeric"

# create object DGEList, contains count matrix and data.frame samples with:
#   groupings, lib sizes, and normalization factors per sample
y <- DGEList(counts = x2, group = groups)

# selects genes to filter out due to very low counts
keep <- filterByExpr(y)

# filter out selected genes
y2 <- y[keep,,keep.lib.sizes=FALSE]

# calculates the normalization factors for each sample using TMM
y3 <- calcNormFactors(y2)

design <- model.matrix(~groups)

# estimate dispersion per sample using qCML for the negative binomial model
y4 <- estimateDisp(y3,design)

# perform exact test to calculate p-values, logFC, and logCPM per gene
et <- exactTest(y4)
saveRDS(et, output_file)

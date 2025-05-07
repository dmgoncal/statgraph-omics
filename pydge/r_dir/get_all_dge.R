#!/usr/bin/env Rscript

####if (!require("BiocManager", quietly = TRUE))
####  install.packages("BiocManager")

pacman::p_load("BiocManager","edgeR")

input_message = "Input in format: Rscript file.R {dge_data.rds file} {temp_output_csv}"
# get arguments from command line
args = commandArgs(trailingOnly=TRUE)

if (length(args)!=2) {
  stop(input_message, call.=FALSE)
} else{
  dge_file <- args[1]
  temp_output_file <- args[2]
}

dge <- readRDS(dge_file)
n <- nrow(dge$table)
# returns top n genes by sort.by, adjusted with adjust.method
#   and only returns if p-value lower than p.value
genes <- topTags(dge, n = n, adjust.method = "BH", sort.by = "PValue", p.value = 1)
write.csv(genes, temp_output_file, row.names = TRUE)

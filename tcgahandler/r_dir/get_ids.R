#!/usr/bin/env Rscript

library('pacman')
pacman::p_load(BiocManager, preprocessCore, DT, TCGAbiolinks, 
               SummarizedExperiment, sesameData, sesame)


input_message = "Input in format: Rscript file.R {project} {path to save file}"
args = commandArgs(trailingOnly=TRUE)  # get arguments from command line

if (length(args)!=2) {
  stop(input_message, call.=FALSE)
} else{
  project <- args[1]
  file_path <- args[2]
}

clinical <- TCGAbiolinks::GDCquery_clinic(project = project, type = "clinical")

saveRDS(clinical$submitter_id, file=file_path)

pacman::p_unload(all)


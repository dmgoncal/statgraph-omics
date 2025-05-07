#!/bin/bash

for target in vital_status primary_diagnosis
do
    for layer in mirna protein
    do
        for proj in TCGA-LGG TCGA-KIRC TCGA-COAD TCGA-OV TCGA-LUAD
        do
            python graph_classification_cv_script.py $proj $layer $target 0.8 -v -y
        done
    done

    for layer in mrna
    do
        for proj in TCGA-LGG TCGA-KIRC TCGA-COAD TCGA-OV TCGA-LUAD
        do
            python graph_classification_cv_script.py $proj $layer $target 0.8 -v -y -d=1000
        done
    done
done

echo 'Finished'

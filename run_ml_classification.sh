#!/bin/bash

for target in vital_status primary_diagnosis
do
    for layer in mirna protein
    do
        for proj in TCGA-LGG TCGA-KIRC TCGA-COAD TCGA-OV TCGA-LUAD
        do
            python ml_classification_script.py $proj $layer $target -v -y
        done
    done

    for layer in mrna
    do
        for proj in TCGA-LGG TCGA-KIRC TCGA-COAD TCGA-OV TCGA-LUAD
        do
            python ml_classification_script.py $proj $layer $target -v -y -d=1000
        done
    done
done

echo 'Finished'
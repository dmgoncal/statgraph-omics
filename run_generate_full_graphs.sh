#!/bin/bash

for target in vital_status primary_diagnosis
do
    for layer in mirna protein mrna
    do
        for proj in TCGA-LGG TCGA-KIRC TCGA-COAD TCGA-OV TCGA-LUAD
        do
            python generate_full_graphs_script.py $proj $layer $target 0.8 -v -y
        done
    done
done

echo 'Finished'
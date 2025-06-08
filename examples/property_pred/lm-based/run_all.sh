#!/bin/bash

binary_classification_datasets=(
    AF_APML
    BBP_APML
    cCPP_Pepland
    Nonfouling
    Solubility
    AMP_PepDiffusion
    AV_APML
    cAB_APML2
    ACE_APML
    ACP_APML
    Aox_APML
    DLAD_BioDADPep
    DPPIV_APML
    Neuro_APML
    QS_APML
    TTCA_TCAHybrid
    Hemo_PeptideBERT
    Tox_APML
)

for dataset in "${binary_classification_datasets[@]}"; do
    ./run_experiments.sh "$dataset" > "${dataset}.log" 2>&1
done

regression_datasets=(
    all-AMP
    E.coli
    P.aeruginosa
    S.aureus
    HemoPI2
)

for dataset in "${regression_datasets[@]}"; do
    ./run_experiments.sh "$dataset" > "${dataset}.log" 2>&1
done

# nohup ./run_all.sh > run_all.log 2>&1 &
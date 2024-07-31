#!/bin/bash

DATA_DIR=./data/
PT_DIR=./preprocessing/
SEQ_SIM=100

: <<COMMENT
mkdir $DATA_DIR
printf "\n\n  DATA DIRECTORY (%s) CREATED!\n" $DATA_DIR

printf "\n\n  DOWNLOADING SIFTS-GO DATA...\n"
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O $DATA_DIR/pdb_chain_go.tsv.gz

printf "\n\n  DOWNLOADING PDB SEQRES SEQUENCES...\n"
wget ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O $DATA_DIR/pdb_seqres.txt.gz

printf "\n\n  DOWNLOADING PDB CLUSTERS...\n"
wget https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-$SEQ_SIM.txt -O $DATA_DIR/clusters-by-entity-$SEQ_SIM.txt

printf "\n\n  DOWNLOADING GO HIERARCHY...\n"
wget https://current.geneontology.org/ontology/subsets/goslim_plant.obo -O $DATA_DIR/goslim-plant.obo
COMMENT

printf "\n\n  PREPROCESSING GO-ANNOTATIONS [Please wait this process may take a few minutes]...\n"
python3 create_nrPDB_GO_annot.py \
    -sifts $DATA_DIR/pdb_chain_go.tsv.gz \
    -bc $DATA_DIR/clusters-by-entity-$SEQ_SIM.txt \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -obo $DATA_DIR/goslim_plant.obo \
    -out $DATA_DIR/nrPDB-GO \

printf "\n\n  RETRIEVING PDB FILES AND CREATING DISTANCE MAPS...\n"
mkdir $DATA_DIR/annot_pdb_chains_npz/
python3 pdb2distmap.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -seqres $DATA_DIR/pdb_seqres.txt.gz \
    -num_threads 20 \
    -bc $DATA_DIR/clusters-by-entity-$SEQ_SIM.txt \
    -out_dir $DATA_DIR/annot_pdb_chains_npz/ \

rm -r obsolete/

printf "\n\n  CREATE PT FILES..."
mkdir $PT_DIR
python3 pydataset3.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -prot_list $DATA_DIR/nrPDB-GO_train.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 30 \
    -num_threads 30 \
    -tfr_prefix $PT_DIR/PDB_GO_train \

python3 pydataset3.py \
    -annot $DATA_DIR/nrPDB-GO_annot.tsv \
    -prot_list $DATA_DIR/nrPDB-GO_valid.txt \
    -npz_dir $DATA_DIR/annot_pdb_chains_npz/ \
    -num_shards 3 \
    -num_threads 3 \
    -tfr_prefix $PT_DIR/PDB_GO_valid \

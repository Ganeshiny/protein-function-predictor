#!/usr/bin/env python
from create_nrPDB_GO_annot import read_fasta
from biotoolbox.structure_file_reader import build_structure_container_for_pdb
from biotoolbox.contact_map_builder import DistanceMapBuilder
from Bio.PDB import PDBList

import numpy as np
import argparse
import time

def make_distance_maps(pdbfile, chain=None, sequence=None):
    pdb_handle = open(pdbfile, 'r')
    structure_container = build_structure_container_for_pdb(pdb_handle.read(), chain).with_seqres(sequence)

    mapper = DistanceMapBuilder(atom="CA", glycine_hack=-1)  
    ca = mapper.generate_map_for_pdb(structure_container)
    cb = mapper.set_atom("CB").generate_map_for_pdb(structure_container)
    pdb_handle.close()

    return ca.chains, cb.chains

import csv
from collections import defaultdict

def load_GO_annot(filename):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        for ont in onts:
            next(reader, None)  
            goterms[ont] = next(reader)
            next(reader, None)  
            gonames[ont] = next(reader)

        next(reader, None)  
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                prot2annot[prot][onts[i]] = [goterm for goterm in prot_goterms[i].split(',') if goterm != '']
    return prot2annot, goterms, gonames

def retrieve_pdb(pdb, chain, chain_seqres, pdir):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb, pdir=pdir)
    ca, cb = make_distance_maps(pdir + '/' + pdb+'.cif', chain=chain, sequence=chain_seqres)
    return ca[chain]['contact-map'], cb[chain]['contact-map']

def load_list(fname):
    pdb_chain_list = set()
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.add(line.strip())
    fRead.close()
    return pdb_chain_list

def write_annot_npz(prot, prot2seq, out_dir):
    pdb, chain = prot.split('-')
    print('pdb=', pdb, 'chain=', chain)
    try:
        A_ca, A_cb = retrieve_pdb(pdb.lower(), chain, prot2seq[prot], pdir=out_dir + 'tmp_PDB_files_dir')
        np.savez_compressed(out_dir + '/' + prot,
                            C_alpha=A_ca,
                            C_beta=A_cb,
                            seqres=prot2seq[prot],
                            )
    except Exception as e:
        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='preprocessing/data/nrPDB-GO_annot.tsv', help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-seqres', type=str, default='preprocessing/data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=20, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-bc', type=str, default='preprocessing/data/clusters-by-entity-95.txt', help="Clusters of PDB chains computed by Blastclust.")
    parser.add_argument('-out_dir', type=str, default='preprocessing/data/annot_pdb_chains_npz/', help="Output directory with distance maps saved in *.npz format.")
    args = parser.parse_args()

    prot2goterms, _, _ = load_GO_annot(args.annot)
    print ("### number of annotated proteins: %d" % (len(prot2goterms)))

    prot2seq = read_fasta(args.seqres)
    print ("### number of proteins with seqres sequences: %d" % (len(prot2seq)))

    import glob
    npz_pdb_chains = glob.glob(args.out_dir + '*.npz')
    npz_pdb_chains = [chain.split('/')[-1].split('.')[0] for chain in npz_pdb_chains]

    to_be_processed = list(prot2goterms.keys())
    to_be_processed = list(set(to_be_processed).difference(npz_pdb_chains))
    print ("Number of pdbs to be processed=", len(to_be_processed))
    print (to_be_processed)

    nprocs = args.num_threads
    out_dir = args.out_dir
    import multiprocessing
    nprocs = min(nprocs, multiprocessing.cpu_count())

    if nprocs > 4:
        pool = multiprocessing.Pool(processes=nprocs)
        pool.starmap(write_annot_npz, zip(to_be_processed, [prot2seq]*len(to_be_processed), [out_dir]*len(to_be_processed)))
    else:
        for prot in to_be_processed:
            write_annot_npz(prot, prot2seq, out_dir)

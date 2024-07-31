from Bio.Data.IUPACData import protein_letters_3to1 
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import networkx as nx
import numpy as np
import argparse
import requests
import obonet
import gzip
import csv


# ## input data:
# go-basic.obo (from: http://geneontology.org/docs/download-ontology/)
# pdb_chain_go.tsv (from: https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html)
# cluster_data (from: ftp://resources.rcsb.org/sequence/clusters/)

exp_evidence_codes = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED'])
root_terms = set(['GO:0008150', 'GO:0003674', 'GO:0005575'])
domain_terms = set(['GO:0048316'])

def read_fasta(fn_fasta):
    '''
    Out: a dictionary of protein ID as keys and fasta sequence as values

    Notes: 
    The key value is a PDB entry, not an entity. This is a 4 letter ID, in uppercase
    The sequences are only read if they are >= 60aa and <= 1000 aa. Larger protein are not read.
    '''
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    with gzip.open(fn_fasta, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            pdb, chain = prot.split('_')
            prot = pdb.upper() + '-' + chain
            if len(seq) >= 60 and len(seq) <= 1000:
                if len((set(seq).difference(aa))) == 0:
                    prot2seq[prot] = seq
    return prot2seq


def write_prot_list(protein_list, filename):
    '''
    Utility function that outputs a file with list of elements that are newline seprataed
    '''
    fWrite = open(filename, 'w')
    for p in protein_list:
        fWrite.write("%s\n" % (p))
    fWrite.close()


def write_fasta(fn, sequences):
    '''
    Takes in a list of sequences, writes in fasta format
    '''
    with open(fn, "w") as output_handle:
        for sequence in sequences:
            SeqIO.write(sequence, output_handle, "fasta")


def load_go_graph(fname):
    print("### Loading Gene Ontology (GO) graph...")
    go_graph = obonet.read_obo(fname)
    print(f"DEBUG: {go_graph}, and the number of nodes: {len(go_graph.nodes)}")
    return go_graph

def load_pdbs(sifts_fname):
    '''
    Reads the SIFTs annotation file 

    Reads the first two columns of SIFTSs file, retrieve the PDB entry and entity identifiers, combine then to obtain set of PDB chains

    Note:
    PDB chains are all in uppercase.
    Format: PDB Entry-Chain Identifer (Alphabet)
    '''
    pdb_chains = set()
    with gzip.open(sifts_fname, mode='rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip()
            pdb_chains.add(pdb + '-' + chain)
    return pdb_chains

def load_plant_ids(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            pdb_entries = set(line.strip().split(','))
    print(pdb_entries)
    return set(pdb_entries)

def read_sifts(fname, entries, go_graph):
    '''
    Inputs: 
    fname: SIFTs annotation file path 
    go_graph: graph object from reading the .obo file
    '''
    print ("### Loading SIFTS annotations...")
    pdb2go = {}
    go2info = {}

    with gzip.open(fname, mode='rt') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        next(reader, None)  # skip the headers
        for row in reader:
            pdb = row[0].strip().upper()
            chain = row[1].strip() # an alphabet char
            evidence = row[4].strip()
            go_id = row[5].strip()
            pdb_chain = pdb + '-' + chain
            if (pdb in entries) and (go_id in go_graph) and (go_id not in root_terms):
                if pdb_chain not in pdb2go:
                    pdb2go[pdb_chain] = {'goterms': [go_id], 'evidence': [evidence]}
                namespace = go_graph.nodes[go_id]['namespace']
                go_ids = nx.descendants(go_graph, go_id)
                go_ids.add(go_id)
                go_ids = go_ids.difference(root_terms)
                for go in go_ids:
                    pdb2go[pdb_chain]['goterms'].append(go)
                    pdb2go[pdb_chain]['evidence'].append(evidence)
                    name = go_graph.nodes[go]['name']
                    if go not in go2info:
                        go2info[go] = {'ont': namespace, 'goname': name, 'pdb_chains': set([pdb_chain])}
                    else:
                        go2info[go]['pdb_chains'].add(pdb_chain)
    return pdb2go, go2info

def write_output_files(fname, pdb2go, go2info, pdb2seq):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()
    for goterm in go2info:
        prots = go2info[goterm]['pdb_chains']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        #print(f"Goterm: {goterm}, Num: {num}, Namespace: {namespace}")
        if (num > 50) and (num <= 5000):
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    for ont in onts:
        print ("###", ont, ":", len(selected_goterms_list[ont]))

    sequences_list = []
    protein_list = []
    with open(fname + '_annot.tsv', 'wt', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for ont in onts:
            tsv_writer.writerow(["### GO-terms (%s)" % (ont)])
            tsv_writer.writerow(selected_goterms_list[ont])
            tsv_writer.writerow(["### GO-names (%s)" % (ont)])
            tsv_writer.writerow(selected_gonames_list[ont])
        tsv_writer.writerow(["### PDB-chain", "GO-terms (molecular_function)", "GO-terms (biological_process)", "GO-terms (cellular_component)"])
        for chain in selected_proteins:
            if chain in pdb2seq:
                goterms = set(pdb2go[chain]['goterms'])
                if len(goterms) > 2:
                    # selected goterms
                    mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                    bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                    cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                    if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                        sequences_list.append(SeqRecord(Seq(pdb2seq[chain], protein_letters_3to1), id=chain, description="nrPDB"))
                        protein_list.append(chain)
                        tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms)])
                    else:
                        print(f"Chain {chain} not found in pdb2seq.")

    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print ("Total number of annot nrPDB=%d" % (len(protein_list)))
    #print(f"Sample protein chain : {protein_list[0]}")

    # select test set based in 30% sequence identity
    test_list = set()
    test_sequences_list = []
    i = 0
    while len(test_list) < 5000 and i < len(protein_list):
        goterms = pdb2go[protein_list[i]]['goterms']
        evidence = pdb2go[protein_list[i]]['evidence']
        goterm2evidence = {goterms[i]: evidence[i] for i in range(len(goterms))}

        # selected goterms
        mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
        bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
        cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))

        mf_evidence = [goterm2evidence[goterm] for goterm in mf_goterms]
        mf_evidence = [1 if evid in exp_evidence_codes else 0 for evid in mf_evidence]

        bp_evidence = [goterm2evidence[goterm] for goterm in bp_goterms]
        bp_evidence = [1 if evid in exp_evidence_codes else 0 for evid in bp_evidence]

        cc_evidence = [goterm2evidence[goterm] for goterm in cc_goterms]
        cc_evidence = [1 if evid in exp_evidence_codes else 0 for evid in cc_evidence]

        if len(mf_goterms) > 0 and len(bp_goterms) > 0 and len(cc_goterms) > 0:
            if sum(mf_evidence) > 0 and sum(bp_evidence) > 0 and sum(cc_evidence) > 0:
                test_list.add(protein_list[i])
                test_sequences_list.append(SeqRecord(Seq(pdb2seq[protein_list[i]], protein_letters_3to1), id=protein_list[i], description="nrPDB_test"))
        i += 1

    print ("Total number of test nrPDB=%d" % (len(test_list)))

    protein_list = list(set(protein_list).difference(test_list))
    np.random.shuffle(protein_list)

    idx = int(0.8*len(protein_list))
    write_prot_list(test_list, fname + '_test.txt')
    write_prot_list(protein_list[:idx], fname + '_train.txt')
    write_prot_list(protein_list[idx:], fname + '_valid.txt')
    write_fasta(fname + '_sequences.fasta', sequences_list)
    write_fasta(fname + '_test_sequences.fasta', test_sequences_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sifts', type=str, default='./preprocessing/data/pdb_chain_go.tsv.gz', help="SIFTS annotation files.")
    parser.add_argument('-bc', type=str, default='./preprocessing/data/clusters-by-entity-100.txt', help="Blastclust of PDB chains.")
    parser.add_argument('-seqres', type=str, default='./preprocessing/data/pdb_seqres.txt.gz', help="PDB chain seqres fasta.")
    parser.add_argument('-obo', type=str, default='./preprocessing/data/go-basic.obo', help="Gene Ontology hierarchy.")
    parser.add_argument('-out', type=str, default='./preprocessing/data/nrPDB-GO', help="Output filename prefix.")
    parser.add_argument('-plant_chains', type=str, default='./preprocessing/data/plant_chains.csv')
    parser.add_argument('-plant_ids', type=str, default='./preprocessing/data/plant_ids.txt')
    args = parser.parse_args()


    annotated_plant_chains_2 = load_plant_ids(args.plant_ids)
    pdb2seq = read_fasta(args.seqres)
    go_graph = load_go_graph(args.obo)
    pdb2go, go2info = read_sifts(args.sifts, annotated_plant_chains_2, go_graph)
    write_output_files(args.out, pdb2go, go2info, pdb2seq) 

import json
import tempfile
import re
import Bio
from Bio import SeqIO
#from Bio.Data.SCOPData import protein_letters_3to1
from Bio.Data.IUPACData import protein_letters_3to1 


class PdbSeqResDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.sequences = []
        self.idx_to_chain = {}
        self.chain_count = 0

        for record in SeqIO.parse(handle, f"{parser_mode}-seqres"):
            if verbose:
                print("Record id %s, chain %s, len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                print(record.dbxrefs)
                print(record.seq)

            if record.annotations['chain'] == chain_name:
                self.sequences.append(record.seq)
                self.idx_to_chain[self.chain_count] = record.annotations['chain']
                self.chain_count += 1
                break

    def has_sequences(self):
        return self.chain_count > 0


class PdbAtomDataParser:
    def __init__(self, handle, parser_mode, chain_name, verbose=False):
        self.idx_to_chain = {}
        self.chain_to_idx = {}
        self.sequences = []
        self.chain_count = 0

        for record in SeqIO.parse(handle, f"{parser_mode}-atom"):
            if verbose:
                print("Record id %s, chain %s len %s" % (record.id, record.annotations["chain"], len(record.seq)))
                print(record.seq)

            if record.annotations['chain'] == chain_name:
                self.sequences.append(record.seq)
                self.idx_to_chain[self.chain_count] = record.annotations['chain']
                self.chain_to_idx[record.annotations['chain']] = self.chain_count
                self.chain_count += 1
                break


class StructureContainer:
    def __init__(self):
        self.structure = None
        self.chains = {}
        self.id_code = None

    def with_id_code(self, id_code):
        self.id_code = id_code
        return self

    def with_structure(self, structure):
        self.structure = structure
        return self

    def with_chain(self, chain_name, seqres_seq, atom_seq):
        chain_info = {'seqres-seq': seqres_seq, 'atom-seq': atom_seq}
        chain_info['seq'] = seqres_seq if seqres_seq is not None else atom_seq
        self.chains[chain_name] = chain_info
        return self

    def with_seqres(self, seqres_seq):
        for chain_name in self.chains:
            self.chains[chain_name]['seqres-seq'] = seqres_seq
        return self

    def to_json(self):
        result = {'chain_info': self.chains, 'id_code': self.id_code}
        return json.dumps(result, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4, skipkeys=True)


def build_structure_container_for_pdb(structure_data, chain_name):
    # Test the data to see if this looks like a PDB or an mmCIF
    tester = re.compile('^_', re.MULTILINE)
    if len(tester.findall(structure_data)) == 0:
        # PDB
        parser_mode = 'pdb'
    else:
        parser_mode = 'cif'

    temp = tempfile.TemporaryFile(mode='w+')
    temp.write(str(structure_data))
    temp.flush()
    temp.seek(0, 0)

    container_builder = StructureContainer()

    seq_res_info = PdbSeqResDataParser(temp, parser_mode, chain_name)
    temp.seek(0, 0)

    atom_info = PdbAtomDataParser(temp, parser_mode, chain_name)

    if len(seq_res_info.idx_to_chain) == len(atom_info.idx_to_chain) \
            and set(seq_res_info.idx_to_chain.values()) != set(atom_info.idx_to_chain.values()):
        print(f"WARNING: The IDs from the seqres lines don't match the IDs from the ATOM lines. This might not work.")
        raise Exception

    temp.seek(0, 0)
    if parser_mode == 'pdb':
        structure = Bio.PDB.PDBParser().get_structure('input', temp)
        id_code = structure.header['idcode']
        container_builder.with_id_code(id_code)
    elif parser_mode == 'cif':
        structure = Bio.PDB.MMCIFParser().get_structure('input', temp)
        id_code = None

    container_builder.with_structure(structure)

    if seq_res_info.has_sequences():
        for i, seqres_seq in enumerate(seq_res_info.sequences):
            chain_name_from_seqres = seq_res_info.idx_to_chain[i]
            try:
                chain_idx = atom_info.chain_to_idx[chain_name_from_seqres]
                atom_seq = atom_info.sequences[chain_idx]
            except (IndexError, KeyError) as e:
                print(f"Error processing {id_code}-{chain_name_from_seqres}: {e}")
                print(f"SEQRES seq: {seqres_seq}")
                print(f"Atom seq:   {atom_seq}")
                continue

            if len(seqres_seq) != len(atom_seq):
                print(f"Discontinuity found in chain {chain_name_from_seqres}")

            container_builder.with_chain(chain_name_from_seqres, seqres_seq, atom_seq)
    else:
        for i, atom_seq in enumerate(atom_info.sequences):
            chain_name_from_seqres = atom_info.idx_to_chain[i]
            container_builder.with_chain(chain_name_from_seqres, None, atom_seq)

    temp.close()
    return container_builder

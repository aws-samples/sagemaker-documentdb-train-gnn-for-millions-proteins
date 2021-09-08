# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Helpers for parsing protein structure files and generating contact maps.
"""
import gzip
import numpy as np
import pandas as pd
from io import StringIO
from Bio.PDB.Polypeptide import three_to_one, is_aa


def gunzip_to_ram(gzip_file_path):
    """
    gunzip a gzip file and decode it to a io.StringIO object.
    """
    content = []
    with gzip.open(gzip_file_path, "rb") as f:
        for line in f:
            content.append(line.decode("utf-8"))

    temp_fp = StringIO("".join(content))
    return temp_fp


def _parse_structure(parser, name, file_path):
    """Parse a .pdb or .cif file into a structure object.
    The file can be gzipped."""
    if pd.isnull(file_path):
        return None
    if file_path.endswith(".gz"):
        structure = parser.get_structure(name, gunzip_to_ram(file_path))
    else:  # not gzipped
        structure = parser.get_structure(name, file_path)
    return structure


parse_pdb_structure = _parse_structure  # for backward compatiblity


def parse_structure(pdb_parser, cif_parser, name, file_path):
    """Parse a .pdb file or .cif file into a structure object.
    The file can be gzipped."""
    if file_path.rstrip(".gz").endswith("pdb"):
        return _parse_structure(pdb_parser, name, file_path)
    else:
        return _parse_structure(cif_parser, name, file_path)


def three_to_one_standard(res):
    """Encode non-standard AA to X."""
    if not is_aa(res, standard=True):
        return "X"
    return three_to_one(res)


def is_aa_by_target_atoms(res):
    """Tell if a Residue object is AA"""
    target_atoms = ["N", "CA", "C", "O"]
    for atom in target_atoms:
        try:
            res[atom]
        except KeyError:
            return False
    return True


def get_atom_coords(residue, target_atoms=["N", "CA", "C", "O"]):
    """Extract the coordinates of the target_atoms from an amino acid residue.
    Handles exception where residue doesn't contain certain atoms
    """
    atom_coords = []
    for atom in target_atoms:
        try:
            coord = residue[atom].coord
        except KeyError:
            coord = [np.nan] * 3
        atom_coords.append(coord)
    return np.asarray(atom_coords)


def chain_to_coords(chain, target_atoms=["N", "CA", "C", "O"], name=""):
    """Convert a protein chain in a PDB file to coordinates of target atoms from all
    residues"""
    output = {}
    # get AA sequence in the pdb structure
    pdb_seq = "".join(
        [
            three_to_one_standard(res.get_resname())
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    if len(pdb_seq) <= 1:
        # has no or only 1 AA in the chain
        return None
    output["seq"] = pdb_seq
    # get the atom coords
    coords = np.asarray(
        [
            get_atom_coords(res, target_atoms=target_atoms)
            for res in chain.get_residues()
            if is_aa(res)
        ]
    )
    output["coords"] = coords.tolist()
    output["name"] = "{}-{}".format(name, chain.id)
    return output


def parse_pdb_file_to_json_record(pdb_parser, pdb_file_path, name=""):
    """Parse a protein structure file (.pdb) to extract all the chains
    to json records."""

    try:
        struct = parse_pdb_structure(pdb_parser, name, pdb_file_path)
    except Exception as e:
        print(pdb_file_path, "raised an error:")
        print(e)
        return []
    else:
        records = []
        chain_ids = set()
        for chain in struct.get_chains():
            if chain.id in chain_ids:  # skip duplicated chains
                continue
            chain_ids.add(chain.id)
            record = chain_to_coords(chain, name=name)
            if record is not None:
                records.append(record)
        return records

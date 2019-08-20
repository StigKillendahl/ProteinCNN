# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import glob
import os.path
import os
import numpy as np
import h5py
from util import AA_ID_DICT, calculate_dihedral_angels, protein_id_to_str, get_structure_from_angles, \
    structure_to_backbone_atoms, write_to_pdb, calculate_dihedral_angles_over_minibatch, \
    get_backbone_positions_from_angular_prediction, encode_primary_string
import torch

MAX_SEQUENCE_LENGTH = 2000
max_proteins = None

def process_raw_data(use_gpu, n_proteins=None, max_sequence_length=2000, force_pre_processing_overwrite=True):
    global MAX_SEQUENCE_LENGTH, max_proteins
    MAX_SEQUENCE_LENGTH = max_sequence_length
    max_proteins = n_proteins

    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:

        if os.name == 'nt':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]
        
        preprocessed_file_name = "data/preprocessed/" + filename
        preprocessed_file_name += "_"+str(max_sequence_length)+".hdf5"
        
        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, use_gpu)
    print("Completed pre-processing.")

def read_protein_from_file(file_pointer):
        dict_ = {}
        _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
        _mask_dict = {'-': 0, '+': 1}

        while True:
            next_line = file_pointer.readline()
            if next_line == '[ID]\n':
                id_ = file_pointer.readline()[:-1]
                dict_.update({'id': id_})
            elif next_line == '[PRIMARY]\n':
                primary = encode_primary_string(file_pointer.readline()[:-1])
                dict_.update({'primary': primary})
            elif next_line == '[EVOLUTIONARY]\n':
                evolutionary = []
                for residue in range(21): evolutionary.append(
                    [float(step) for step in file_pointer.readline().split()])
                dict_.update({'evolutionary': evolutionary})
            elif next_line == '[SECONDARY]\n':
                secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
                dict_.update({'secondary': secondary})
            elif next_line == '[TERTIARY]\n':
                tertiary = []
                # 3 dimension
                for axis in range(3): tertiary.append(
                    [float(coord) for coord in file_pointer.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif next_line == '[MASK]\n':
                mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
                dict_.update({'mask': mask})
            elif next_line == '\n':
                return dict_
            elif next_line == '':
                return None


def process_file(input_file, output_file, use_gpu):
    print("Processing raw data file", input_file)
    current_buffer_size = 1
    current_buffer_allocaton = 0
    error_c = 0
    proteins_dropped = 0
    error_ids = []

    # Create output file and datasets to store
    f = h5py.File(output_file, 'w')
    dt = h5py.special_dtype(vlen=bytes)
    dset1 = f.create_dataset('primary',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='int32')
    dset2 = f.create_dataset('tertiary',(current_buffer_size,MAX_SEQUENCE_LENGTH,9),maxshape=(None,MAX_SEQUENCE_LENGTH, 9),dtype='float')
    dset3 = f.create_dataset('mask',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='uint8')
    dset4 = f.create_dataset('evolutionary',(current_buffer_size,MAX_SEQUENCE_LENGTH,21),maxshape=(None,MAX_SEQUENCE_LENGTH, 21),dtype='float')
    dset5 = f.create_dataset('id', (current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None, MAX_SEQUENCE_LENGTH),dtype=dt)
    dset6 = f.create_dataset('seq_length', (current_buffer_size,1),maxshape=(None,1),dtype='uint32')
    input_file_pointer = open("data/raw/" + input_file, "r")

    while True: 
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)

        if max_proteins is not None and current_buffer_allocaton >= max_proteins:
            break
        if next_protein is None:
            break

        p_id = next_protein['id']
        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            proteins_dropped += 1
            continue

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        evolutionary_padded = np.zeros((21, MAX_SEQUENCE_LENGTH))

        # Handle primary sequence
        primary_padded[:sequence_length] = next_protein['primary']
        prim = primary_padded

        # Handle evolutionary (PSSM)
        ev_transposed = np.ravel(np.array(next_protein['evolutionary']).T)
        ev_reshaped = np.reshape(ev_transposed, (sequence_length,21)).T
        evolutionary_padded[:,:sequence_length] = ev_reshaped
        evo = torch.Tensor(evolutionary_padded).view(21,-1).transpose(0,1)

        # Handle tertiary sttructure
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length,9)).T


        tertiary_padded[:,:sequence_length] = t_reshaped
        pos = torch.Tensor(tertiary_padded).view(9,-1).transpose(0,1) / 100

        # Handle mask
        mask_padded[:sequence_length] = next_protein['mask']
        mask = mask_padded

        check = [not tertiary_padded[:3,i].any() for i in np.where(mask == 1)[0]]

        if True in check:
            error_c += 1
            continue

        if current_buffer_allocaton >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size,MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset4.resize((current_buffer_size,MAX_SEQUENCE_LENGTH, 21))
            dset5.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset6.resize((current_buffer_size,1))
            
        dset1[current_buffer_allocaton] = prim
        dset2[current_buffer_allocaton] = pos.numpy()
        dset3[current_buffer_allocaton] = mask
        dset4[current_buffer_allocaton] = evo.numpy()
        dset5[current_buffer_allocaton] = p_id
        dset6[current_buffer_allocaton] = sequence_length
        current_buffer_allocaton += 1

    print("Wrote output to", current_buffer_allocaton, "proteins to", output_file)
    if error_c > 0:
        # errfile = open("data/preprocessed/"+input_file+"errors.txt", "a")
        # for e_id in error_ids:
        #     errfile.write(e_id)
        print("Among file", error_c, "proteins did not process correctly and were discarded")
        #print("All ID's of discarded proteins have been written to data/preprocessed/" + input_file + "errors.txt")
        #errfile.close()
    if proteins_dropped > 0:
        print("Dropped", proteins_dropped, "proteins as length too long")


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))
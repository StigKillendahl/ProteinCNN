# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import glob
from numpy import genfromtxt
import numpy as np
import torch
import os
from util import *
from dashboard import start_dashboard_server
import requests
import PResNet

from models import *
from cnn_models import *

def run():
    print('Starting server...')
    start_dashboard_server()
    time.sleep(5)

    data_file = "data/preprocessed/validation_100.hdf5"

    protein_loader = contruct_dataloader_from_disk(data_file, 1, with_id=True)
    dataset_size = protein_loader.dataset.__len__()

    use_gpu = False
    if torch.cuda.is_available():
        write_out("CUDA is available, using GPU")
        use_gpu = True

    drmsd_total = 0
    p_id_to_structure = parse_tertiary()
    for minibatch_id, training_minibatch in enumerate(protein_loader, 0):

        print("Predicting next protein")

        primary_sequence, tertiary_positions, mask, p_ids = training_minibatch

        p_id = str(p_ids[0][0],'utf-8')
        predicted_pos = p_id_to_structure[p_id]
        predicted_pos = torch.Tensor(predicted_pos)
        predicted_pos = predicted_pos / 100 # to angstrom units
        pos = tertiary_positions[0]
        mask = mask[0][:len(predicted_pos)]
        pos = apply_mask(pos, mask)
        predicted_pos = predicted_pos[mask.nonzero()].squeeze(dim=1)
        primary_sequence = torch.masked_select(primary_sequence[0], mask)
        angles = calculate_dihedral_angels(pos, use_gpu)
        angles_pred = calculate_dihedral_angels(predicted_pos, use_gpu)

        # predicted_structure = get_structure_from_angles(primary_sequence[0], angles_pred)
        # actual_structure = get_structure_from_angles(primary_sequence[0], angles)
        write_to_pdb(get_structure_from_angles(primary_sequence, angles), "actual")
        write_to_pdb(get_structure_from_angles(primary_sequence, angles_pred), "predicted")

        pos = pos.contiguous().view(-1,3)
        predicted_pos = predicted_pos.contiguous().view(-1,3)


        drmsd = calc_drmsd(pos, predicted_pos, use_gpu).item()

        drmsd_total += drmsd
        print("DRMSD:", drmsd )

        data = {}
        data["pdb_data_pred"] = open("output/protein_predicted.pdb","r").read()
        data["pdb_data_true"] = open("output/protein_actual.pdb","r").read()
        data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:,1]])
        data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1,2]])
        data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:,1]])
        data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1,2]])
        data["validation_dataset_size"] = dataset_size
        data["sample_num"] = [0]
        data["train_loss_values"] = [0]
        data["validation_loss_values"] = [0]
        data["drmsd_avg"] = [drmsd]
        data["rmsd_avg"] = [0]
        res = requests.post('http://localhost:5000/graph', json=data)
        if res.ok:
            print(res.json())
        input("Press Enter to continue...")

    print(drmsd_total/dataset_size)

def parse_tertiary():
    files = glob.glob('output/runs/rundata/outputsValidation/*.tertiary')
    protein_to_structure = dict()

    for fp in files:
        

        p_id = fp.split('/')[-1].split('.')[0]
        if os.name == 'nt':
            p_id = p_id.split('\\')[1]
        xs, ys, zs = genfromtxt(fp,delimiter=' ',skip_header=2)
        structure = np.array([i for xyzs in zip(xs, ys, zs) for i in xyzs]).reshape((-1,9))
        protein_to_structure[p_id] = structure
    
    return protein_to_structure

run()

# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import torch
from util import *
from dashboard import start_dashboard_server
import requests
import PResNet
from tmscore.score import TMscore
from models import *
from cnn_models import *

# torch.manual_seed(12)

model_name = "cnn_200_50_long"
data_file = "testing_200"

show_ui=True
quick_average=False
use_tm = False

if show_ui:
    print('Starting server...')
    start_dashboard_server()
    time.sleep(5)

data_file = "data/preprocessed/"+data_file+".hdf5"
model_path = "output/models/" + model_name + ".model"

protein_loader = contruct_dataloader_from_disk(data_file, 64)

model = torch.load(model_path, map_location='cpu')

model.use_gpu = False
model.model.use_gpu = False

model = model.eval()
dataset_size = protein_loader.dataset.__len__()

#Set all variables needed for stats.
drmsd_total = drmsd_fm = drmsd_tbm = \
    tm_counter = fm_counter = tbm_counter = \
    tm_total = tm_fm = tm_tbm = 0

predictions = []

for minibatch_id, training_minibatch in enumerate(protein_loader, 0):

    primary_sequence, tertiary_positions, mask, p_id, evolutionary = training_minibatch
    input_sequence, batch_sizes = one_hot_encode(primary_sequence, 21, False)
    evolutionary, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.pack_sequence(evolutionary))
    input_sequence = torch.cat((input_sequence, evolutionary.view(-1, len(batch_sizes), 21)), 2)
    predicted_dihedral_angles, predicted_backbone_atoms, batch_sizes = model((input_sequence, batch_sizes))
    cpu_predicted_angles = predicted_dihedral_angles.transpose(0,1).cpu().detach()
    cpu_predicted_backbone_atoms = predicted_backbone_atoms.transpose(0,1).cpu().detach()
    batch_data = list(zip(primary_sequence, p_id, tertiary_positions, cpu_predicted_backbone_atoms, cpu_predicted_angles, mask))

    for primary_sequence, pid, pos, predicted_pos, predicted_angles, mask in batch_data:
        pos = apply_mask(pos, mask)
        predicted_pos = apply_mask(predicted_pos[:len(primary_sequence)], mask)

        pid = str(pid[0],'utf-8')
        angles = calculate_dihedral_angels(pos, False)

        angles_pred = apply_mask(predicted_angles[:len(primary_sequence)], mask, size=3)
        angles_pred = calculate_dihedral_angels(predicted_pos,False)

        primary_sequence = torch.masked_select(primary_sequence, mask)

        predicted_pos = predicted_pos.contiguous().view(-1,3)

        pos = pos.contiguous().view(-1,3)

        drmsd = calc_drmsd(predicted_pos, pos, 'c_alpha', False).item()

        tm_score = 0

        if (not torch.isnan(angles).any()) and use_tm:
            write_to_pdb(get_structure_from_angles(primary_sequence, angles), "actual")
            write_to_pdb(get_structure_from_angles(primary_sequence, angles_pred), "predicted")
            tmscore = TMscore('tmscore/./TMscore')
            tmscore("output/protein_actual.pdb", "output/protein_predicted.pdb")
            tm_score = tmscore.get_tm_score()

        predictions.append((pid, drmsd, tm_score, primary_sequence, angles, angles_pred))

predictions = sorted(predictions, key=lambda x: x[1])

for protein_id, drmsd, tm_score, primary, angles, angles_pred in predictions:
    
    print("Prediction for protein:", protein_id)
    
    tm_total += tm_score

    if pid.startswith('FM'):
        print('Free modeling prediction, dRMSD:', drmsd)
        drmsd_fm += drmsd
        fm_counter += 1
        if use_tm:
            tm_fm += tm_score
            print("TM-SCORE:",tm_score)

    elif pid.startswith('TBM'):
        print('Template Based Model prediction, dRMSD:', drmsd)
        drmsd_tbm += drmsd
        tbm_counter += 1
        if use_tm:
            print("TM-SCORE:",tm_score)
            tm_tbm += tm_score
    else:
        print("TM-SCORE:",tm_score)
        print("DRMSD:", drmsd)

    if show_ui:
        write_to_pdb(get_structure_from_angles(primary, angles), "actual")
        write_to_pdb(get_structure_from_angles(primary, angles_pred), "predicted")
        data = {}
        data["pdb_data_pred"] = open("output/protein_actual.pdb","r").read()
        data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:,1]])
        data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1,2]])
        data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:,0]])
        data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1,1]])
        data["validation_dataset_size"] = dataset_size
        data["sample_num"] = [0]
        data["train_loss_values"] = [0]
        data["validation_loss_values"] = [0]
        data["drmsd_avg"] = [drmsd]
        data["rmsd_avg"] = [0]
    
        res = requests.post('http://localhost:5000/graph', json=data)
        if res.ok:
            print(res.json())
        input("Press Enter for prediction...")
        data["pdb_data_pred"] = open("output/protein_predicted.pdb","r").read()

        res = requests.post('http://localhost:5000/graph', json=data)
        if res.ok:
            print(res.json())
        input("Press Enter for next protein...")




    drmsd_total += drmsd
    if quick_average and not show_ui:
        print("Evaluating next prediction")
    else:
        input("Press Enter for next prediction")


if fm_counter > 0:
    print('---Results from free-modeling---')
    print('Average dRMSD:', drmsd_fm / fm_counter)
    if use_tm:
        print("Average TM-score:", tm_fm/ fm_counter)


if tbm_counter > 0:
    print('---Results from TBM---')
    print('Average dRMSD:', drmsd_tbm / tbm_counter)
    if use_tm:
        print("Average TM-score:", tm_tbm/ tbm_counter)

    

print('---Overall results---')
print("Average drmsd:", drmsd_total / dataset_size)
if use_tm:
    print("Average TM-score:", tm_total / dataset_size)

print("No more proteins in file")




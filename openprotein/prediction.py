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

if __name__ == "__main__":
    model_name = "resnet_18_200_17k_2019-04-11_09_56_58-TRAIN-LR0_0001-MB32_val"
    data_file = "validation_200"

    predict(model_name, data_file, use_gpu=torch.cuda.is_available, batch_size=32, show_ui=True)


def predict(model_name, prediction_file, use_gpu=False, batch_size=32, show_ui=True, loss_atoms="c_alpha"):

    if show_ui:
        print('Starting server...')
        start_dashboard_server()
        time.sleep(5)

    data_file = "data/preprocessed/"+prediction_file+".hdf5"
    model_path = "output/models/" + model_name + ".model"



    portein_loader = contruct_dataloader_from_disk(data_file, batch_size)

    if torch.cuda.is_available():
        write_out("CUDA is available, using GPU")
        model = torch.load(model_path)
        use_gpu = True

    else:
        model = torch.load(model_path, map_location='cpu')


    model.use_gpu = use_gpu
    model.model.use_gpu = use_gpu



    model = model.eval()

    dataset_size = portein_loader.dataset.__len__()

    #Set all variables needed for stats.
    drmsd_total = drmsd_fm = drmsd_tbm = \
        tm_counter = fm_counter = tbm_counter = \
        tm_total = tm_fm = tm_tbm = 0

    use_tm = False
    for minibatch_id, training_minibatch in enumerate(portein_loader, 0):

        print("Predicting next minibatch of size:", batch_size)

        primary_sequence, tertiary_positions, mask, p_id, evolutionary = training_minibatch

        #pos = tertiary_positions[0]
        #One Hot encode amino string and concate PSSM values.
        input_sequence, batch_sizes = one_hot_encode(primary_sequence, 21, use_gpu)

        
        evolutionary, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.pack_sequence(evolutionary))
        
        if use_gpu:
            evolutionary = evolutionary.cuda()

        input_sequence = torch.cat((input_sequence, evolutionary.view(-1, len(batch_sizes), 21)), 2)
        
        predicted_dihedral_angles, predicted_backbone_atoms, batch_sizes = model((input_sequence, batch_sizes))

        cpu_predicted_angles = predicted_dihedral_angles.transpose(0,1).cpu().detach()
        cpu_predicted_backbone_atoms = predicted_backbone_atoms.transpose(0,1).cpu().detach()

        batch_data = list(zip(primary_sequence, p_id, tertiary_positions, cpu_predicted_backbone_atoms, cpu_predicted_angles, mask))

        for primary_sequence, pid, pos, predicted_pos, predicted_angles, mask in batch_data:
            pos = apply_mask(pos, mask)
            predicted_pos = apply_mask(predicted_pos[:len(primary_sequence)], mask)

            pid = str(pid[0],'utf-8')
            angles = calculate_dihedral_angels(pos, use_gpu)

            angles_pred = apply_mask(predicted_angles[:len(primary_sequence)], mask, size=3)

            primary_sequence = torch.masked_select(primary_sequence, mask)
            #angles_pred = calculate_dihedral_angels(predicted_pos, use_gpu)
            if show_ui:
                write_to_pdb(get_structure_from_angles(primary_sequence, angles), "actual")
                write_to_pdb(get_structure_from_angles(primary_sequence, angles_pred), "predicted")

            if (not torch.isnan(angles).any()) and use_tm:
                print('TM scores:')
                tmscore = TMscore('tmscore/./TMscore')
                tmscore("output/protein_actual.pdb", "output/protein_predicted.pdb")
                tmscore.print_info()
                print('--- End scores ---')
                tm_counter += 1
                tm_total += tmscore.get_tm_score()
            # predicted_structure = get_structure_from_angles(primary_sequence, angles_pred)
            # actual_structure = get_structure_from_angles(primary_sequence, angles)


            predicted_pos = predicted_pos.contiguous().view(-1,3)

            pos = pos.contiguous().view(-1,3)


            drmsd = calc_drmsd(predicted_pos, pos, loss_atoms, use_gpu).item()

            if pid.startswith('FM'):
                print('Free modeling prediction, dRMSD:', drmsd)
                drmsd_fm += drmsd
                fm_counter += 1
                if use_tm:
                    tm_fm += tmscore.get_tm_score()
            elif pid.startswith('TBM'):
                print('Template Based Model prediction, dRMSD:', drmsd)
                drmsd_tbm += drmsd
                tbm_counter += 1
                if use_tm:
                    tm_tbm += tmscore.get_tm_score()
            else:
                print("DRMSD:", drmsd )

            if show_ui:
                data = {}
                data["pdb_data_pred"] = open("output/protein_predicted.pdb","r").read()
                data["pdb_data_true"] = open("output/protein_actual.pdb","r").read()
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
                input("Press Enter to continue...")

            drmsd_total += drmsd
            print("Evaluating next prediction")


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
        print("Average TM-score:", tm_total/ tm_counter)
    
    print("No more proteins in file")




# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

from preprocessing import process_raw_data
import torch
import torch.utils.data
import h5py
import torch.autograd as autograd
import torch.optim as optim
import argparse
import numpy as np
import time
import requests
import PResNet
import prediction
import math
import os
import subprocess
from config import RunConfig
from modelsummary import summary
from dashboard import start_dashboard_server
from glob import glob

from models import *
from cnn_models import *
from util import contruct_dataloader_from_disk, set_experiment_id, write_out, \
    evaluate_model, write_model_to_disk, write_result_summary, write_to_pdb, calculate_dihedral_angels, \
    get_structure_from_angles, protein_id_to_str

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

print("------------------------")
print("--- ProteinCNN v1.0 ---")
print("------------------------")

parser = argparse.ArgumentParser(description = "ProteinCNN version 0.2")
parser.add_argument('--config', dest='config', default='example', help='Name of the config file from the configurations folder.')
parser.add_argument('--model', dest='model', default=None, help='Specify name of existing model for continous training or prediction. \
    training will continure based on config file. model must be in the ./output/models/ folder.')
parser.add_argument('--prediction', dest='prediction', default=None, help='Specify that prediction is done on the data file given. \
    If no model is given using --model, the latest created model is used.')
parser.add_argument('--plotruns', dest='plotruns', type=str2bool, default=False, help='plot all runs in the ./output/runs/ folder.')
parser.add_argument('--show_ui', dest='show_ui', type=str2bool, default=False)
args, unknown = parser.parse_known_args()

if args.plotruns:
    subprocess.Popen('python plotruns.py', shell=True) # TODO: Probably shouldn't be a shell call?
    exit()

use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True

if args.prediction is not None:
    if args.model is not None:
        model_name = args.model
    else:
        models = glob('output/models/*.model')
        models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if os.name == 'nt':
            model_name = models[0].split('\\')[1].split('.')[0]
        else:
            model_name = models[0].replace('output/models/', '').split('.')[0]

    data_file = args.prediction
    prediction.predict(model_name, data_file, use_gpu=use_gpu, show_ui=args.show_ui)
    exit()

#Retrieve config file and create interval config dictionary.
config_file = "configurations/" + args.config + ".config"
write_out('Using config: %s' % config_file)
configs = RunConfig(config_file)

if configs.run_params["hide_ui"]:
    write_out("Live plot deactivated, see output folder for plot.")

max_seq_length, use_evolutionary, n_proteins = configs.run_params["max_sequence_length"], configs.run_params["use_evolutionary"], configs.run_params["n_proteins"]


# start web server
if not configs.run_params["hide_ui"]:
    start_dashboard_server()


process_raw_data(use_gpu, n_proteins=n_proteins, max_sequence_length=max_seq_length, force_pre_processing_overwrite=False)

datafolder = "data/preprocessed/"
training_file = datafolder + configs.run_params["training_file"] + "_" + str(max_seq_length) + ".hdf5"
validation_file = datafolder + configs.run_params["validation_file"] + "_" + str(max_seq_length) + ".hdf5"
testing_file = datafolder + configs.run_params["testing_file"] + "_" + str(max_seq_length) + ".hdf5"

def train_model(data_set_identifier, train_file, val_file, learning_rate, minibatch_size, name):
    set_experiment_id(data_set_identifier, learning_rate, minibatch_size, name)

    train_loader = contruct_dataloader_from_disk(train_file, minibatch_size, use_evolutionary=True)
    validation_loader = contruct_dataloader_from_disk(val_file, minibatch_size, use_evolutionary=True)
    validation_dataset_size = validation_loader.dataset.__len__()
    train_dataset_size = train_loader.dataset.__len__()



    embedding_size = 21
    if configs.run_params["use_evolutionary"]:
        embedding_size = 42


    #Load in existing model if given as argument
    if args.model is not None:
        model_path = "output/models/" + args.model + ".model"
        model = load_model_from_disk(model_path, use_gpu)
    else:
    #else construct new model from config file
        model = construct_model(configs.model_params, embedding_size, use_gpu,minibatch_size)
    
    #optimizer parameters
    betas = tuple(configs.run_params["betas"])
    weight_decay = configs.run_params["weight_decay"]
    angle_lr = configs.run_params["angles_lr"]

    if configs.model_params['architecture'] == 'cnn_angles':
        optimizer = optim.Adam(model.parameters(), betas=betas, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam([
            {'params' : model.model.parameters(), 'lr':learning_rate},
            {'params' : model.soft_to_angle.parameters(), 'lr':angle_lr}], betas=betas, weight_decay=weight_decay)
    
    #print number of trainable parameters
    print_number_of_parameters(model)
    #For creating a summary table of the model (does not work on ExampleModel!)
    if configs.run_params["print_model_summary"]:
        if configs.model_params["architecture"] != 'rnn':
            summary(model, configs.run_params["max_sequence_length"], 2)
        else:
            write_out("DETAILED MODEL SUMMARY IS NOT SUPPORTED FOR RNN MODELS")
    
    if use_gpu:
        model = model.cuda()

    # TODO: is soft_to_angle.parameters() included here?

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()
    rmsd_avg_values = list()
    drmsd_avg_values = list()
    break_point_values = list()

    breakpoints = configs.run_params['breakpoints']
    best_model_loss = 1e20
    best_model_train_loss = 1e20
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    loss_atoms = configs.run_params["loss_atoms"]
    start_time = time.time()
    max_time = configs.run_params["max_time"]
    C_epochs = configs.run_params["c_epochs"] # TODO: Change to parameter
    C_batch_updates = C_epochs

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)
        start_time_n_minibatches = time.time()
        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            training_minibatch = list(training_minibatch)
            primary_sequence, tertiary_positions, mask, p_id = training_minibatch[:-1]
            # Update C
            C = 1.0 if minibatches_proccesed >= C_batch_updates else float(minibatches_proccesed) / C_batch_updates

            #One Hot encode amino string and concate PSSM values.
            amino_acids, batch_sizes = one_hot_encode(primary_sequence, 21, use_gpu)

            if configs.run_params["use_evolutionary"]:
                evolutionary = training_minibatch[-1]

                evolutionary, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.pack_sequence(evolutionary))
                
                if use_gpu:
                    evolutionary = evolutionary.cuda()

                amino_acids = torch.cat((amino_acids, evolutionary.view(-1, len(batch_sizes) , 21)), 2)

            start_compute_loss = time.time()

            if configs.run_params["only_angular_loss"]:
                #raise NotImplementedError("Only_angular_loss function has not been implemented correctly yet.")
                loss = model.compute_angular_loss((amino_acids, batch_sizes), tertiary_positions, mask)
            else:
                loss = model.compute_loss((amino_acids, batch_sizes), tertiary_positions, mask, C=C, loss_atoms=loss_atoms)
            
            if C != 1:
                write_out("C:", C)
            write_out("Train loss:", float(loss))
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, float(loss))
            end = time.time()
            write_out("Loss time:", start_compute_grad-start_compute_loss, "Grad time:", end-start_compute_grad)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            # for every eval_interval samples, plot performance on the validation set
            if minibatches_proccesed % configs.run_params["eval_interval"] == 0:
                model.eval()
                write_out("Testing model on validation set...")
                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, data_total, rmsd_avg, drmsd_avg = evaluate_model(validation_loader,
                     model, use_gpu, loss_atoms, configs.run_params["use_evolutionary"])
                prim = data_total[0][0]
                pos = data_total[0][1]
                pos_pred = data_total[0][3]
                mask = data_total[0][4]
                pos = apply_mask(pos, mask)
                angles_pred = data_total[0][2]

                angles_pred = apply_mask(angles_pred, mask, size=3)

                pos_pred = apply_mask(pos_pred, mask)
                prim = torch.masked_select(prim, mask)

                if use_gpu:
                    pos = pos.cuda()
                    pos_pred = pos_pred.cuda()

                angles = calculate_dihedral_angels(pos, use_gpu)
                #angles_pred = calculate_dihedral_angels(pos_pred, use_gpu)
                #angles_pred = data_total[0][2] # Use angles output from model - calculate_dihedral_angels(pos_pred, use_gpu)

                write_to_pdb(get_structure_from_angles(prim, angles), "test")
                write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")
                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)

                if train_loss < best_model_train_loss:
                    best_model_train_loss = train_loss
                    best_model_train_path = write_model_to_disk(model, model_type="train")

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
                write_out("Best model so far (validation loss): ", best_model_loss, "at time", best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Best model train stored at " + best_model_train_path)
                write_out("Minibatches processed:",minibatches_proccesed)

                end_time_n_minibatches = time.time()
                n_minibatches_time_used = end_time_n_minibatches - start_time_n_minibatches
                minibatches_left = configs.run_params["max_updates"] - minibatches_proccesed
                seconds_left = int(n_minibatches_time_used * (minibatches_left/configs.run_params["eval_interval"]))
                
                m, s = divmod(seconds_left, 60)
                h, m = divmod(m, 60)
                write_out("Estimated time until maximum number of updates:", '{:d}:{:02d}:{:02d}'.format(h, m, s) )
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(train_loss)
                validation_loss_values.append(validation_loss)
                rmsd_avg_values.append(rmsd_avg)
                drmsd_avg_values.append(drmsd_avg)
                
                if breakpoints and minibatches_proccesed > breakpoints[0]:
                    break_point_values.append(drmsd_avg)
                    breakpoints = breakpoints[1:]

                data = {}
                data["pdb_data_pred"] = open("output/protein_test_pred.pdb","r").read()
                data["pdb_data_true"] = open("output/protein_test.pdb","r").read()
                data["validation_dataset_size"] = validation_dataset_size
                data["sample_num"] = sample_num
                data["train_loss_values"] = train_loss_values
                data["break_point_values"] = break_point_values
                data["validation_loss_values"] = validation_loss_values
                data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:,1]])
                data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1,2]])
                data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:,1]])
                data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1,2]])
                data["drmsd_avg"] = drmsd_avg_values
                data["rmsd_avg"] = rmsd_avg_values
                if not configs.run_params["hide_ui"]:
                    res = requests.post('http://localhost:5000/graph', json=data)
                    if res.ok:
                        print(res.json())
                
                # Save run data
                write_run_to_disk(data)

                #Check if maximum time is reached.
                start_time_n_minibatches = time.time()
                time_used = time.time() - start_time

                time_condition = (max_time is not None and time_used > max_time)
                max_update_condition = minibatches_proccesed >= configs.run_params["max_updates"]
                min_update_condition = (minibatches_proccesed > configs.run_params["min_updates"] and minibatches_proccesed > best_model_minibatch_time * 2)

                model.train()
                #Checking for stop conditions
                if time_condition or max_update_condition or min_update_condition:
                    stopping_condition_met = True
                    break
    write_out("Best validation model found after" , best_model_minibatch_time , "minibatches.")
    write_result_summary(best_model_loss)
    return best_model_path

def print_number_of_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: %i' % pytorch_total_params)


def construct_model(model_parameters, embedding_size, use_gpu, minibatch_size):
    model_type = model_parameters["architecture"]
    mixture_size = model_parameters["output_size"]
    dropout = model_parameters["dropout"]
    model = None
    soft_max_to_angle = models.soft_to_angle(mixture_size)
    if model_type == "rnn":
        model = ExampleModel(embedding_size, minibatch_size, use_gpu, dropout=dropout, mixture_size=mixture_size, hidden_size=model_parameters["hidden_size"])
    
    elif model_type == "cnn" or model_type == "cnn_angles":
        num_layers = model_parameters["layers"]
        channels = [embedding_size] + model_parameters["channels"][:num_layers-1] + [mixture_size] 
        kernels = model_parameters["kernel"] * num_layers
        paddings = model_parameters["padding"] * num_layers
        stride = model_parameters["stride"] * num_layers
        dilation = model_parameters["dilation"] * num_layers
        spatial_dropout = model_parameters["spatial_dropout"]
        layers = []
        for i in range(num_layers):
            params = (channels[i], channels[i+1], kernels[i], paddings[i], stride[i], dilation[i])
            layers.append(params)
        if model_type == "cnn_angles":
            soft_max_to_angle = None
            model = CNNBaseModelAngles(embedding_size, layers, minibatch_size, use_gpu, mixture_size=mixture_size)
        else:
            model = CNNBaseModel(embedding_size, layers, minibatch_size, use_gpu, dropout=dropout, mixture_size=mixture_size, spatial_dropout=spatial_dropout)
    
    elif model_type == "resnet":
        resnet_type = model_parameters["resnet_type"]
        kernel = model_parameters["kernel"]
        padding = model_parameters["padding"]
        stride = model_parameters["stride"]
        droprate = model_parameters["dropout"] * 5

        parameters = {
            "input_channels":embedding_size,
            "out_channels":mixture_size, 
            "kernel": kernel,
            "padding":padding,
            "stride":stride,
            "use_gpu":use_gpu,
            "droprate":droprate
        }

        model_func = PResNet.name_dict.get(resnet_type, None)
        if model_func is None:
            write_out('RESNET TYPE NOT SUPPORTED PLEASE USE SUPPORTED TYPE [resnet18,resnet34,resnet50,restnet101,resnet152] BY SPECIFYING "resnet_type" IN CONFIG FILE')
            exit()
        
        model = model_func(**parameters)
    else:
        write_out("MODEL TYPE NOT RECOGNICED PLEASE USE A SUPPORTED ARCHITECTURE IN CONFIG FILE [cnn,cnn_angles,resnet,rnn]")
        exit()
    return openprotein.BaseModel(use_gpu, mixture_size, model, soft_max_to_angle)


stop_restarts = False
max_restarts = configs.run_params['max_restarts']
restarts = 0
start_time = time.time()
while not stop_restarts:
    run_name = configs.run_params['name'] + '_' + str(restarts) if restarts != 0 else configs.run_params['name']

    train_model_path = train_model("TRAIN", training_file, validation_file, configs.run_params["learning_rate"], configs.run_params["minibatch_size"], run_name)
    restarts += 1

    if restarts >= max_restarts:
        stop_restarts = True



print(train_model_path)

# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

import torch.nn as nn
from util import *
import torch.nn.utils.rnn as rnn_utils
import time
import numpy as np
import openprotein
import models

class CNNBaseModel(nn.Module):

    def __init__(self, embedding_size, layer_parameters, minibatch_size, use_gpu, dropout=[0.0,0.0], mixture_size=500, spatial_dropout=False):
        super(CNNBaseModel, self).__init__()
        self.use_input_dropout = dropout[0] > 0.0
        self.use_layer_dropout = dropout[1] > 0.0
        self.mixture_size = mixture_size
        self.minibatch_size = minibatch_size
        self.layer_parameters = layer_parameters
        self.spatial_dropout = spatial_dropout
        self.ReLu = nn.ReLU()
        self.soft = nn.LogSoftmax(2)
        # Create convolutional and batch normalization layers       
        layers = []
        if self.use_input_dropout:
            layers.append(nn.Dropout(p=dropout[0]))
        
        for c in self.layer_parameters[:-1]:
            in_channels, out_channels, kernel_size, padding, stride, dilation = c
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
            bn = nn.BatchNorm1d(out_channels)
            layers.append(conv)
            layers.append(bn)
            layers.append(self.ReLu)
            if self.use_layer_dropout:
                if self.spatial_dropout:
                    drop = SpatialDropout(p=dropout[1])
                else:
                    drop = nn.Dropout(p=dropout[1])
                layers.append(drop)

        in_channels, out_channels, kernel_size, padding, stride, dilation = self.layer_parameters[-1]

        conv_out = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
        bn = nn.BatchNorm1d(out_channels)


        layers.append(conv_out)
        layers.append(bn)
        layers.append(self.soft)

        self.layers = nn.Sequential(*layers)

        
        #self.softmax_to_angle = models.soft_to_angle(self.mixture_size)
        


    def _get_network_emissions(self, amino_acids):

        data, batch_sizes =  amino_acids
        data = data.transpose(0,1).transpose(1,2)

        x = self.layers(data)
        x = x.transpose(1,2)
        p = torch.exp(x)

        # output_angles = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)
        # backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes, self.use_gpu)
        return p, batch_sizes


class CNNBaseModelAngles(nn.Module):

    def __init__(self, embedding_size, layer_parameters, minibatch_size, use_gpu, mixture_size=3):
        super(CNNBaseModelAngles, self).__init__(use_gpu, embedding_size)

        self.mixture_size = mixture_size
        self.minibatch_size = minibatch_size
        self.layer_parameters = layer_parameters
        self.ReLu = nn.ReLU()
        self.soft = nn.LogSoftmax(2)
        self.lambda_v = 0.1
        # Create convolutional and batch normalization layers       
        layers = []
        for c in self.layer_parameters[:-1]:
            in_channels, out_channels, kernel_size, padding, stride, dilation = c
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
            bn = nn.BatchNorm1d(out_channels)
            layers.append(conv)
            layers.append(bn)
            layers.append(self.ReLu)

        in_channels, out_channels, kernel_size, padding, stride, dilation = self.layer_parameters[-1]

        conv_out = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation)
        bn = nn.BatchNorm1d(out_channels)


        layers.append(conv_out)
        layers.append(bn)
        layers.append(self.soft)

        self.layers = nn.Sequential(*layers)

    def _direct_to_angles(self, x):

        angles = torch.fmod(x, 2 * math.pi)
        # eps = 10 ** (-4)

        # ones = torch.ones([6], dtype=torch.float)
        # if self.use_gpu:
        #     ones = ones.cuda()
        
        # x = ones - x
        # x = self.lambda_v * torch.norm(ones - x, p=2, dim=0, keepdim=True)

        # phi_sin = x[:,:,0].unsqueeze(2)
        # phi_cos = x[:,:,1].unsqueeze(2)
        # psi_sin = x[:,:,2].unsqueeze(2)
        # psi_cos = x[:,:,3].unsqueeze(2) 
        # omega_sin = x[:,:,4].unsqueeze(2)
        # omega_cos = x[:,:,5].unsqueeze(2)
        # phi = torch.atan2(phi_sin, phi_cos + eps)
        # psi = torch.atan2(psi_sin, psi_cos + eps)
        # omega = torch.atan2(omega_sin, omega_cos + eps)

        # angles = torch.cat((phi, psi, omega), 2)
        return angles

    def _get_network_emissions(self, amino_acids):
        data, batch_sizes =  amino_acids
        data = data.transpose(0,1).transpose(1,2)

        x = self.layers(data)
        x = x.transpose(1,2)

        output_angles = self._direct_to_angles(x).transpose(0,1)

        return output_angles, batch_sizes
        # backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, batch_sizes, self.use_gpu)

class SpatialDropout(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__()
        self.p = p

    def forward(self, x):
        x = x.permute(0, 2, 1)   # convert to [batch, channels, time]
        x = torch.nn.functional.dropout2d(x,self.p)
        x = x.permute(0, 2, 1)   # back to [batch, time, channels]
        return x

class soft_to_angle_2(nn.Module):
    def __init__(self, mixture_size):
        super(soft_to_angle_2, self).__init__()
        # Omega intializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))  # Initialize omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1, 1).float())

    def forward(self, p_phi, p_psi, p_omega):
        phi_input_sin = torch.matmul(p_phi, torch.sin(self.phi_components))

        phi_input_cos = torch.matmul(p_phi, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(p_psi, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(p_psi, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(p_omega, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(p_omega, torch.cos(self.omega_components))

        eps = 10 ** (-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos + eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos + eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos + eps)

        return torch.cat((phi, psi, omega), 2)






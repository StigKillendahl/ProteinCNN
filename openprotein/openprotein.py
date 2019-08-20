# This file is part of the ProteinCNN project.
#
# @author Stig Killendahl & Kevin Jon Jensen
#
# Based on the OpenProtein framework, please see the LICENSE file in the root directory.

from util import *
import time
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

#torch.manual_seed(7)

class BaseModel(nn.Module):
    def __init__(self, use_gpu, embedding_size, model, soft_to_angle):
        super(BaseModel, self).__init__()

        self.model = model
        self.soft_to_angle = soft_to_angle

        self.apply_soft_to_angle = soft_to_angle is not None
        # initialize model variables
        self.use_gpu = use_gpu
        self.embedding_size = embedding_size

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string):
        data, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(original_aa_string))

        # one-hot encoding
        start_compute_embed = time.time()
        prot_aa_list = data.unsqueeze(1)
        embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2)) # 21 classes
        if self.use_gpu:
            prot_aa_list = prot_aa_list.cuda()
            embed_tensor = embed_tensor.cuda()
        input_sequences = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1,2)
        end = time.time()
        write_out("Embed time:", end - start_compute_embed)
        packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)
        return packed_input_sequences

    def _mask_minibatch(self, pred, actual, emissions, mask, batch_sizes):
        emissions = emissions.transpose(0,1)

        if self.use_gpu:
            actual = [a.cuda() for a in actual]
            mask = [m.cuda() for m in mask]
        
        pred_masked = None
        if pred is not None:
            pred = pred.transpose(0,1)
            pred_masked = [apply_mask(pred[i][:batch_sizes[i]], mask[i]) for i in range(batch_sizes.size(0))]
            pred_masked = sorted(pred_masked, key=len, reverse=True)
            pred_masked, _  = torch.nn.utils.rnn.pad_packed_sequence(
                torch.nn.utils.rnn.pack_sequence(pred_masked))

        
        actual_masked = [apply_mask(actual[i], mask[i]) for i in range(batch_sizes.size(0))]
        emissions_masked = [apply_mask(emissions[i][:batch_sizes[i]], mask[i], size=3) for i in range(batch_sizes.size(0))]

        actual_masked = sorted(actual_masked, key=len, reverse=True)
        emissions_masked = sorted(emissions_masked, key=len, reverse=True)


        actual_masked, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(actual_masked))
        emissions_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(emissions_masked))

        return pred_masked, actual_masked, emissions_masked, batch_sizes


    def compute_loss(self, amino_acids, actual_coords_list, masks, C=1, loss_atoms="all"):

        model_out = self.model._get_network_emissions(amino_acids)
        emissions, batch_sizes = model_out
        if self.apply_soft_to_angle:
            emissions = self.soft_to_angle(emissions).transpose(0,1) # max size, minibatch size, 3 (angels)

        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(emissions, batch_sizes, self.use_gpu)

        backbone_atoms_padded, actual_coords_list_padded, emissions, batch_sizes = self._mask_minibatch(backbone_atoms_padded,
            actual_coords_list, emissions, masks, batch_sizes)


        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()

        drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_list_padded, batch_sizes, self.use_gpu, loss_atoms=loss_atoms)

        if self.use_gpu:
            drmsd_avg = drmsd_avg.cuda()
        
        if C != 1:
            emissions_actual, batch_sizes_actual = \
                calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, batch_sizes, self.use_gpu)

            if self.use_gpu:
                emissions = emissions.cuda()
                emissions_actual = emissions_actual.cuda()

            angular_loss = calc_angular_difference(emissions, emissions_actual)
            return C * drmsd_avg + (1-C) * angular_loss

        return drmsd_avg

    def compute_angular_loss(self, amino_acids, actual_coords_list, masks):

        model_out = self.model._get_network_emissions(amino_acids)
        emissions, batch_sizes = model_out
        if self.apply_soft_to_angle:
            emissions = self.soft_to_angle(emissions).transpose(0,1) # max size, minibatch size, 3 (angels)

        _, actual_coords_list_padded, emissions, batch_sizes = self._mask_minibatch(None,
            actual_coords_list, emissions, masks, batch_sizes)

        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()

        emissions_actual, _ = \
            calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, batch_sizes, self.use_gpu)
        write_out("Angle calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()

        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return angular_loss



    def forward(self, amino_acids):
        model_out = self.model._get_network_emissions(amino_acids)
        emissions, batch_sizes = model_out
        if self.apply_soft_to_angle:
            emissions = self.soft_to_angle(emissions).transpose(0,1) # max size, minibatch size, 3 (angels)

        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(emissions, batch_sizes, self.use_gpu)
        return emissions, backbone_atoms_padded, batch_sizes
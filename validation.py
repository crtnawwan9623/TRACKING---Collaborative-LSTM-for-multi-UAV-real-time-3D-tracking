import os
import pickle
import argparse
import time
import subprocess


import torch
from torch.autograd import Variable

import numpy as np
from utils import DataLoader
from helper import get_mean_error, get_final_error

from helper import *
from grid import getSequenceGridMask


def main():
    
    parser = argparse.ArgumentParser()
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=27, #original value 15
                        help='Epoch of model to be loaded')
    
    parser.add_argument('--seq_length', type=int, default=40, #orginal value 20
                        help='RNN sequence length')

    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')

    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=5,
                        help='Total number of validation dataset will be visualized')
    # gru support
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    #for drive run
    prefix = ''
    f_prefix = '.'
    if sample_args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'
    

    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # Save directory
    save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
    #plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

    plot_validation_file_directory = 'validation'
    result_directory_my = os.path.join(f_prefix, 'myresult/', method_name)


    # Define the path for the config file for saved args
    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    origin = (0,0)
    reference_point = (0,1)
    net = get_model(sample_args.method, saved_args, True)
    if sample_args.use_cuda:        
        net = net.cuda()

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)

    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 1, sample_args.seq_length, num_of_validation = sample_args.num_validation, forcePreProcess = True, infer = True)
    create_directories(plot_directory, [plot_validation_file_directory])
    create_directories(os.path.join(result_directory_my, model_name), dataloader.get_all_directory_namelist())
    dataloader.reset_batch_pointer()

    print('****************Validation dataset batch processing******************')
    dataloader.reset_batch_pointer(valid=False)
    dataset_pointer_ins = dataloader.dataset_pointer

    loss_epoch = 0
    err_epoch = 0
    f_err_epoch = 0
    num_of_batch = 0
    smallest_err = 100000

    #results of one epoch for all validation datasets
    epoch_result = []
    #results of one validation dataset
    results = []
    submission_my = []
    iteration_submission_my = []
    submission_store_my = []
    # For each batch
    for batch in range(dataloader.num_batches):
        start = time.time()
        # Get batch data
        x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch()

        if dataset_pointer_ins is not dataloader.dataset_pointer:
            if dataloader.dataset_pointer is not 0:
                print('Finished prosessed file : ', dataloader.get_file_name(-1),' Avarage error : ', err_epoch/num_of_batch)
                num_of_batch = 0
                epoch_result.append(results)

            dataset_pointer_ins = dataloader.dataset_pointer
            results = []



        # Loss for this batch
        loss_batch = 0
        err_batch = 0
        f_err_batch = 0

        # For each sequence
        for sequence in range(dataloader.batch_size):
            # Get data corresponding to the current sequence
            x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
            target_id = target_ids[sequence]

            folder_name = dataloader.get_directory_name_with_pointer(d_seq)
            dataset_data = dataloader.get_dataset_dimension(folder_name)


            
            #dense vector creation
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            
            #will be used for error calculation
            orig_x_seq = x_seq.clone() 
            
            target_id_values = x_seq[0][lookup_seq[target_id], 0:2]

            #grid mask calculation
            if sample_args.method == 2: #obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
            elif  sample_args.method == 1: #social lstm   
                grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

            #vectorize datapoints
            x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

            # <---------------- Experimental block (may need update in methods)----------------------->
            # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)
            # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
            # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
            # # Compute grid masks
            # grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, sample_args.neighborhood_size, sample_args.grid_size, sample_args.use_cuda)
            # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


            if sample_args.use_cuda:                    
                x_seq = x_seq.cuda()

            if sample_args.method == 3: #vanilla lstm
                ret_x_seq, loss = sample_validation_data_vanilla(x_seq, PedsList_seq, sample_args, net, lookup_seq, numPedsList_seq, dataloader)

            else:
                ret_x_seq, loss = sample_validation_data(x_seq, PedsList_seq, grid_seq, sample_args, net, lookup_seq, numPedsList_seq, dataloader)
            
            #<---------------------Experimental inverse block -------------->
            # ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, target_id_values, first_values_dict)
            # ret_x_seq = rotate_traj_with_target_ped(ret_x_seq, -angle, PedsList_seq, lookup_seq)
            # ret_x_seq = translate(ret_x_seq, PedsList_seq, lookup_seq ,-target_id_values)
            #revert the points back to original space
            ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)
            
            err = get_mean_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, sample_args.use_cuda, lookup_seq)
            f_err = get_final_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, lookup_seq)
           

            loss_batch += loss
            err_batch += err
            f_err_batch += f_err
            results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq, dataloader.get_frame_sequence(sample_args.seq_length), target_id))
            submission_my.append(submission_preprocess_my(dataloader, ret_x_seq.data[1:, lookup_seq[target_id], :].numpy(), sample_args.seq_length-1, 1, target_id))
        
        end = time.time()
        print('Current file : ', dataloader.get_file_name(0),' Batch : ', batch+1, ' Sequence: ', sequence+1, ' Sequence mean error: ', err,' Sequence final error: ',f_err,' time: ', end - start)
        loss_batch = loss_batch / dataloader.batch_size
        err_batch = err_batch / dataloader.batch_size
        f_err_batch = f_err_batch / dataloader.batch_size
        num_of_batch += 1
        loss_epoch += loss_batch.item()
        err_epoch += err_batch
        f_err_epoch += f_err_batch
    
    iteration_submission_my.append(submission_my)
    submission_store_my.append(iteration_submission_my)
    dataloader.write_to_file(submission_store_my[0], result_directory_my, prefix, model_name)
    epoch_result.append(results)
    

    if dataloader.num_batches != 0:            
        loss_epoch = loss_epoch / dataloader.num_batches
        err_epoch = err_epoch / dataloader.num_batches
        f_err_epoch = f_err_epoch / dataloader.num_batches
        print('valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}'.format(loss_epoch, err_epoch, f_err_epoch))

    dataloader.write_to_plot_file(epoch_result, os.path.join(plot_directory, plot_validation_file_directory))

def submission_preprocess_my(dataloader, ret_x_seq, pred_length, obs_length, target_id):
    seq_lenght = pred_length + obs_length

    #begin and end index of obs. frames in this seq.
    begin_obs = (dataloader.frame_pointer - seq_lenght)
    end_obs = (dataloader.frame_pointer - pred_length)

    # get original data for frame number and ped ids
    observed_data = dataloader.orig_data[dataloader.dataset_pointer][begin_obs:end_obs, :]
    frame_number_predicted = dataloader.get_frame_sequence(pred_length)
    ret_x_seq_c = ret_x_seq.copy()
    ret_x_seq_c[:,[0,1]] = ret_x_seq_c[:,[1,0]] # x, y -> y, x
    repeated_id = np.repeat(target_id, pred_length) # add id
    id_integrated_prediction = np.append(repeated_id[:, None], ret_x_seq_c, axis=1)
    frame_integrated_prediction = np.append(frame_number_predicted[:, None], id_integrated_prediction, axis=1) #add frame number
    result = np.append(observed_data, frame_integrated_prediction, axis = 0)

    return result

if __name__ == '__main__':
    main()



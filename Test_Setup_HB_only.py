# Test setup for the algorithm
# HB over different depths, batch sizes, learning rates
# Decay, dropout, number of convolutional layers used from vgg are constant
# HB epochs = 81

import random, numpy as np, time, os
import sigmoidNN as sigNN, hyperband_nn as hybNN, designchoice as dc, TF_deepnn as deepnn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ranges
depths = [3]
batch_sizes = [50, 100, 200, 500, 1000]
learning_rates = [0, -1, -2, -3, -4]
last_phis = [-5, -4, -3]
phi_ratios = [-6,-5,-4,-3]
num_phis = 100

# Constants - DO NOT CHANGE! The log files depend on this
decay_steps = 1000
decay_rate = 0.99
epochs_hb = 1
epochs_dc = 1
num_conv_layers = 3
dropout = 0.5

data_type = sys.argv[2]
actv_fn = 'relu'

base_dir = 'Output_MainTestSetup/' + data_type

if not os.path.exists(base_dir):
    os.mkdir(base_dir)

common_file_dc = 'Output_MainTestSetup/'+data_type+'_DC_only_all_op.txt' # Contains all the output
common_file_dc_hb = 'Output_MainTestSetup/'+data_type+'_HB_DC_all_op.txt'
# DC + HB
dchb = True

def get_all_phis(ratio, phi, num):
    phi_inner = []

    last_phi_high = phi
    last_phi_low = phi / 10.0

    last_phi = random.uniform(last_phi_low, last_phi_high)
    phi_inner.append(last_phi)

    for i in range(num):
        low_phi_next = last_phi_low * 10 ** ratio
        high_phi_next = last_phi_high * 10 ** ratio

        next_phi = random.uniform(low_phi_next, high_phi_next)

        phi_inner.append(next_phi)
        last_phi_low = low_phi_next
        last_phi_high = high_phi_next

    phi_inner.reverse()
    phi_inner.append(random.uniform(1e-2, 1e-1))

    return phi_inner

def get_cdf(accuracies):
    cdf = 0.0
    cdf_all = []

    num = len(accuracies)

    for i in range(num):
        cdf += accuracies[i]
        cdf_all.append(cdf)

    for i in range(num):
        cdf_all[i] = 1.0*cdf_all[i]/cdf

    return cdf_all

if dchb:
    common_file = open(common_file_dc, 'a')
    common_file_dchb = open(common_file_dc_hb, 'a')

    for i in range(len(depths)):
        depth = depths[i]
        print ('Starting for depth %r' %depth)

        for j in range(len(batch_sizes)):
            batch_size = batch_sizes[j]
            print ('Starting for batch size %r' % batch_size)
            nn = deepnn.deepNN(num_hidden_layers=depth - 1,
                               data_type=data_type,
                               activation_fn=actv_fn,
                               batch_size=batch_size,
                               do_batch_norm=True,
                               vgg_pretrained=True,
                               vgg_num_conv_layers=num_conv_layers,
                               decay_steps=decay_steps,
                               decay_rate=decay_rate
                               )

            for k in range(len(learning_rates)):
                learning_rate = learning_rates[k]

                for l in range(len(last_phis)):
                    phis = []
                    last_phi = last_phis[l]
                    print ('Starting for last_phis %r' % last_phi)
                    # Case 1: take ratios one at a time
                    for m in range(len(phi_ratios)):
                        print ('Starting for phi ratios taken one at a time %r' % depth)

                        ratio = phi_ratios[m]

                        # Get different phis
                        for num in range(num_phis):
                            phis.append(get_all_phis(ratio, 10**last_phi, depth-2))

                        # Get different hidden layer sizes corresponding to phis
                        design_choice = dc.DesignChoice(data_type=data_type,
                                                        vgg_transformed=True,
                                                        new_input_length=num_conv_layers,
                                                        depth=depth,
                                                        phis=phis,
                                                        max_epochs=epochs_dc)
                        hidden_layer_sizes = design_choice.get_all_hidden_layers()

                        accuracy = []

                        # Run the nn for all the hidden layer lengths
                        for hls_iter in range(len(hidden_layer_sizes)):
                            print("Iteration - %d with hidden layer size %r" % (hls_iter, hidden_layer_sizes[hls_iter]))

                            h = sigNN.Hyperparams(hidden_layer_sizes[hls_iter],
                                                  float(10**learning_rate))
                            accuracy.append(1.0 - nn.run_eval_hyperparam_withbs(h,
                                                                                design_choice.max_epochs,
                                                                                keepProb=dropout))

                        # Write to intermediate output
                        outputFileName = '/DC_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                         '_' + str(last_phi) + '_' + str(ratio) + '_constant.txt'

                        f = open(base_dir + outputFileName, 'a')

                        # First line is all the info contained in the file name
                        first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(last_phi) + '\t' + str(ratio) + '\n'
                        #f.write(first_line)
                        #common_file.write(first_line)

                        rest_info = ""
                        rest_info_common = ""
                        for hls_iter in range(len(hidden_layer_sizes)):
                            for item in hidden_layer_sizes[hls_iter]:
                                rest_info += str(item[0]) if depth >= 5 else str(item) + " "# Because output is formatted differently for both cases
                            rest_info += '\t'
                            for item in phis[hls_iter]:
                                rest_info += str(item) + " "
                            rest_info += '\t' + str(accuracy[hls_iter]) + '\n'
                            rest_info_common += rest_info + first_line + '\t'

                        f.write(rest_info)
                        common_file.write(rest_info_common)

                        f.close()

                        # Run Hyperband on above values
                        print 'Now starting Hyperband...'

                        cdf = get_cdf(accuracy)
                        hls_cdf = {}
                        for cdf_range in range(len(cdf)):
                            hls_cdf[cdf[cdf_range]] = hidden_layer_sizes[cdf_range]

                        nn_hb = deepnn.deepNN(num_hidden_layers=depth - 1,
                                              data_type=data_type,
                                              activation_fn=actv_fn,
                                              batch_size=batch_size,
                                              vgg_pretrained=True,
                                              vgg_num_conv_layers=num_conv_layers,
                                              decay_steps=decay_steps,
                                              decay_rate=decay_rate,
                                              random_hls_gen=True,
                                              custom_hls_distr=True,
                                              hls_distr= hls_cdf,
                                              random_lr_gen=True,
                                              lr_highr=1e-0,
                                              lr_lowr=1e-4,
                                              save_models=True)

                        test_hyperband = hybNN.HyperbandNN(nn_hb.get_from_hyperparam_space,
                                                           nn_hb.run_eval_hyperparam_withbs,
                                                           max_epochs=epochs_hb)
                        test_hyperband.run(True)

                        # Write to final DC + HB output

                        outputFileName = '/DC_HB_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                         '_' + str(last_phi) + '_' + str(ratio) + '_constant.txt'

                        f = open(base_dir + outputFileName, 'a')

                        # First line is all the info contained in the file name
                        first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                            last_phi) + '\t' + str(round(ratio, 3)) + '\n'
                        # f.write(first_line)
                        #common_file.write(first_line)

                        rest_info = ""
                        rest_info_common = ""

                        for hb_outcome in test_hyperband.outcomes:
                            hidden_ls = []
                            hls_size = 0
                            for hls_outcome in hb_outcome['hyperparams_hidden_layer_sizes']:
                                hidden_ls.append(hls_outcome)
                                hls_size += 1

                            index = hidden_layer_sizes.index(hidden_ls)
                            rest_info = str(hidden_ls) + '\t' + str(hb_outcome['hyperparams_learning_rate']) + '\t' + str(phis[index]) + '\t' \
                                        + str(accuracy[index]) + '\t' + str(1.0 - hb_outcome['error']) + '\t' + str(hb_outcome['s'])
                            rest_info_common += rest_info + '\t' + first_line + '\n'
                            rest_info += '\n'

                        f.write(rest_info)
                        common_file_dchb.write(rest_info_common)

                        f.close()

                    # Case 2: Randomly sample a ratio from the range of ratios
                    print ('Starting for random phi ratios...')
                    del phis[:]
                    # Get different phis
                    min_phi_ratio = min(phi_ratios)
                    max_phi_ratio = max(phi_ratios)
                    for num in range(num_phis):
                        ratio = random.uniform(min_phi_ratio, max_phi_ratio)
                        phis.append(get_all_phis(ratio, 10**last_phi, depth - 2))

                    # Get different hidden layer sizes corresponding to phis
                    design_choice = dc.DesignChoice(data_type=data_type,
                                                    vgg_transformed=True,
                                                    new_input_length=num_conv_layers,
                                                    depth=depth,
                                                    phis=phis,
                                                    max_epochs=epochs_dc)
                    hidden_layer_sizes = design_choice.get_all_hidden_layers()

                    accuracy = []

                    # Run the nn for all the hidden layer lengths
                    for hls_iter in range(len(hidden_layer_sizes)):
                        print("Iteration - %d with hidden layer size %r" % (hls_iter, hidden_layer_sizes[hls_iter]))

                        h = sigNN.Hyperparams(hidden_layer_sizes[hls_iter],
                                              float(10**learning_rate))
                        accuracy.append(1.0 - nn.run_eval_hyperparam_withbs(h,
                                                                            design_choice.max_epochs,
                                                                            keepProb=dropout))

                    # Write to intermediate output
                    outputFileName = '/DC_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                     '_' + str(last_phi) + '_' + str(round(ratio,3)) + '_range.txt'

                    f = open(base_dir + outputFileName, 'a')

                    # First line is all the info contained in the file name
                    first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                        last_phi) + '\t' + str(round(ratio, 3)) + '\n'
                    # f.write(first_line)
                    #common_file.write(first_line)

                    rest_info = ""
                    rest_info_common = ""
                    for hls_iter in range(len(hidden_layer_sizes)):
                        for item in hidden_layer_sizes[hls_iter]:
                            rest_info += str(item[0]) if depth >= 5 else str(
                                item) + " "  # Because output is formatted differently for both cases
                        rest_info += '\t'
                        for item in phis[hls_iter]:
                            rest_info += str(item) + " "
                        rest_info += '\t' + str(accuracy[hls_iter])
                        rest_info_common += rest_info + '\t' + first_line + '\n'
                        rest_info += '\n'

                    f.write(rest_info)
                    common_file.write(rest_info_common + '\n')

                    f.close()

                    # Run hyperband on above values
                    cdf = get_cdf(accuracy)
                    hls_cdf = {}
                    for cdf_range in range(len(cdf)):
                        hls_cdf[cdf[cdf_range]] = hidden_layer_sizes[cdf_range]

                    nn_hb = deepnn.deepNN(num_hidden_layers=depth - 1,
                                          data_type=data_type,
                                          activation_fn=actv_fn,
                                          batch_size=batch_size,
                                          vgg_pretrained=True,
                                          vgg_num_conv_layers=num_conv_layers,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          random_hls_gen=True,
                                          custom_hls_distr=True,
                                          hls_distr=hls_cdf,
                                          random_lr_gen=True,
                                          lr_highr=1e-0,
                                          lr_lowr=1e-4,
                                          save_models=True)

                    test_hyperband = hybNN.HyperbandNN(nn_hb.get_from_hyperparam_space,
                                                       nn_hb.run_eval_hyperparam_withbs,
                                                       max_epochs=epochs_hb)
                    test_hyperband.run(True)

                    # Write to final DC + HB output

                    outputFileName = '/DC_HB_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                     '_' + str(last_phi) + '_' + str(ratio) + '_range.txt'

                    f = open(base_dir + outputFileName, 'a')

                    # First line is all the info contained in the file name
                    first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                        last_phi) + '\t' + str(round(ratio, 3)) + '\n'
                    # f.write(first_line)
                    # common_file.write(first_line)

                    rest_info = ""
                    rest_info_common = ""

                    for hb_outcome in test_hyperband.outcomes:

                        hidden_ls = []
                        hls_size = 0
                        for hls_outcome in hb_outcome['hyperparams_hidden_layer_sizes']:
                            hidden_ls.append(hls_outcome)
                            hls_size += 1

                        index = hidden_layer_sizes.index(hidden_ls)
                        rest_info = str(hidden_ls) + '\t' + str(hb_outcome['hyperparams_learning_rate']) + '\t' + str(
                            phis[index]) + '\t' + str(accuracy[index]) \
                                    + '\t' + str(1.0 - hb_outcome['error']) + '\t' + str(hb_outcome['s'])
                        rest_info_common += rest_info + '\t' + first_line + '\n'
                        rest_info += '\n'

                    f.write(rest_info)
                    common_file_dchb.write(rest_info_common)

                    f.close()




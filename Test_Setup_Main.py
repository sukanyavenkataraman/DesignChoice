# Test setup for the algorithm
# DC + HB over different depths, batch sizes, learning rates, phi ratios (both constant, and random over a range), last phis
# HB over different depths, batch sizes, learning rates
# Decay, dropout, number of convolutional layers used from vgg are constant
# HB epochs = 81

import random, numpy as np, time, os, sys
import sigmoidNN as sigNN, hyperband_nn as hybNN, designchoice as dc, TF_deepnn as deepnn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Ranges
depths = [3]
batch_sizes = [int(sys.argv[1])]#[50, 100, 200, 500, 1000]
learning_rates = [int(sys.argv[2])]#[0, -1, -2, -3]
last_phis = [-5, -4, -3]
phi_ratios = [-6,-5,-4,-3]
num_phis = int(sys.argv[7])

# Constants - DO NOT CHANGE! The log files depend on this
decay_steps = 1000
decay_rate = 0.99
epochs_hb = int(sys.argv[8])
epochs_hb_only = 4 # since 64log64 is 192 which is approx no. of phis + 16log16. Please note that these have to approx. match to be able to compare HB and HB+DC
epochs_dc = 1
vgg_pretrained = sys.argv[4] == '1'
if len(sys.argv) > 9:
    num_conv_layers = int(sys.argv[9])
else:
    num_conv_layers = 3

dropout = 0.5
eta = 4

data_type = sys.argv[3]
actv_fn = 'relu'

model_dir = sys.argv[5]

if model_dir is None:
    print 'model dir is none. Using deault'
    model_dir = '/home/sukanya/DesignChoice_Hyperband/Models_Medical/'

output_dir = sys.argv[6]

if output_dir is None:
    output_dir = 'Output_MainTestSetup_Medical'

base_dir = output_dir + '/' + data_type

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

common_file_dc = base_dir + '_DC_only_all_op.txt' # Contains all the output
common_file_dc_hb = base_dir + '_HB_DC_all_op.txt'
config_filename = base_dir + '.config' # If configs change, change the output directory

config_file = open(config_filename, 'a')
config_file.write('Depths - %r \nBatch sizes - %r \nlearning rates (10**) - %r \n'
                  'last phis (10**) - %r \nphi ratios (10**) - %r \nnum phis - %r'
                  ' \ndecay steps - %r \ndecay rate - %r \nmax epochs hyperband - %r'
                  '\nmax epochs design choice - %r \nmax epochs hyperband with design choice - %r \n'
                  'VGG pretrained ? %r\nNum conv layers if vgg %r\nDropout %r\nEta %r\nActivation function %r\n'
                  %(depths, batch_sizes, learning_rates, last_phis, phi_ratios, num_phis, decay_steps, decay_rate,
                    epochs_hb_only, epochs_dc, epochs_hb, vgg_pretrained, num_conv_layers, dropout, eta, actv_fn ))

config_file.close()

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

#TODO: Separate this out into main function and other functions

common_file = open(common_file_dc, 'a')
common_file_dchb = open(common_file_dc_hb, 'a')

input_data = deepnn.DataSet(data_type=data_type,
                            use_resnet_pretrained=True,
                            use_vgg_pretrained=False,
                            num_conv_layers=num_conv_layers)

time_before = time.time()
for i in range(len(depths)):
    depth = depths[i]
    print ('Starting for depth %r' %depth)

    for j in range(len(batch_sizes)):
        batch_size = batch_sizes[j]
        print ('Starting for batch size %r' % batch_size)
        nn = deepnn.deepNN(num_hidden_layers=depth - 1,
                           input_data=input_data,
                           activation_fn=actv_fn,
                           batch_size=batch_size,
                           do_batch_norm=True,
                           decay_steps=decay_steps,
                           decay_rate=decay_rate
                           )

        for k in range(len(learning_rates)):
            learning_rate = learning_rates[k]
            # First run Hyperband alone
            # TODO: Make these separate function calls?
            # TODO: Make data loading a separate function and pass it to deepNN

            for l in range(len(last_phis)):
                last_phi = last_phis[l]
                print ('Starting for last_phis %r' % last_phi)
                # Case 1: take ratios one at a time
                for m in range(len(phi_ratios)):
                    phis = []
                    t_b_1 = time.time()

                    ratio = phi_ratios[m]

                    print ('Starting for phi ratios taken one at a time %r' % ratio)
                    # Get different phis
                    for num in range(num_phis):
                        phis.append(get_all_phis(ratio, 10**last_phi, depth-2))

                    # Get different hidden layer sizes corresponding to phis
                    design_choice = dc.DesignChoice(pixel_size=input_data.size,
                                                    num_classes=input_data.num_classes,
                                                    depth=depth,
                                                    phis=phis)
                    hidden_layer_sizes = design_choice.get_all_hidden_layers()

                    hls = {}
                    hls_copy = list(hidden_layer_sizes)
                    phis_copy = list(phis)

                    for hls_each in range(len(hls_copy)):
                        hls_string = ' '.join(str(dim) for dim in hls_copy[hls_each])
                        if hls_string not in hls:
                            hls[hls_string] = 1
                        else:
                            hidden_layer_sizes.remove(hls_copy[hls_each])
                            phis.remove(phis_copy[hls_each])

                    accuracy = []

                    # Run the nn for all the hidden layer lengths
                    for hls_iter in range(len(hidden_layer_sizes)):
                        print("Iteration - %d with hidden layer size %r" % (hls_iter, hidden_layer_sizes[hls_iter]))

                        h = sigNN.Hyperparams(hidden_layer_sizes[hls_iter],
                                              float(10**learning_rate))
                        acc = 1.0 - nn.run_eval_hyperparam_withbs(h,
                                                                  epochs_dc,
                                                                  keepProb=dropout)[0]
                        accuracy.append(acc)

                    t_a_1 = time.time() - t_b_1
                    print ('Time taken for this run of DC is %r' %t_a_1)
                    # Write to intermediate output
                    outputFileName = '/DC_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                     '_' + str(last_phi) + '_' + str(ratio) + '_constant.txt'

                    f = open(base_dir + outputFileName, 'a')

                    # First line is all the info contained in the file name
                    first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(last_phi) + \
                                 '\t' + str(ratio) + '\t' + str(t_a_1)

                    rest_info = ""
                    for hls_iter in range(len(hidden_layer_sizes)):
                        for item in hidden_layer_sizes[hls_iter]:
                            rest_info += str(item[0]) if depth >= 5 else str(item) + " "# Because output is formatted differently for both cases
                        rest_info += '\t'
                        for item in phis[hls_iter]:
                            rest_info += str(item) + " "
                        rest_info += '\t' + str(accuracy[hls_iter])
                        rest_info += '\t'+first_line+'\n'

                    f.write(rest_info)
                    common_file.write(rest_info)

                    f.close()

                    # Run Hyperband on above values
                    print 'Now starting Hyperband after Design Choice...'
                    if len(hidden_layer_sizes) <= 1:
                        break
                    cdf = get_cdf(accuracy)
                    hls_cdf = {}
                    for cdf_range in range(len(cdf)):
                        hls_cdf[cdf[cdf_range]] = hidden_layer_sizes[cdf_range]

                    nn_hb = deepnn.deepNN(num_hidden_layers=depth - 1,
                                          input_data=input_data,
                                          activation_fn=actv_fn,
                                          batch_size=batch_size,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          random_hls_gen=True,
                                          custom_hls_distr=True,
                                          hls_distr= hls_cdf,
                                          random_lr_gen=True,
                                          lr_highr=1e-1,
                                          lr_lowr=1e-4,
                                          save_models=True,
                                          model_dir=model_dir,
                                          model_name='DC_HB'+str(learning_rate)+str(int(ratio))+'_range')

                    test_hyperband = hybNN.HyperbandNN(nn_hb.get_from_hyperparam_space,
                                                       nn_hb.run_eval_hyperparam_withbs,
                                                       max_epochs=epochs_hb,
                                                       eta=eta)
                    test_hyperband.run(True)
                    t_a_2 = time.time() - t_b_1
                    print ('Time taken for this run of DC + HB is %r' %t_a_2)

                    # Write to final DC + HB output

                    outputFileName = '/DC_HB_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                     '_' + str(last_phi) + '_' + str(ratio) + '_constant.txt'

                    f = open(base_dir + outputFileName, 'a')

                    # First line is all the info contained in the file name
                    first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                        last_phi) + '\t' + str(round(ratio, 3)) + '\t' + str(t_a_2)

                    rest_info = ""

                    for hb_outcome in test_hyperband.outcomes:
                        hidden_ls = []
                        hls_size = 0
                        for hls_outcome in hb_outcome['hyperparams_hidden_layer_sizes']:
                            hidden_ls.append(hls_outcome)
                            hls_size += 1

                        index = hidden_layer_sizes.index(hidden_ls)
                        rest_info += str(hidden_ls) + '\t' + str(hb_outcome['hyperparams_learning_rate']) + '\t' + str(phis[index]) + '\t' \
                                    + str(accuracy[index]) + '\t' + str(1.0 - hb_outcome['error']) + '\t' + str(1.0 - hb_outcome['test_error']) + \
                                     '\t' + str(hb_outcome['s'])
                        rest_info += '\t' + first_line + '\n'

                    f.write(rest_info)
                    common_file_dchb.write(rest_info)

                    f.close()

                # Case 2: Randomly sample a ratio from the range of ratios
                print ('Starting for random phi ratios...')
                t_b_2 = time.time()
                del phis[:]
                # Get different phis
                min_phi_ratio = min(phi_ratios)
                max_phi_ratio = max(phi_ratios)
                for num in range(num_phis):
                    ratio = random.uniform(min_phi_ratio, max_phi_ratio)
                    phis.append(get_all_phis(ratio, 10**last_phi, depth - 2))

                # Get different hidden layer sizes corresponding to phis
                design_choice = dc.DesignChoice(pixel_size=input_data.size,
                                                num_classes=input_data.num_classes,
                                                depth=depth,
                                                phis=phis)
                hidden_layer_sizes = design_choice.get_all_hidden_layers()

                hls = {}
                hls_copy = list(hidden_layer_sizes)
                phis_copy = list(phis)

                for hls_each in range(len(hls_copy)):
                    hls_string = ' '.join(str(dim) for dim in hls_copy[hls_each])
                    if hls_string not in hls:
                        hls[hls_string] = 1
                    else:
                        hidden_layer_sizes.remove(hls_copy[hls_each])
                        phis.remove(phis_copy[hls_each])

                accuracy = []

                # Run the nn for all the hidden layer lengths
                for hls_iter in range(len(hidden_layer_sizes)):
                    print("Iteration - %d with hidden layer size %r" % (hls_iter, hidden_layer_sizes[hls_iter]))

                    h = sigNN.Hyperparams(hidden_layer_sizes[hls_iter],
                                          float(10**learning_rate))
                    acc = 1.0 - nn.run_eval_hyperparam_withbs(h,
                                                              epochs_dc,
                                                              keepProb=dropout)[0]
                    accuracy.append(acc)

                t_a_1 = time.time() - t_b_2
                print ('Time taken for this run of DC is %r' %t_a_1)
                # Write to intermediate output
                outputFileName = '/DC_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                 '_' + str(last_phi) + '_' + str(round(ratio,3)) + '_range.txt'

                f = open(base_dir + outputFileName, 'a')

                # First line is all the info contained in the file name
                first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                    last_phi) + '\t' + str(round(ratio, 3)) + '\t' + str(t_a_1)

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
                    rest_info += '\t' + first_line + '\n'

                f.write(rest_info)
                common_file.write(rest_info)

                f.close()

                # Run hyperband on above values
                cdf = get_cdf(accuracy)
                hls_cdf = {}
                for cdf_range in range(len(cdf)):
                    hls_cdf[cdf[cdf_range]] = hidden_layer_sizes[cdf_range]

                nn_hb = deepnn.deepNN(num_hidden_layers=depth - 1,
                                      input_data=input_data,
                                      activation_fn=actv_fn,
                                      batch_size=batch_size,
                                      decay_steps=decay_steps,
                                      decay_rate=decay_rate,
                                      random_hls_gen=True,
                                      custom_hls_distr=True,
                                      hls_distr=hls_cdf,
                                      random_lr_gen=True,
                                      lr_highr=1e-1,
                                      lr_lowr=1e-4,
                                      save_models=True,
                                      model_dir=model_dir,
                                      model_name='DC_HB'+str(learning_rate)+str(int(ratio))+'_constant')

                test_hyperband = hybNN.HyperbandNN(nn_hb.get_from_hyperparam_space,
                                                   nn_hb.run_eval_hyperparam_withbs,
                                                   max_epochs=epochs_hb,
                                                   eta=eta)
                test_hyperband.run(True)

                t_a_2 = time.time() - t_b_2
                print ('Time taken for this run of DC+HB is %r' %t_a_2)
                # Write to final DC + HB output

                outputFileName = '/DC_HB_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + \
                                 '_' + str(last_phi) + '_' + str(ratio) + '_range.txt'

                f = open(base_dir + outputFileName, 'a')

                # First line is all the info contained in the file name
                first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t' + str(
                    last_phi) + '\t' + str(round(ratio, 3)) +'\t'+ str(t_a_2)

                rest_info = ""

                for hb_outcome in test_hyperband.outcomes:

                    hidden_ls = []
                    hls_size = 0
                    for hls_outcome in hb_outcome['hyperparams_hidden_layer_sizes']:
                        hidden_ls.append(hls_outcome)
                        hls_size += 1

                    index = hidden_layer_sizes.index(hidden_ls)
                    rest_info += str(hidden_ls) + '\t' + str(hb_outcome['hyperparams_learning_rate']) + '\t' + str(
                        phis[index]) + '\t' + str(accuracy[index]) \
                                + '\t' + str(1.0 - hb_outcome['error']) + '\t' + str(1.0 - hb_outcome['test_error']) + '\t' + str(hb_outcome['s'])
                    rest_info += '\t' + first_line + '\n'

                f.write(rest_info)
                common_file_dchb.write(rest_info)

                f.close()

print ('Total time taken is - %r' %(time.time() - time_before))
common_file.close()
common_file_dchb.close()





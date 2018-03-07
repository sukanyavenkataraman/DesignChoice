# Test setup for the algorithm
# HB over different depths, batch sizes, learning rates
# Decay, dropout, number of convolutional layers used from vgg are constant

import random, numpy as np, time, os, sys
import sigmoidNN as sigNN, hyperband_nn as hybNN, designchoice as dc, TF_deepnn as deepnn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

print sys.argv, len(sys.argv)
# Ranges
depths = [3]
batch_sizes = [int(sys.argv[1])]#[50, 100, 200, 500, 1000]
learning_rates = [int(sys.argv[2])]#[0, -1, -2, -3]
num_phis = 150

# Constants - DO NOT CHANGE! The log files depend on this
decay_steps = 1000
decay_rate = 0.99
epochs_hb = 16 #We don't actually use this here
epochs_hb_only = int(sys.argv[7]) # since 64log64 is 192 which is approx no. of phis + 16log16. Please note that these have to approx. match to be able to compare HB and HB+DC
epochs_dc = 1
vgg_pretrained = sys.argv[4] == '1'
if len(sys.argv) > 8:
    num_conv_layers = int(sys.argv[8])
else:
    num_conv_layers = 3
dropout = 0.5
eta = 4

actv_fn = 'relu'
data_type = sys.argv[3]
print data_type

model_dir = sys.argv[5]

if model_dir is None:
    model_dir = '/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Models_Medical/'

output_dir = sys.argv[6]

if output_dir is None:
    output_dir = 'Output_MainTestSetup_Medical'

base_dir = output_dir + '/' + data_type

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

common_file_hb = base_dir + '_HB_only_all_op.txt'
config_filename = base_dir + '_HB.config' # If configs change, change the output directory

config_file = open(config_filename, 'a')
config_file.write('Depths - %r \nBatch sizes - %r \nlearning rates (10**) - %r \n'
                  '\nnum phis - %r\ndecay steps - %r \ndecay rate - %r \nmax epochs hyperband - %r'
                  '\nmax epochs design choice - %r \nmax epochs hyperband with design choice - %r \n'
                  'VGG pretrained ? %r\nNum conv layers if vgg %r\nDropout %r\nEta %r\nActivation function %r\n'
                  %(depths, batch_sizes, learning_rates, num_phis, decay_steps, decay_rate,
                    epochs_hb_only, epochs_dc, epochs_hb, vgg_pretrained, num_conv_layers, dropout, eta, actv_fn ))

config_file.close()

#TODO: Separate this out into main function and other functions

common_file_hb = open(common_file_hb, 'a')

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

            print ('Running hyperband alone for learning rate %r' %learning_rate)
            t_b = time.time()
            nn_hb = deepnn.deepNN(num_hidden_layers=depth - 1,
                                  input_data=input_data,
                                  activation_fn=actv_fn,
                                  batch_size=batch_size,
                                  decay_steps=decay_steps,
                                  decay_rate=decay_rate,
                                  random_hls_gen=True,
                                  custom_hls_distr=False,
                                  hls_low=100,
                                  hls_high=5000,
                                  random_lr_gen=False,
                                  learning_rate = float(10**learning_rate),
                                  save_models=True,
                                  model_dir=model_dir,
                                  model_name='HB_'+str(learning_rate))

            test_hyperband = hybNN.HyperbandNN(nn_hb.get_from_hyperparam_space,
                                               nn_hb.run_eval_hyperparam_withbs,
                                               max_epochs=epochs_hb_only,
                                               eta=eta)
            test_hyperband.run(True)
            time_after = time.time() - t_b
            print ('Time taken for this run is %r' %time_after)

            # Write to final HB output

            outputFileName = '/HB_' + str(depth) + '_' + str(batch_size) + '_' + str(learning_rate) + '.txt'

            f = open(base_dir + outputFileName, 'a')

            # First line is all the info contained in the file name
            first_line = str(depth) + '\t' + str(batch_size) + '\t' + str(learning_rate) + '\t'+ str(time_after)

            rest_info = ""
            print test_hyperband.outcomes

            for hb_outcome in test_hyperband.outcomes:
                hidden_ls = []
                hls_size = 0
                for hls_outcome in hb_outcome['hyperparams_hidden_layer_sizes']:
                    hidden_ls.append(hls_outcome)
                    hls_size += 1

                rest_info += str(hidden_ls) + '\t' + str(hb_outcome['hyperparams_learning_rate']) + '\t' + \
                            str(1.0 - hb_outcome['error']) + '\t' + str(1.0 - hb_outcome['test_error']) + '\t' + str(hb_outcome['s'])
                rest_info += '\t' + first_line + '\n'

            print rest_info
            f.write(rest_info)
            common_file_hb.write(rest_info)#_common)

            f.close()
print ('Total time taken is - %r' %(time.time() - time_before))
common_file_hb.close()

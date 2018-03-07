# Run Vamsi's algo, take output from that and then run hyperband on top of it
# Input to hyperband -> no. of hidden layers, first layer length, output layer length, validation set, size of each hidden layer
# Output from hyperband -> learning rate, batch size?, error corresponding to the learning rate -> can we also use knowledge of the W's calculated in the subsequent runs?
# Hyperband hyperparameters -> learning rate, batch size?

import random, numpy as np, time, os
import sigmoidNN as sigNN, hyperband_nn as hybNN, designchoice as dc, TF_deepnn as deepnn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dshp = 0
if dshp == -1 : # Use TF's fully automated API for dnn
    hidden_layer_size_1 = [100, 150]

    sigmoid_nn = sigNN.SigmoidNN(hidden_layer_size_1,
                                 num_hidden_layers=2,
                                 data_type='svhn',
                                 activation_fn='relu',
                                 batch_size=50)

    learning_rate = 0.0014294837137666
    h = sigNN.Hyperparams(hidden_layer_size_1,
                          learning_rate = learning_rate)
    print learning_rate
    loss = sigmoid_nn.run_eval_hyperparam_withbs(h,
                                                 num_epochs=10,
                                                 dropout = 0.25)

if dshp == 0: #Sanity check
    hidden_layer_size_1 = [512]

    batch_size = 32
    learning_rate = 1.0
    decay_steps = 10000
    decay_rate = 0.50
    data_type = 'av45'

    print 'Running for configurations - '
    print str(data_type) + '_' + \
               str(hidden_layer_size_1) +'hls_' + str(batch_size) + 'bs_' + str(learning_rate) + 'lr_' + str(decay_steps) + \
               'decay_' + str(decay_rate) + '_' + '10epochs_0.5dropout_withvgg_3layers_check'

    input_data = deepnn.DataSet(data_type=data_type, use_resnet_pretrained = True, use_vgg_pretrained=False, num_conv_layers=5, kl_type='mean', num_classes=4)
    num_phis = 2
    phis = [[0.001, 0.001, 0.5], [0.001, 0.001, 0.5]]
    '''
    # Get different hidden layer sizes corresponding to phis
    design_choice = dc.DesignChoice(pixel_size=input_data.size,
                                    num_classes=input_data.num_classes,
                                    # vgg_transformed=input_data.vgg_pretrained,
                                    # new_input_length=num_conv_layers,
                                    depth=3,
                                    phis=phis)
    hidden_layer_sizes = design_choice.get_all_hidden_layers()
    print hidden_layer_sizes
    '''
    deep_nn = deepnn.deepNN(input_data=input_data,
                            hidden_layer_sizes=hidden_layer_size_1,
                            num_hidden_layers=2,
                            activation_fn='relu',
                            batch_size=batch_size,
                            do_batch_norm=True,
                            scattering_transform=False,
                            decay_steps=decay_steps,
                            decay_rate=decay_rate)

    h = deepnn.Hyperparams(hidden_layer_size_1,
                           learning_rate)

    deep_nn.run_eval_hyperparam_withbs(hyperparams=h,
                                       num_epochs=50,
                                       keepProb=0.5)

if dshp == 1:
    # Do only Hyperband

    start_time = time.clock()

    # Set the params for the neural net

    # Network specific params
    dropout = 0.5
    num_hidden_layers = 2
    data_type = 'cifar10'
    batch_size = 50
    max_epochs = 81
    actv_fn = 'relu'

    # Learning rate specific params
    random_lr_gen = False
    learning_rate = 0.1
    lr_highr = 1e-4
    lr_lowr = 1e-2

    # Hidden layer size specific params
    random_hls_gen = True
    num_hidden_layers = 2
    hls_low = 1
    hls_high = 1000

    decay_steps = 1000
    decay_rate = 0.99
    '''
    sigmoid_nn = sigNN.SigmoidNN(random_hls_gen=random_hls_gen,
                                 num_hidden_layers=num_hidden_layers,
                                 hls_low=hls_low,
                                 hls_high=hls_high,
                                 batch_size=batch_size,
                                 data_type=data_type,
                                 random_lr_gen=random_lr_gen,
                                 learning_rate=learning_rate,
                                 lr_highr=lr_highr,
                                 lr_lowr=lr_lowr,
                                 activation_fn=actv_fn)
    '''
    sigmoid_nn = deepnn.deepNN(random_hls_gen=random_hls_gen,
                                 num_hidden_layers=num_hidden_layers,
                                 hls_low=hls_low,
                                 hls_high=hls_high,
                                 batch_size=batch_size,
                                 data_type=data_type,
                                 random_lr_gen=random_lr_gen,
                                 learning_rate=learning_rate,
                                 lr_highr=lr_highr,
                                 lr_lowr=lr_lowr,
                                 activation_fn=actv_fn,
                                 do_batch_norm=True,
                                 scattering_transform=False,
                                 vgg_pretrained=True,
                                 vgg_num_conv_layers=3,
                                 decay_steps=decay_steps,
                                 decay_rate=decay_rate,
                                 save_models=True)

    test_hyperband = hybNN.HyperbandNN(sigmoid_nn.get_from_hyperparam_space,
                                       sigmoid_nn.run_eval_hyperparam_withbs,
                                       max_epochs=max_epochs)

    debug = True

    test_hyperband.run(debug)
    print(test_hyperband.outcomes)

    base_dir = 'Output/' + data_type

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    outputFileName = '/Hyperband_'+ \
                     str(dropout) + '_' + str(num_hidden_layers) + '_' + \
                     str(max_epochs) + '_' + str(hls_low) + '_' + str(hls_high) + '_' + str(learning_rate) + '_'\
                     + actv_fn + '_' + str(batch_size) + '_' + str(decay_rate) + '_' + str(decay_steps) + '_withvgg_3layers_withdecay.txt'

    f = open(base_dir + outputFileName, 'a')

    for i in test_hyperband.outcomes:
        for j in i['hyperparams_hidden_layer_sizes']:
            f.write(str(j) + " ")
        f.write('\t' + str(i['hyperparams_learning_rate']) + '\t' + str(i['error']) + '\t' + str(i['s']) + '\n')

    end_time = time.clock()
    print(end_time - start_time)

    f.write(str(end_time - start_time))
    f.close()


if dshp == 2:
    # Do only design choice

    vary_phi1 = False
    vary_phi2 = False # Deciding which phi to fix
    vary_both = False # Never set both as true together!
    vary_all = True # For depth > 3

    if vary_both:
        phi_iter = 1
        low_firstphi = 1e-11
        high_firstphi = 1e-9

        low_secondphi = 9e-5
        high_secondphi = 1e-5

    if vary_phi1 or vary_phi2:
        phi_iter = 4

        low_firstphi = 1e-8
        high_firstphi = 1e-5

        low_secondphi = 1e-4
        high_secondphi = 1e-2

    if vary_all:
        phi_iter = 3

        low_last_phi = 1e-5
        high_last_phi = 1e-5

        phi_ratio_low = -3 # Equivalent to 1e-6 and 1e-3
        phi_ratio_high = -3

        phi_ratio_in_order = False

    # Design choice specific params
    max_epochs = 1
    depth = 3
    data_type = 'cifar10'
    learning_rate = 0.1
    decay_steps = 1000
    decay_rate = 0.1
    actv_fn = 'relu'
    vgg_num_conv_layers = 3

    batch_size = 50

    use_tf_api = False

    # First initialise the neural network class

    if use_tf_api:
        sigmoid_nn = sigNN.SigmoidNN(num_hidden_layers=depth - 1,
                                     data_type=data_type,
                                     activation_fn=actv_fn,
                                     batch_size=batch_size)

    else:
        sigmoid_nn = deepnn.deepNN(num_hidden_layers=depth - 1,
                                   data_type=data_type,
                                   activation_fn=actv_fn,
                                   batch_size=batch_size,
                                   scattering_transform=False,
                                   vgg_pretrained=True,
                                   vgg_num_conv_layers=vgg_num_conv_layers,
                                   decay_steps=decay_steps,
                                   decay_rate=decay_rate
                                   )

    for j in range(phi_iter):
        num_varying_phis = 100

        # generate random values of phi_hidden_layer, phi_op_minus1, phi_op
        phi_hidden_layer = []
        phi_op_minus1 = []
        phi_op = []

        constant = 0
        if vary_phi2:
            constant = random.uniform(low_firstphi*10**j, low_firstphi*10**(j+1))

        if vary_phi1:
            constant = random.uniform(low_secondphi*10**j, low_secondphi*10**(j+1))

        phis = []
        for i in range(num_varying_phis):

            phi_inner = [] # This is only for the vary all case

            if vary_phi1:
                phi_hidden_layer.append(random.uniform(low_firstphi, high_firstphi))
                phi_op_minus1.append(constant)

            if vary_phi2:
                phi_hidden_layer.append(constant)
                phi_op_minus1.append(random.uniform(low_secondphi, high_secondphi))

            if vary_both:
                phi_hidden_layer.append(random.uniform(low_firstphi, high_firstphi))
                phi_op_minus1.append(random.uniform(low_secondphi, high_secondphi))

            if vary_all:
                # For this, we start filling in phis from the last. We reverse the list in the end
                low_phi_prev = low_last_phi*10**j
                high_phi_prev = high_last_phi*10**j

                prev_phi = random.uniform(low_phi_prev, high_phi_prev)
                phi_inner.append(prev_phi)

                for layer_no in range(depth-2):

                    ratio = random.uniform(phi_ratio_low, phi_ratio_high) # TODO: Check if we should changes ratios to randint?
                    low_phi_next = low_phi_prev*10**ratio
                    high_phi_next = high_phi_prev*10**ratio

                    next_phi = random.uniform(low_phi_next, high_phi_next)
                    print 'Previous and next phis are - '
                    print prev_phi, next_phi

                    #if next_phi > 0.1:
                    #    next_phi = 0.1

                    phi_inner.append(next_phi)
                    low_phi_prev = low_phi_next
                    high_phi_prev = high_phi_next

                print 'Phi inner before - ', phi_inner
                phi_inner.reverse()
                print 'Phi inner after - ', phi_inner

                phi_inner.append(random.uniform(1e-2, 1e-1))
                phis.append(phi_inner)

            # TODO: Fix below to varying values?!
        phi_op.append(random.uniform(1e-2, 1e-1)) # For now, this doesn't matter since we fix psi as 0.5

        if vary_phi1:
            phi_hidden_layer = sorted(phi_hidden_layer)

        if vary_phi2:
            phi_op_minus1 = sorted(phi_op_minus1)

        if vary_phi1 or vary_phi2 or vary_both:
            first_phi = np.asarray(phi_hidden_layer)[np.newaxis].transpose()
            second_phi = np.asarray(phi_op_minus1)[np.newaxis].transpose()
            last_phi = np.asarray(phi_op)[np.newaxis].transpose()

            phis = np.concatenate((np.concatenate((first_phi, second_phi), axis=1), last_phi), axis=1)
            #print phis

        start_time = time.clock()
        design_choice = dc.DesignChoice(data_type=data_type,
                                        vgg_transformed=True,
                                        new_input_length=vgg_num_conv_layers,
                                        depth=depth,
                                        max_epochs=max_epochs,
                                        phis=phis)

        hidden_layer_sizes = design_choice.get_all_hidden_layers()
        print(hidden_layer_sizes)

        acc = []

        # Run for each value and get the results
        for i in range(len(hidden_layer_sizes)):
            print("Iteration - %d %d" %(j, i))

            h = sigNN.Hyperparams(hidden_layer_sizes[i],
                                  learning_rate)
            acc.append(1.0 - sigmoid_nn.run_eval_hyperparam_withbs(h,
                                                                   design_choice.max_epochs,
                                                                   design_choice.psi))

        # Write output to file since we'll be reusing this when we run hyperband on it!

        base_dir = 'Output/' + data_type

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        outputFileName = '/hls_phis_acc_'
        if vary_both:
            outputFileName += 'bothvaried'
        if vary_all:
            outputFileName += 'allvaried'
        if vary_phi1:
            outputFileName += 'secondphiconstant'
        if vary_phi2:
            outputFileName += 'firstphiconstant'

        if vary_all:
            outputFileName += '_' + str(design_choice.psi) + '_' + str(design_choice.network_depth - 1) + '_' + \
                          str(max_epochs) + '_' + str(round(learning_rate, 5)) + '_' + actv_fn + '_' + str(batch_size) + '_' + str(decay_rate) + '_' + str(decay_steps) +\
                              '_' + 'withvgg_3layers_includedinDC'
        else:
            outputFileName += '_' + str(design_choice.psi) + '_' + str(design_choice.network_depth - 1) + '_' + \
                          str(max_epochs) + '_' + (str(low_firstphi*10**j) if vary_phi2 else str(low_secondphi*10**j)) + \
                          '_' + str(round(learning_rate, 5)) + '_' + actv_fn + '_' + str(batch_size)

        fmt = '.txt'

        f = open(base_dir + outputFileName + fmt, 'a')

        for i in range(len(hidden_layer_sizes)):
            for item in hidden_layer_sizes[i]:
                f.write((str(item[0]) if depth >=5 else str(item)) + " ") # Because output is formatted differently for both cases
            f.write('\t')
            for item in phis[i]:
                f.write(str(item) + " ") # Change below accordingly
            f.write('\t' + str(acc[i]) + '\n')
            #f.write(str(phi_hidden_layer[i]) + '\t' + (str(phi_op_minus1[i]) if (vary_phi2 | vary_both) else str(phi_op_minus1[0])) + '\t' + str(acc[i]) + '\n')

        f.close()

        max_acc = max(acc)
        min_acc = min(acc)

        print('Max accuracy - %f for '% max_acc)
        print(hidden_layer_sizes[acc.index(max_acc)])
        print('with phis - ')
        print(phis[acc.index(max_acc)])

        print('Min accuracy - %f for ' % min_acc)
        print(hidden_layer_sizes[acc.index(min_acc)])
        print('with phis - ')
        print(phis[acc.index(min_acc)])

        fmt = '.png'

        # plot
        if vary_both: # 3D plot!
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(phi_hidden_layer, phi_op_minus1, acc, c='r', marker='o')

            ax.set_xlabel('First phi')
            ax.set_ylabel('Second phi')
            ax.set_zlabel('Accuracy')

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])

            plt.savefig(base_dir + outputFileName + fmt)
            plt.close(fig)
        else:

            if not vary_all:
                fig, ax = plt.subplots(1)
                print(phi_op_minus1)
                ax.plot(phi_op_minus1 if vary_phi2 else phi_hidden_layer, acc)
                ax.set_xlabel('Second phi' if vary_phi2 else 'First phi')
                ax.set_ylabel('Accuracy')

                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

                plt.savefig(base_dir + outputFileName + fmt)
                plt.close(fig)

        print('Time taken for iteration %d %d %f'%(j, i, time.clock() - start_time))

# Run hyperband on a given distribution of phis
if dshp == 3:

    filename = "/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Output/cifar10/hls_phis_acc_allvaried_0.5_2_1_0.00743_relu_50.txt"

    hls_phis = {}
    acc_phis = {}
    hls_cdf = {}

    cdf_phis = []
    phis = []

    hls_phis_inv = {}

    tot_acc = 0.0
    cdf = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()

        i = 0
        for line in lines:
            parts = line.split('\t')
            print(parts)
            # Different values of phi point to the same hidden layer lengths, with different losses. What do we do about this?

            # First part is hidden layer lengths
            #phis.append(parts[1].strip() + '\t' + parts[2].strip()) #old format
            phis.append(parts[1].strip()) #New format
            hls_phis[phis[i]] = parts[0].strip()

            if parts[0].strip() in hls_phis_inv:
                hls_phis_inv[parts[0].strip()] += '\t' + str(phis[i])
            else:
                hls_phis_inv[parts[0].strip()] = str(phis[i])

            #acc_phis[phis[i]] = 1.0 - float(parts[3].strip()) #old format

            acc_phis[phis[i]] = float(parts[2].strip())
            tot_acc += acc_phis[phis[i]]

            cdf += acc_phis[phis[i]]
            cdf_phis.append(cdf)

            i += 1

    # Normalise accuracies to get a probability distribution across phis

    for i in range(len(cdf_phis)):
        newVal = 1.0*cdf_phis[i]/tot_acc
        hls_cdf[newVal] = hls_phis[phis[i]]
        cdf_phis[i] = newVal

    # Sending hls_cdf to hyperband

    start_time = time.clock()

    # Design choice specific params
    max_epochs = 10
    depth = 3
    data_type = 'cifar10'
    actv_fn = 'relu'
    decay_steps = 1000
    decay_rate = 0 #No decay for now. Change to decay

    batch_size = 50

    use_tf_api = False

    if use_tf_api:
        sigmoid_nn = sigNN.SigmoidNN(random_hls_gen=True,
                                     custom_hls_distr=True,
                                     hls_distr= hls_cdf,
                                     num_hidden_layers=depth - 1,
                                     data_type=data_type,
                                     activation_fn=actv_fn,
                                     batch_size=batch_size,
                                     random_lr_gen=True,
                                     lr_highr=1e-2,
                                     lr_lowr=1e-4,
                                     )

    else:
        sigmoid_nn = deepnn.deepNN(random_hls_gen=True,
                                   custom_hls_distr=True,
                                   hls_distr= hls_cdf,
                                   num_hidden_layers=depth - 1,
                                   data_type=data_type,
                                   activation_fn=actv_fn,
                                   batch_size=batch_size,
                                   scattering_transform=False,
                                   random_lr_gen=True,
                                   lr_highr=1e-2,
                                   lr_lowr=1e-4,
                                   vgg_pretrained=True,
                                   vgg_num_conv_layers=5,
                                   decay_steps=1000,
                                   decay_rate = 0,
                                   save_models=True
                                   ) # For now, no decay. change to decay
    '''  
    sigmoid_nn = sigNN.SigmoidNN(random_hls_gen=True,
                                 custom_hls_distr=True,
                                 hls_distr= hls_cdf,
                                 num_hidden_layers=2,
                                 hls_low=1,
                                 hls_high=1000,
                                 batch_size=768,
                                 data_type='cifar10',
                                 random_lr_gen=True,
                                 lr_highr=1e-4,
                                 lr_lowr=1e-2)
'''
    test_hyperband = hybNN.HyperbandNN(sigmoid_nn.get_from_hyperparam_space,
                                       sigmoid_nn.run_eval_hyperparam_withbs,
                                       max_epochs=max_epochs)

    test_hyperband.run(True)
    print(test_hyperband.outcomes)

    f = open('Output/Hyperband_DesignChoice_hls_phis_acc_allvaried_0.5_2_1_0.00743_relu_50_10epochs_withvgg_nodecay.txt', 'a')

    for i in test_hyperband.outcomes:

        hidden_ls = ""
        for j in i['hyperparams_hidden_layer_sizes']:
            hidden_ls += str(j) + " "

        hidden_ls = hidden_ls.strip()

        # Get the corresponding phi(s) for the hidden layer sizes

        phis = hls_phis_inv.get(hidden_ls)
        print(phis)

        for each_phis in phis.split('\t'):
            print(
                hidden_ls + '\t' + str(i['hyperparams_learning_rate']) + '\t' + str(each_phis) + '\t' + str(
                    acc_phis[each_phis]) + '\t' + str(1.0 - i['error']) + '\t' + str(i['s']) + '\n')
            f.write(hidden_ls + '\t' + str(i['hyperparams_learning_rate']) + '\t' + str(each_phis) + '\t' + str(acc_phis[each_phis]) + '\t' + str(1.0 - i['error']) + '\t' + str(i['s']) + '\n')


    f.close()
    end_time = time.clock()

    print(end_time - start_time)


if dshp == 4: # Just plotting!
    print('Just plotting!')

    filename = "/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Output/hls_phis_loss_bothphivaried.txt"
    filename_after = "/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Output/Hyperband_DesignChoice.txt"

    firstphi = []
    secondphi = []
    acc = []
    hls = []

    firstphi_after = []
    secondphi_after = []
    acc_after = []
    acc_before = []

    with open(filename, 'r') as f:
        lines = f.readlines()

        i = 0
        for line in lines:
            parts = line.split('\t')
            print(parts)

            firstphi.append(float(parts[1].strip()))
            secondphi.append(float(parts[2].strip()))

            acc.append(1.0 - float(parts[3].strip()))

    f.close()

    with open(filename_after, 'r') as f:
        lines = f.readlines()

        i = 0
        for line in lines:
            parts = line.split('\t')
            print(parts)

            firstphi_after.append(float(parts[2].strip()))
            secondphi_after.append(float(parts[3].strip()))

            acc_before.append(float(parts[4].strip()))
            acc_after.append(float(parts[5].strip()))

        # Original plot

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(firstphi, secondphi, acc, c='r', marker='o')
        ax.scatter(firstphi_after, secondphi_after, acc_before, c='b', marker='o')
        ax.scatter(firstphi_after, secondphi_after, acc_after, c='b', marker='s')

        ax.set_xlabel('First phi')
        ax.set_ylabel('Second phi')
        ax.set_zlabel('Accuracy')

        plt.show()





















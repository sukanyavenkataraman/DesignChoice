This folder contains ALL the programs written for this project, so there might be some redundancies.

Files -

*_input.py - For all kinds of inputs. Since they're all downloaded from different sources and in different formats, there's one file for each input. The processing is mostly the same for all of them -
1. Read the test and train data (medical images don't have this separation, so we explicitly separate them out)
2. Store them in the DataSet format which is defined in mrbi_input.py. This enables them to be fed to the tensorflow training pipeline

TF_deepnn.py - This contains classes to initialise the DataSet, and the neural network of a given architecture
To the DataSet class, we pass data type, if we're using vgg pretrained weights, and the number of convolutional layers of the vgg architecture we're using (if we use the pretrained weights)
Depending on the data set type, we get the data from *_input.py, as well as a reference to the next_batch function which feeds batches into the training pipeline.
We then pass this data set to the deepNN class, which, apart from initialising data properties (from the above data set), also initialises network properties such as batch size, activation function, learning rate, etc.
Models are saved in the model_dir, under the name 'model_name'
This class also contains functions to initialise hyperparameters (either from design choice's custom distribution of hidden layer sizes, or at random given highest and least values of hidden layer sizes)
The training happens in the function run_eval_hyperparam_withbs.
TF_deepnn only constructs fully connected networks. If using with vgg pretrained weights, then we only tune and train the fully connected layers


designchoice.py - This contains classes/functions which take in input data statistics (pixel size and number of classes), as phi values to give out hidden layer sizes. For now, the dropout rate is fixed and not dependent on the last phi (even though it should be according to the equations)
The phi values are given as a list of lists, so for each inner list, one set of hidden layer sizes are given as output.

hyperband_nn.py - This contains the hyperband code, which gets hyperparameters at random (from TF_deepnn.py), and applies the hyperband algorithm. We save the best result after every inner loop.

sigmoidNN.py - Is similar to TF_deepnn.py but uses TensorFlow's estimator API for fully connected networks. We don't use this anymore

vgg_pretrained.py - Contains functions to read from pretrained vgg weights (saved in .npy format) and return back the weights applied to a specific input set

vgg_save_int.py - Uses the above vgg_pretrained.py and *_input.py to get the nth layer outputs of a vgg pretrained model for the specified input, and save these into npy files. Please note that we ran into memory issues here because tensorflow has a maximum tensor size. Thus the npy output files are usually sharded. *_input.py takes care of this when loading the data

Test_Setup_Main.py - The main test pipeline for running design choice + hyperband. Sample run -
python Test_Setup_Main.py 10 -1 av45 0 > save_op.logs 2>&1 &
This runs the test pipeline on av45 data, with batchsize 10, starting learning rate as 0.1, and without using vgg pretrained weights (the '0' after av45 is for this)
The test setup iterates over different network depths (for now it is set to 3, but can add more depths to the list), different phi ratios as well as different last phis. The other phis are calculated from the last phi and the phi ratios. The phi ratios are used in 2 ways - a constant ratio, and a random ratio chosen from the given range of phi ratios.
Output logs are saved in Output_MainTestSetup_Medical (for medical data specifically. For other data, please change the scripts to write to Output_MainTestSetup)/<data_type>/dc_* for only deisgn choice runs, i.e, one epoch of training over all the hidden layer lengths that come out of design choice; and,
/dc_hb_* for design choice + hyperband
Models are saved in Models_Medical(again, for non-medical data please change the scripts to Models/)/<data_type>/DC_HB*/ for design choice + hyperband. We do not store the models for just one epoch of training since it isn't required. Please note that on medical data these models are HUGE(in the order of GBs) so they tend to occupy a lot of space. 
The configuration with which Test_Setup_Main is run is written to <data_type>.config. All configs are written to this, which is a bad design but honestly configs are written almost everywhere so we don't really need this.. just wanted to avoid having config files for each run :\
All logs that are written to Output_MainTest*/<data_type>/dc* are consolidated and written to <data_type>*_all_op.txt. This comes in handy when we want to plot stuff later on. Also makes sure that we have two copies of everything! Also, Output_Main*/<data_type>/dc* contains output logs written as per network depth, learning rate and specific phis, are useful when we want to only take a look at how a specific phi ratio performed.

Test_Setup_Main_HB.py - This is almost identical to the above in terms of where outputs are saved, etc., except that this runs for ONLY hyperband. For accurate comparisons, please make sure that the same script arguments are passed when running Test_Setup_Main_HB.py

Sample scripts for both of the above can be found as run_av45_tests.sh and run_fdg_tests.sh

All data is found under /home/sukanya/Documents/ (Bad location, yes. Need to move this to one of the other drives, but the code unfortunately has this in a lot of places :/)


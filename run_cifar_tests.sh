#!/bin/bash
#python Test_Setup_Main.py 32 -1 cifar10 1 Models_cifar_81_32 Output_MainTestSetup_81 250 64 3 > dc_hb_cifar10_bs_32_lr_-1_epochs_81_vgg_3_logs 2>&1 &
#sleep 5s
python Test_Setup_Main_HB.py 32 -1 cifar10 1 Models_cifar_81_32 Output_MainTestSetup_81 81 3> hb_cifar10_bs_32_lr_-1_epochs_81_vgg_3_logs 2>&1 &
#sleep 5s
#python Test_Setup_Main.py 64 -1 cifar10 1 Models_cifar_81_64 Output_MainTestSetup_81 250 64 3> dc_hb_cifar10_bs_64_lr_-1_epochs_81_vgg_3_logs 2>&1 &
#sleep 5s
#python Test_Setup_Main_HB.py 64 -1 cifar10 1 Models_cifar_81_64 Output_MainTestSetup_81 81 3> hb_cifar10_bs_64_lr_-1_epochs_81_vgg_3_logs 2>&1 &

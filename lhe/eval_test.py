##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0
seed = 2024

##########################################################################################
# Path Config

import os, random, math, time
import pytz
import argparse
import pprint as pp
import sys
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
# sys.path.insert(0, "..")  # for problem_def
# sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils import *
from Tester_train import Tester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
'problem':'CVRP',
#choices=["ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW","OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW","OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
}

model_params = {
    'model_type':'MOE_LIGHT',
#choices=["SINGLE", "MTL", "MOE", "MOE_LIGHT"]
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
'decoder_layer_num':1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
#choices=["argmax", "softmax"]
'num_experts':4,
'norm':'instance',
#choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"]
'norm_loc':'norm_last',
#choices=["norm_last", "norm_last"]
'topk':2,
'expert_loc':['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
#choices=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec']
'routing_level':'node',
#choices=["node", "instance", "problem"]
'routing_method':'input_choice',
#choices=["input_choice", "expert_choice", "soft_moe", "random"]s
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './checkpoints/',  # directory path of pre-trained model and log files saved.
        'epoch': 5000,  # epoch version of pre-trained model to laod.
    },
'sample_size':10,
    'test_episodes': 1000,
    'test_batch_size': 1000,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 100,
'test_set_path':None,
'fine_tune_epochs':0,
'fine_tune_episodes':10000,
'fine_tune_batch_size':64*2,
'fine_tune_aug_factor':1,
'lr':1e-4,
'weight_decay':1e-6,

'test_set_opt_sol_path':None,

    'test_data_load': {
        'enable': True,
        'filename': './vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # create_logger(**logger_params)
    # _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    # copy_all_src(tester.result_folder)

    avg_obj = tester.run()
    
    return avg_obj


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]
    
    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    
    if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-5000.pt")):
        raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")

    if mood == 'train':
        tester_params['test_episodes'] = 10
        tester_params['test_batch_size'] = 10
        env_params['problem_size'] = problem_size
        avg_obj = main()
        print("[*] Average:")
        print(avg_obj)
    
    else :
        tester_params['test_episodes'] =100

        tester_params['test_batch_size'] = 10
        env_params['problem_size'] = problem_size
        avg_obj = main()
        print(f"[*] Average for {problem_size}: {avg_obj}")
"""
File generating script for memory disaggregation experiments.
"""

import os, sys, getopt
import math, pprint, copy
import numpy as np
import utilities

####################################################################################################
# Simulation Parameters 
####################################################################################################

# Directory Setup
print("[Setup] Setup directory")
BASE_DIRECTORY = "/Users/bwu/src/calculon/"
OUTPUT_DIRECTORY = BASE_DIRECTORY + "temp/"
SYSTEM_DIRECTORY = BASE_DIRECTORY + "systems/"
MODEL_DIRECTORY = BASE_DIRECTORY + "models/"
ARCH_DIRECTORY = BASE_DIRECTORY + "examples/"
EXECUTION_DIRECTORY = BASE_DIRECTORY + "execution/"

# Base file setup
gpu = "h100_80g_nvl8"
model = "gpt3-175B"
arch = "3072_t4_p64_d12_mbs4_full"
system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
model_base_filename = MODEL_DIRECTORY + model + ".json"
arch_base_filename = ARCH_DIRECTORY + arch + ".json"
system_base = utilities.parseJSON(system_base_filename)
# model_base = utilities.parseJSON(model_base_filename)
# arch_base = utilities.parseJSON(arch_base_filename)

def generateMemBandwidthExperiment():
    # mem1 = local memory; mem2 = remote memory
    # llm models/megatron-1T.json examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80g.json -
    model_config_files = [model_base_filename]
    arch_config_files = [arch_base_filename]
    sys_config_files = []
    mem1_GBps_list = [100, 500, 1000]
    for mem1_GBps in mem1_GBps_list:
        new_system = copy.deepcopy(system_base)
        new_system["mem1"]["GBps"] = mem1_GBps
        filename = utilities.generateSystemFileNameString(new_system)
        utilities.dumpJSON(SYSTEM_DIRECTORY + filename, new_system)
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model, arch, gpu)
        sys_config_files.append(SYSTEM_DIRECTORY + filename + " " + output_dir + filename)
    config_files = utilities.cartesianProduct([model_config_files, arch_config_files, sys_config_files])
    return config_files 
    
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"he:",["exp="])
    except getopt.GetoptError:
        print('[HELP] python3 generate_experiment.py -e <experiment>')
        sys.exit(2)
    exp_type = ""
    for opt, arg in opts:
        if opt == '-h':
            print('python3 generate_experiment.py -exp_id <experiment_number>')
            sys.exit()
        elif opt in ("-e", "--exp"):
            exp_type = arg
    config_files = []
    if exp_type == "mem_bw":
        config_files = generateMemBandwidthExperiment()
    elif exp_type == "message_size":
        print('')
    else:
        print("[Error] Invalid Experiment Type")
    if config_files:
        bash_script = utilities.generateBashScript(EXECUTION_DIRECTORY, config_files)
        # utilities.generateExecutionScript(EXECUTION_DIRECTORY, bash_script_names)
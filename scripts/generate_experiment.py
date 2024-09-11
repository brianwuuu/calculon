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
gpu = "h100_80g_nvl8" # "h100_inf_nvl8", "h100_80g_nvl8" # "h100_80g_nvl8"
model = "gpt3-175B"
arch = "4096_t8_p64_d8_mbs4_full" # "3072_t4_p64_d12_mbs4_full"
system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
model_base_filename = MODEL_DIRECTORY + model + ".json"
arch_base_filename = ARCH_DIRECTORY + arch + ".json"
system_base = utilities.parseJSON(system_base_filename)
model_base = utilities.parseJSON(model_base_filename)
arch_base = utilities.parseJSON(arch_base_filename)

def generateSingleExperiment():
    model_config_file = model_base_filename
    arch_config_file = arch_base_filename
    sys_config_file = system_base_filename
    # modify system
    new_system = copy.deepcopy(system_base)
    # new_system["mem1"]["GBps"] = 2048
    new_system_filename = utilities.generateSystemFileNameString(new_system)
    utilities.dumpJSON(SYSTEM_DIRECTORY + new_system_filename, new_system)
    
    # modify model architecture
    new_arch = copy.deepcopy(arch_base)
    new_arch["num_procs"] = 2
    new_arch["tensor_par"] = 1
    new_arch["pipeline_par"] = 1
    new_arch["data_par"] = 2
    new_arch["weight_offload"] = False
    new_arch["activations_offload"] = False
    new_arch["optimizer_offload"] = False
    new_arch_filename = utilities.generateArchFileNameString(new_arch)
    utilities.dumpJSON(ARCH_DIRECTORY + new_arch_filename + ".json", new_arch)
    # create output directory
    output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model, new_arch_filename, gpu)
    sys_config_file =  SYSTEM_DIRECTORY + new_system_filename + " " + output_dir + new_system_filename
    return [[model_config_file, ARCH_DIRECTORY + new_arch_filename + ".json", sys_config_file]]

def generateNetworkSizeExperiment():
    # scale to 10,000
    system_filename = utilities.generateSystemFileNameString(system_base)
    arch_params = [(8,2,2,2), (16,2,4,2), (32,2,8,2), (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,4,32,4),
                  (1024,4,32,8), (2048,8,32,8), (4096,8,32,16), (8192,8,32,32), (16384,8,32,64), (32768,8,32,128)]
    config_files = []
    for num_proc, tensor_par, pipe_par, data_par in arch_params:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        # offloading
        new_arch["weight_offload"] = True
        new_arch["activations_offload"] = True
        new_arch["optimizer_offload"] = True
        arch_filename = utilities.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        utilities.dumpJSON(arch_config_file, new_arch)
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model, arch_filename, gpu)
        sys_config_file = system_base_filename + " " + output_dir + system_filename
        config_files.append([model_base_filename, arch_config_file, sys_config_file]) 
    return config_files

def generateModelSizeExperimentV1():
    # scale to 1T
    system_filename = utilities.generateSystemFileNameString(system_base)
    models = ["megatron-126M", "megatron-5B", "megatron-22B", "megatron-40B", "megatron-1T"]
    config_files = []
    for model in models:
        model_base_filename = MODEL_DIRECTORY + model + ".json"
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model, arch, gpu)
        sys_config_file = system_base_filename + " " + output_dir + system_filename
        config_files.append([model_base_filename, arch_base_filename, sys_config_file]) 
    return config_files

def generateModelSizeExperimentV2():
    # scale to 1T
    arch = "2_t1_p1_d2_mbs4_full" # "4096_t8_p32_d16_mbs4_full" # "4096_t4_p32_d32_mbs4_full" # "4096_t4_p32_d32_mbs4_wo_ao_oo_full"
    arch_base_filename = ARCH_DIRECTORY + arch + ".json"
    system_filename = utilities.generateSystemFileNameString(system_base)
    model_params = [(24576,192,128), (32768,205,160), (40960,213,192), (50176,224,224),
              (60416,236,256), (70656,245,288), (81920,256,320), (94208,268,352),
              (106496,277,384), (119808,288,416), (134144,299,448), (148480,309,480)
              ] # hidden, attn size, # blocks
    config_files = []
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 8192
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_filename = utilities.generateModelFileNameString(new_model)
        model_config_file = MODEL_DIRECTORY + model_filename + ".json"
        utilities.dumpJSON(model_config_file, new_model)
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_filename, arch, gpu)
        sys_config_file = system_base_filename + " " + output_dir + system_filename
        config_files.append([model_config_file, arch_base_filename, sys_config_file]) 
    return config_files

def generateMultipleExperiment():
    ########################################################################################
    #####################              Model           #####################################
    ########################################################################################
    model_params = [
              (12288,128,96) # GPT3-175B
            #   (24576,192,128), (32768,205,160), (40960,213,192), (50176,224,224),
            #   (60416,236,256), (70656,245,288), (81920,256,320), (94208,268,352),
            #   (106496,277,384), (119808,288,416), (134144,299,448), (148480,309,480)
            #   (24576,192,128)
            #   (148480,309,480)
              ] # hidden, attn size, # blocks
    model_config_files = []
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 2048 # 2048 (GPT-3), 8192
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_filename = utilities.generateModelFileNameString(new_model)
        model_config_file = MODEL_DIRECTORY + model_filename + ".json"
        utilities.dumpJSON(model_config_file, new_model)
        model_config_files.append(model_config_file)
    
    ########################################################################################
    #####################              Architecture       ##################################
    ########################################################################################
    arch_params = [
                    # (8,2,2,2)
                    # (16,2,4,2)
                    # (4096,16,32,8)
                    # (8192,32,32,8)
                    (8,2,2,2), (16,2,4,2), (32,2,8,2), (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,4,32,4),(1024,4,32,8), (2048,8,32,8), (4096,8,32,16)
                    # (1024,4,32,8), (2048,8,32,8), (4096,8,32,16), (8192,8,32,32), (16384,8,32,64), (32768,8,32,128)
                    # (1024,4,64,4), (2048,4,64,8), (4096,8,64,8), (8192,16,64,8), (16384,16,64,16), (32768,16,64,32)
                    # (4096,8,32,16), 
                    # (16384,8,32,64)
                    ]
    arch_config_files = []
    for num_proc, tensor_par, pipe_par, data_par in arch_params:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        new_arch["tensor_par_net"] = 0
        new_arch["pipeline_par_net"] = 1
        new_arch["data_par_net"] = 1
        # offloading
        new_arch["weight_offload"] = False
        new_arch["activations_offload"] = False
        new_arch["optimizer_offload"] = False
        arch_filename = utilities.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        utilities.dumpJSON(arch_config_file, new_arch)
        arch_config_files.append(arch_config_file)
        
    ########################################################################################
    ######################             System             ##################################
    ########################################################################################
    mem_params = [
                #   (256,50), (512,50), (1024,50), (2048,50),
                #   (256,100), (512,100), (1024,100), (2048,100)
                # (500,3072,10000,128,"PCIe"), (500,3072,10000,900,"NVLink"), (500,3072,10000,6100,"SiP")
                # (1000,3072,10000000,128), (1000,3072,10000000,900), (1000,3072,10000000,6100)
                # (80, 3000, 1000, 80)
                ] # mem1 GB, GBps; mem2 GB, GBps; 
    net_params = [
                # (10, 1e-5, 0.15, 50, 2e-5, 0.02)
                ] # net1 GBps, latency, proc_util; net2 GBps, latency, proc_util

    ##### Aug 5, 2024 ##### 
    # mem_bw and net_bw tradeoffs
    
    # physical dimensions
    # TODO: change dimension and number of GPU dies
    # TODO: change processor usage
    total_length_mm = 96 #
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    per_pic_bw_GBps = 2048 # 2033, 2048
    
    num_local_hbms = [1,2,4,6] # 0,1,2,4,6
    per_hbm_bw_GBps = 500
    per_hbm_capacity_GB = 16
    
    sys_config_files = []
    for num_local_hbm in num_local_hbms:
        local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
        local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
        local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
        num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
        max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
        # ratio
        mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
        print(mem_to_net_split)
        mem_params, net_params = [], []
        for mem_ratio, net_ratio in mem_to_net_split:
            print(mem_ratio*per_pic_bw_GBps, local_hbm_bw_GBps)
            mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
            # mem_params.append((80, 1500, 1000, mem_ratio*per_pic_bw_GBps))
            net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
            assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
        for mem_param, net_param in zip(mem_params, net_params):
            print(f"Mem: {mem_param}; Net: {net_param}")
            new_system = copy.deepcopy(system_base)
            new_system["mem1"]["GiB"], new_system["mem1"]["GBps"] = mem_param[0], mem_param[1]
            new_system["mem2"]["GiB"], new_system["mem2"]["GBps"] = mem_param[2], mem_param[3]
            new_system["networks"][0]["bandwidth"], new_system["networks"][0]["latency"], new_system["networks"][0]["processor_usage"] = net_param[0], net_param[1], net_param[2]
            new_system["networks"][1]["bandwidth"], new_system["networks"][1]["latency"], new_system["networks"][1]["processor_usage"] = net_param[3], net_param[4], net_param[5]
            new_system["networks"][0]["size"] = 32768
            new_system["processing_mode"] = "no_overlap" # roofline, no-overlap
            system_filename = utilities.generateSystemFileNameString(new_system)
            sys_config_file = SYSTEM_DIRECTORY + system_filename
            utilities.dumpJSON(sys_config_file, new_system)
            sys_config_files.append(sys_config_file)
            # output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_filename, arch, gpu)
            # sys_config_files.append(SYSTEM_DIRECTORY + system_filename + " " + output_dir + system_filename)
    
    config_files = utilities.cartesianProduct([model_config_files, arch_config_files, sys_config_files])
    new_config_files = []
    for model, arch, system in config_files:
        model_str = (model.split("/")[-1]).split(".")[0]
        arch_str = (arch.split("/")[-1]).split(".")[0]
        sys_str = system.split("/")[-1]
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_str, arch_str, gpu)
        output_str = system + " " + output_dir + sys_str
        new_config_files.append((model, arch, output_str))
    return new_config_files

def generateMemBandwidthExperiment():
    # mem1 = local memory; mem2 = remote memory
    # llm models/megatron-1T.json examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80g.json -
    model_config_files = [model_base_filename]
    arch_config_files = [arch_base_filename]
    sys_config_files = []
    mem1_GBps_list = [2048]
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
    elif exp_type == "single":
        config_files = generateSingleExperiment()
    elif exp_type == "network_size":
        config_files = generateNetworkSizeExperiment()
    elif exp_type == "model_size":
        config_files = generateModelSizeExperimentV2()
    elif exp_type == "multiple":
        config_files = generateMultipleExperiment()
    # elif exp_type == "cluster":
    #     config_files = generateClusterSizeExperiment()
    else:
        print("[Error] Invalid Experiment Type")
    if config_files:
        bash_script = utilities.generateBashScript(EXECUTION_DIRECTORY, config_files)
        # utilities.generateExecutionScript(EXECUTION_DIRECTORY, bash_script_names)
"""
File generating script for memory disaggregation experiments.
"""

import os, sys, getopt
import math, pprint, copy
import numpy as np
import utilities
from table import get_workload_info, get_cu_info, get_mem_info

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

def generate_model_configs(model_params, **kwargs):
    """
    Args:
        model_params: [(hidden, attn_size, num_blocks)]
    """
    model = "megatron-5B" # "gpt3-175B", 
    model_base_filename = MODEL_DIRECTORY + model + ".json"
    model_base = utilities.parseJSON(model_base_filename)
    seq_size = 2048 # 2048 (GPT-3), 8192
    model_config_files = []
    if not model_params: model_config_files.append(model_base_filename)
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = seq_size 
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_filename = utilities.generateModelFileNameString(new_model)
        model_config_file = MODEL_DIRECTORY + model_filename + ".json"
        utilities.dumpJSON(model_config_file, new_model)
        model_config_files.append(model_config_file)
    return model_config_files

def generate_arch_configs(arch_params, **kwargs):
    """
    Args:
        arch_params: [(num_procs, tensor_par, pipe_par, data_par)]
    """
    arch = "4096_t8_p64_d8_mbs4_full" # "3072_t4_p64_d12_mbs4_full"
    arch_base_filename = ARCH_DIRECTORY + arch + ".json"
    arch_base = utilities.parseJSON(arch_base_filename)
    arch_config_files = []
    if not arch_params: arch_config_files.append(arch_base_filename)
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
        # parallelization params
        new_arch["optimizer_sharding"] = False
        arch_filename = utilities.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        utilities.dumpJSON(arch_config_file, new_arch)
        arch_config_files.append(arch_config_file)
    return arch_config_files

def generate_system_configs(mem_params, net_params):
    """
    Args:
        mem_params: mem1 GB, GBps, ns; mem2 GB, GBps, ns
        net_params: net1 GBps, latency, proc_util; net2 GBps, latency, proc_util
    """
    gpu = "h100_80g_nvl8" # "h100_inf_nvl8", "h100_80g_nvl8" # "h100_80g_nvl8"
    system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
    system_base = utilities.parseJSON(system_base_filename)
    sys_config_files = []
    if not mem_params or not net_params: sys_config_files.append(system_base_filename)
    for mem_param, net_param in zip(mem_params, net_params):
            new_system = copy.deepcopy(system_base)
            new_system["mem1"]["GiB"], new_system["mem1"]["GBps"], new_system["mem1"]["ns"] = mem_param[0], mem_param[1], mem_param[2]
            new_system["mem2"]["GiB"], new_system["mem2"]["GBps"], new_system["mem2"]["ns"] = mem_param[3], net_param[4], net_param[5]
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
    return sys_config_files

def generate_output_files(model_config_files, arch_config_files, sys_config_files):
    config_files = utilities.cartesianProduct([model_config_files, arch_config_files, sys_config_files])
    new_config_files = []
    for model, arch, system in config_files:
        model_str = (model.split("/")[-1]).split(".")[0]
        arch_str = (arch.split("/")[-1]).split(".")[0]
        sys_str = system.split("/")[-1]
        output_dir = utilities.createOutputDirectory(OUTPUT_DIRECTORY, model_str, arch_str)
        output_str = system + " " + output_dir + sys_str
        new_config_files.append((model, arch, output_str))
    return new_config_files

def setup_experiment(mem_params, net_params, model_params, arch_params):
    sys_config_files = generate_system_configs(mem_params, net_params)
    model_config_files = generate_model_configs(model_params)
    arch_config_files = generate_arch_configs(arch_params)
    config_files = generate_output_files(model_config_files, arch_config_files, sys_config_files)
    bash_script = utilities.generateBashScript(EXECUTION_DIRECTORY, config_files)
    # utilities.generateExecutionScript(EXECUTION_DIRECTORY, bash_script_names)

def generate_sipam_experiment():
    model_params = [(4096, 128, 24)] # GPT3-175B: (12288,128,96)
    arch_params = [
                   (1,1,1,1)
                #    (8,2,2,2), (16,2,4,2), (32,2,8,2), 
                #    (64,2,16,2), (128,2,32,2), (256,2,32,4), 
                #    (512,4,32,4),(1024,4,32,8), (2048,8,32,8), (4096,8,32,16)
                   ]
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    per_pic_bw_GBps = 2048 # 2033, 2048
    
    num_local_hbms = [0] # 0,1,2,4,6
    per_hbm_bw_GBps = 500
    per_hbm_capacity_GB = 16
    
    mem_params, net_params = [], []
    for num_local_hbm in num_local_hbms:
        local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
        local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
        local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
        num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
        max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
        # ratio
        mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
        for mem_ratio, net_ratio in mem_to_net_split:
            print(mem_ratio, net_ratio)
            mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
            net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
            assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
    setup_experiment(mem_params, net_params, model_params, arch_params)

def get_required_mem_bw(arithmetic_intensity, peak_flops):
    """
    Args: 
        arithmetic_intensity: FLOPs/Byte
        peak_flops: FLOPs
    Return:
        Byte/s
    """
    return peak_flops / arithmetic_intensity

def get_num_required_mem_units(req_mem_bw_GBps, mem_bw_GBps):
    return req_mem_bw_GBps / mem_bw_GBps

def get_num_required_gpu(workload_size_GB, per_gpu_mem_cap_GB, pow_of_2=False):
    num_required_gpu = np.ceil(workload_size_GB / per_gpu_mem_cap_GB)
    if pow_of_2: return utilities.nearest_pow_of_2(num_required_gpu)
    return num_required_gpu

def get_par_params(num_gpu):
    """ Try to map to multiple of 4 and power of 2
    TP, PP, DP
    """
    return (1,1,num_gpu)
    
def optimize_flow():
    # variables
    gpu = "h100"
    workload = "megatron-5B"
    mem = "HBM3"
    datatype = "float16"
    
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bw_GBps = 2048
    
    workload_info = get_workload_info(workload)
    cu_info = get_cu_info(gpu, datatype)
    mem_info = get_mem_info(mem)
    
    required_mem_bw_GBps = get_required_mem_bw(workload_info["ai"], cu_info["matrix"]) / 1e9
    num_req_mu_per_gpu = get_num_required_mem_units(required_mem_bw_GBps, mem_info["bw_GBps"])
    per_gpu_mem_cap_GB = num_req_mu_per_gpu * mem_info["cap_GB"]
    per_gpu_mem_bw_GBps = num_req_mu_per_gpu * mem_info["bw_GBps"]
    num_gpu = get_num_required_gpu(workload_info["size_GB"], per_gpu_mem_cap_GB)
    
    num_mem_pic_per_gpu = np.ceil(per_gpu_mem_bw_GBps / per_pic_bw_GBps)
    num_net_pic_per_gpu = (total_length_mm - (per_pic_length_mm * num_mem_pic_per_gpu)) // per_pic_length_mm # round down
    net_bw_GBps = num_net_pic_per_gpu * per_pic_bw_GBps
    par_params = get_par_params(num_gpu)
    
    mem_params = [(per_gpu_mem_cap_GB, per_gpu_mem_bw_GBps, mem_info["lat_ns"], 0,0,0)] # mem1 GB, GBps, ns; mem2 GB, GBps, ns
    net_params = [(net_bw_GBps, 0, 0.15, net_bw_GBps, 0, 0.15)] # net1 GBps, latency, proc_util; net2 GBps, latency, proc_util
    model_params = [] # [(hidden, attn_size, num_blocks)]
    arch_params = [(num_gpu, par_params[0], par_params[1], par_params[2])] # [(num_procs, tensor_par, pipe_par, data_par)]
    setup_experiment(mem_params, net_params, model_params, arch_params)

if __name__ == "__main__":
    # generate_sipam_experiment()
    optimize_flow()
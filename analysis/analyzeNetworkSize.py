"""
Analysis script for memory disaggregation simulations.
"""

import sys, os, getopt
import pprint, math, copy
import numpy as np
import collections
import plot_util
import util

####################################################################################################
# Analysis Parameters 
####################################################################################################

# Directory Setup
print("[Analysis] Start ...")
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
system_base = util.parseJSON(system_base_filename)
model_base = util.parseJSON(model_base_filename)
arch_base = util.parseJSON(arch_base_filename)

def analyzeTiming():
    system_filename = util.generateSystemFileNameString(system_base)
    arch_param = [(8,2,2,2), (16,2,4,2), (32,2,8,2), (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,4,32,4),
                  (1024,4,32,8), (2048,8,32,8), (4096,8,32,16), (8192,8,32,32), (16384,8,32,64), (32768,8,32,128)]
    
    # [
    #     (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,2,64,4), # (8,2,2,2), (16,2,4,2), (32,2,8,2), 
    #     (1024,2,64,8), (2048,4,64,8), (4096,8,64,8), (8192,8,64,16), (16384,8,128,16), (32768,8,128,32)
    # ]
    job_stats = collections.defaultdict(list)
    for num_proc, tensor_par, pipe_par, data_par in arch_param:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        # offloading
        new_arch["weight_offload"] = True
        new_arch["activations_offload"] = True
        new_arch["optimizer_offload"] = True
        arch_filename = util.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
        assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
        exec_output = util.parseJSON(output_dir + system_filename)
        job_stats["FW Pass"].append(exec_output["Batch FW time"])
        job_stats["BW Pass"].append(exec_output["Batch BW time"])
        job_stats["Optim Step"].append(exec_output["Batch optim time"])
        job_stats["PP Bubble"].append(exec_output["Batch bubble overhead"])
        job_stats["FW Recompute"].append(exec_output["Batch recompute overhead"] + exec_output["Batch recomm overhead"])
        job_stats["TP Comm"].append(exec_output["Batch TP comm time on link"])
        job_stats["PP Comm"].append(exec_output["Batch PP comm time on link"])
        job_stats["DP Comm"].append(exec_output["Batch DP comm time on link"])
    pprint.pprint(job_stats)
    x_ = {"label": "# GPUs", "data": [str(x[0]) for x in arch_param], "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": 10, "limit": None}
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    
def analyzeMemoryUsage():
    system_filename = util.generateSystemFileNameString(system_base)
    arch_param = [(8,2,2,2), (16,2,4,2), (32,2,8,2), (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,4,32,4),
                  (1024,4,32,8), (2048,8,32,8), (4096,8,32,16), (8192,8,32,32), (16384,8,32,64), (32768,8,32,128)]
    # arch_param = [(64,2,16,2), (128,2,32,2), (256,2,32,4), (512,2,64,4), # (8,2,2,2), (16,2,4,2), (32,2,8,2), 
    #               (1024,2,64,8), (2048,4,64,8), (4096,8,64,8), (8192,8,64,16), (16384,8,128,16), (32768,8,128,32)
    #              ]
    job_stats = collections.defaultdict(list)
    for num_proc, tensor_par, pipe_par, data_par in arch_param:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        # offloading
        new_arch["weight_offload"] = True
        new_arch["activations_offload"] = True
        new_arch["optimizer_offload"] = True
        arch_filename = util.generateArchFileNameString(new_arch)
        arch_config_file = ARCH_DIRECTORY + arch_filename + ".json"
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
        if os.path.isfile(output_dir + system_filename):
            exec_output = util.parseJSON(output_dir + system_filename)
            job_stats["Weights"].append(util.toBytes(exec_output["Weights"])  / (10 ** 9))
            job_stats["Activations"].append((util.toBytes(exec_output["Act"]) + util.toBytes(exec_output["Act CP"])) / (10 ** 9))
            job_stats["Act Gradients"].append(util.toBytes(exec_output["Act grad"]) / (10 ** 9))
            job_stats["Weight Gradients"].append(util.toBytes(exec_output["Weight grad"]) / (10 ** 9))
            job_stats["Optimizer Space"].append(util.toBytes(exec_output["Optim space"]) / (10 ** 9))
    pprint.pprint(job_stats)
    x_ = {"label": "# GPUs", "data": [str(x[0]) for x in arch_param], "log": None, "limit": None}
    y_ = {"label": "HBM Consumption (GB)", "data": job_stats, "log": None, "limit": None}
    fig_path = "/Users/bwu/Downloads/mem_usage.png"
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = fig_path)

def main():
    # analyzeTiming()
    analyzeMemoryUsage()
        
if __name__ == '__main__':
    main()
"""
Analysis script for memory disaggregation simulations.
"""

import sys, os
import pprint, copy
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
gpu = "h100_inf_nvl8" # "h100_80g_nvl8"
model = "gpt3-175B"
arch = "4096_t8_p64_d8_mbs4_full" # "3072_t4_p64_d12_mbs4_full"
system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
model_base_filename = MODEL_DIRECTORY + model + ".json"
arch_base_filename = ARCH_DIRECTORY + arch + ".json"
system_base = util.parseJSON(system_base_filename)
model_base = util.parseJSON(model_base_filename)
arch_base = util.parseJSON(arch_base_filename)

def analyzeTimingV1():
    system_filename = util.generateSystemFileNameString(system_base)
    models = ["megatron-126M", "megatron-5B", "megatron-22B", "megatron-40B", "megatron-1T"]
    job_stats = collections.defaultdict(list)
    for model in models:
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch + "/" + gpu + "/"
        if os.path.isfile(output_dir + system_filename):
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
    x_ = {"label": "# GPUs", "data": [x.split("-")[1] for x in models], "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": 10, "limit": None}
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    
def analyzeMemoryUsageV1():
    system_filename = util.generateSystemFileNameString(system_base)
    models = ["megatron-126M", "megatron-5B", "megatron-22B", "megatron-40B", "megatron-1T"]
    job_stats = collections.defaultdict(list)
    for model in models:
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch + "/" + gpu + "/"
        if os.path.isfile(output_dir + system_filename):
            exec_output = util.parseJSON(output_dir + system_filename)
            job_stats["Weights"].append(util.toBytes(exec_output["Weights"])  / (10 ** 9))
            job_stats["Activations"].append((util.toBytes(exec_output["Act"]) + util.toBytes(exec_output["Act CP"])) / (10 ** 9))
            job_stats["Act Gradients"].append(util.toBytes(exec_output["Act grad"]) / (10 ** 9))
            job_stats["Weight Gradients"].append(util.toBytes(exec_output["Weight grad"]) / (10 ** 9))
            job_stats["Optimizer Space"].append(util.toBytes(exec_output["Optim space"]) / (10 ** 9))
    pprint.pprint(job_stats)
    x_ = {"label": "# GPUs", "data": [x.split("-")[1] for x in models], "log": None, "limit": None}
    y_ = {"label": "HBM Consumption (GB)", "data": job_stats, "log": 10, "limit": None}
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    
def analyzeTimingV2():
    arch = "4096_t8_p64_d8_mbs4_full"
    system_filename = util.generateSystemFileNameString(system_base)
    model_params = [(24576,192,128,"1T"), (32768,205,160,"2T"), (40960,213,192,"4T"), (50176,224,224,"7T"),
              (60416,236,256,"11T"), (70656,245,288,"18T"), (81920,256,320,"26T"), (94208,268,352,"37T"),
              (106496,277,384,"53T"), (119808,288,416,"72T"), (134144,299,448,"96T"), (148480,309,480,"128T")
              ] # hidden, attn size, # blocks
    job_stats = collections.defaultdict(list)
    model_sizes = []
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 8192
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_sizes.append(param[3])
        model_filename = util.generateModelFileNameString(new_model)
        output_dir = OUTPUT_DIRECTORY + model_filename + "/" + arch + "/" + gpu + "/"
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
    x_ = {"label": "Model Sizes", "data": model_sizes, "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": 10, "limit": None}
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    
def analyzeMemoryUsageV2():
    arch = "2_t1_p1_d2_mbs4_full" # "4096_t8_p64_d8_mbs4_full"
    system_filename = util.generateSystemFileNameString(system_base)
    model_params = [(24576,192,128,"1T"), (32768,205,160,"2T"), (40960,213,192,"4T"), (50176,224,224,"7T"),
              (60416,236,256,"11T"), (70656,245,288,"18T"), (81920,256,320,"26T"), (94208,268,352,"37T"),
              (106496,277,384,"53T"), (119808,288,416,"72T"), (134144,299,448,"96T"), (148480,309,480,"128T")
              ] # hidden, attn size, # blocks
    job_stats = collections.defaultdict(list)
    model_sizes = []
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 8192
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_sizes.append(param[3])
        model_filename = util.generateModelFileNameString(new_model)
        output_dir = OUTPUT_DIRECTORY + model_filename + "/" + arch + "/" + gpu + "/"
        assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
        exec_output = util.parseJSON(output_dir + system_filename)
        # job_stats["Weights"].append(util.toBytes(exec_output["Weights"])  / (10 ** 12))
        # job_stats["Activations"].append((util.toBytes(exec_output["Act"]) + util.toBytes(exec_output["Act CP"])) / (10 ** 12))
        # job_stats["Act Gradients"].append(util.toBytes(exec_output["Act grad"]) / (10 ** 12))
        # job_stats["Weight Gradients"].append(util.toBytes(exec_output["Weight grad"]) / (10 ** 12))
        # job_stats["Optimizer Space"].append(util.toBytes(exec_output["Optim space"]) / (10 ** 12))
        total_mem = 0
        # total_mem += util.toBytes(exec_output["Weights"])  / (10 ** 12)
        # total_mem += (util.toBytes(exec_output["Act"]) + util.toBytes(exec_output["Act CP"])) / (10 ** 12)
        # total_mem += util.toBytes(exec_output["Act grad"]) / (10 ** 12)
        # total_mem += util.toBytes(exec_output["Weight grad"]) / (10 ** 12)
        total_mem += util.toBytes(exec_output["Optim space"]) / (10 ** 12)
        job_stats["mem"].append(total_mem)
    pprint.pprint(job_stats)
    x_ = {"label": "Model Sizes", "data": model_sizes, "log": None, "limit": None}
    y_ = {"label": "Mem Consumption (TB)", "data": job_stats, "log": None, "limit": None}
    path = "/Users/bwu/Desktop/mem_requirement.png"
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    
def analyzeEfficiencyV2():
    arch = "4096_t8_p32_d16_mbs4_wo_ao_oo_full" # "4096_t8_p64_d8_mbs4_full"
    system_filename = util.generateSystemFileNameString(system_base)
    model_params = [
              (24576,192,128,"1T"), (32768,205,160,"2T"), (40960,213,192,"4T"), (50176,224,224,"7T"),
            #   (60416,236,256,"11T"), (70656,245,288,"18T"), (81920,256,320,"26T"), (94208,268,352,"37T"),
            #   (106496,277,384,"53T"), (119808,288,416,"72T"), (134144,299,448,"96T"), (148480,309,480,"128T")
              ] # hidden, attn size, # blocks
    job_stats = collections.defaultdict(list)
    model_sizes = []
    for param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 8192
        new_model["hidden"] = param[0]
        new_model["attn_size"] = param[1]
        new_model["num_blocks"] = param[2]
        new_model["feedforward"] = 4 * param[0]
        new_model["attn_heads"] = param[2]
        model_sizes.append(param[3])
        model_filename = util.generateModelFileNameString(new_model)
        output_dir = OUTPUT_DIRECTORY + model_filename + "/" + arch + "/" + gpu + "/"
        if os.path.isfile(output_dir + system_filename):
            exec_output = util.parseJSON(output_dir + system_filename)
            job_stats["Compute efficiency"].append(exec_output["Compute efficiency"])
            job_stats["System efficiency"].append(exec_output["System efficiency"])
            job_stats["Total efficiency"].append(exec_output["Total efficiency"])
        else:
            print("[Analysis] File does not exist: {}".format(output_dir + system_filename))
            job_stats["Compute efficiency"].append(0)
            job_stats["System efficiency"].append(0)
            job_stats["Total efficiency"].append(0)
    pprint.pprint(job_stats)
    x_ = {"label": "Model Sizes", "data": model_sizes, "log": None, "limit": None}
    y_ = {"label": "Efficiency (%)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")

def main():
    # analyzeTimingV2()
    analyzeMemoryUsageV2()
    # analyzeEfficiencyV2()
        
if __name__ == '__main__':
    main()
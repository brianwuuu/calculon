"""
Analysis script for memory disaggregation simulations.
"""

import sys, os, pprint
import pprint, math, copy
import numpy as np
from collections import defaultdict
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

#  Base file setup
gpu = "h100_80g_nvl8" # "h100_inf_nvl8" # "h100_80g_nvl8"
model = "12288h_49152ff_2048ss_96ah_128as_96nb" # "gpt3-175B"
arch_param = (16,2,4,2) # (4096,16,32,8), (8,2,2,2), (8192,32,32,8), (16,2,4,2)
arch = f"{arch_param[0]}_t{arch_param[1]}_p{arch_param[2]}_d{arch_param[3]}_mbs4_wo_ao_oo_full"
system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
model_base_filename = MODEL_DIRECTORY + model + ".json"
arch_base_filename = ARCH_DIRECTORY + arch + ".json"
system_base = util.parseJSON(system_base_filename)
model_base = util.parseJSON(model_base_filename)
arch_base = util.parseJSON(arch_base_filename)

def analyzeHBMBandwidth():
    ## analyze time vs local hbm capacity 
    #
    # physical dimensions
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    per_hbm_capacity_GB = 16
    per_pic_bw_GBps = 2048 # 2033, 2048
    
    # hbm
    num_local_hbms = [1,2,4,6]
    per_hbm_bws_GBps = [200, 300, 600, 1000, 2048, 4000]
    job_stats = defaultdict(list)
    for per_hbm_bw_GBps in per_hbm_bws_GBps:
        for num_local_hbm in num_local_hbms:
            local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
            local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
            local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
            num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
            max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
            mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
            mem_params, net_params = [], []
            for mem_ratio, net_ratio in mem_to_net_split:
                mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
                # mem_params.append((80, mem_ratio*per_pic_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
                net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
                assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
            end_time = []
            for mem_param, net_param in zip(mem_params, net_params):
                print(mem_param, net_param)
                new_system = copy.deepcopy(system_base)
                new_system["mem1"]["GiB"], new_system["mem1"]["GBps"] = mem_param[0], mem_param[1]
                new_system["mem2"]["GiB"], new_system["mem2"]["GBps"] = mem_param[2], mem_param[3]
                new_system["networks"][0]["bandwidth"], new_system["networks"][0]["latency"], new_system["networks"][0]["processor_usage"] = net_param[0], net_param[1], net_param[2]
                new_system["networks"][1]["bandwidth"], new_system["networks"][1]["latency"], new_system["networks"][1]["processor_usage"] = net_param[3], net_param[4], net_param[5]
                new_system["networks"][0]["size"] = 32768
                new_system["processing_mode"] = "no-overlap" # roofline, no-overlap
                system_filename = util.generateSystemFileNameString(new_system)
            
                # offloading
                new_arch = copy.deepcopy(arch_base)
                new_arch["weight_offload"] = False
                new_arch["activations_offload"] = False
                new_arch["optimizer_offload"] = False
                arch_filename = util.generateArchFileNameString(new_arch)
                output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
                assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                exec_output = util.parseJSON(output_dir + system_filename)
                end_time.append(exec_output["Batch total time"])
            job_stats[f"{per_hbm_bw_GBps} GBps"].append(np.min(end_time))
    pprint.pprint(job_stats)
    local_hbm_capacities = [num * per_hbm_capacity_GB for num in num_local_hbms]
    x_ = {"label": "Local HBM Capacity (GB)", "data": local_hbm_capacities, "log":2, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")


def analyzePICBandwidth():
    ## analyze time vs local hbm capacity 
    #
    # physical dimensions
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    per_hbm_bw_GBps = 500
    per_hbm_capacity_GB = 16
    
    # hbm
    num_local_hbms = [1,2,4,6]
    per_pic_bws_GBps = [200, 300, 600, 1000, 2048, 4000] # 2033, 2048
    job_stats = defaultdict(list)
    for per_pic_bw_GBps in per_pic_bws_GBps:
        for num_local_hbm in num_local_hbms:
            local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
            local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
            local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
            num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
            max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
            mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
            mem_params, net_params = [], []
            for mem_ratio, net_ratio in mem_to_net_split:
                mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
                # mem_params.append((80, mem_ratio*per_pic_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
                net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
                assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
            end_time = []
            for mem_param, net_param in zip(mem_params, net_params):
                new_system = copy.deepcopy(system_base)
                new_system["mem1"]["GiB"], new_system["mem1"]["GBps"] = mem_param[0], mem_param[1]
                new_system["mem2"]["GiB"], new_system["mem2"]["GBps"] = mem_param[2], mem_param[3]
                new_system["networks"][0]["bandwidth"], new_system["networks"][0]["latency"], new_system["networks"][0]["processor_usage"] = net_param[0], net_param[1], net_param[2]
                new_system["networks"][1]["bandwidth"], new_system["networks"][1]["latency"], new_system["networks"][1]["processor_usage"] = net_param[3], net_param[4], net_param[5]
                new_system["networks"][0]["size"] = 32768
                new_system["processing_mode"] = "no-overlap" # roofline, no-overlap
                system_filename = util.generateSystemFileNameString(new_system)
            
                # offloading
                new_arch = copy.deepcopy(arch_base)
                new_arch["weight_offload"] = False
                new_arch["activations_offload"] = False
                new_arch["optimizer_offload"] = False
                arch_filename = util.generateArchFileNameString(new_arch)
                output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
                assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                exec_output = util.parseJSON(output_dir + system_filename)
                end_time.append(exec_output["Batch total time"])
            job_stats[f"{per_pic_bw_GBps} GBps"].append(np.max(end_time))
    pprint.pprint(job_stats)
    local_hbm_capacities = [num * per_hbm_capacity_GB for num in num_local_hbms]
    x_ = {"label": "Local HBM Capacity (GB)", "data": local_hbm_capacities, "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")

def analyzeMemNetRatio():
    # physical dimensions
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    
    # hbm
    num_local_hbm = 4 # 0, 1, 2, 4
    per_hbm_capacity_GB = 16
    per_hbm_bws_GBps = [128, 256, 512, 1024, 2048] # 300, 600, 1000, 2000
    job_stats = defaultdict(list)
    for per_hbm_bw_GBps in per_hbm_bws_GBps:
        local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
        local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
        local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
    
        # pic
        num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
        per_pic_bw_GBps = 2048 # 2033, 2048
        max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
        # ratio
        mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
        mem_params, net_params = [], []
        for mem_ratio, net_ratio in mem_to_net_split:
            mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
            net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
            assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
        
        for mem_param, net_param in zip(mem_params, net_params):
            print(mem_param, net_param)
            new_system = copy.deepcopy(system_base)
            new_system["mem1"]["GiB"], new_system["mem1"]["GBps"] = mem_param[0], mem_param[1]
            new_system["mem2"]["GiB"], new_system["mem2"]["GBps"] = mem_param[2], mem_param[3]
            new_system["networks"][0]["bandwidth"], new_system["networks"][0]["latency"], new_system["networks"][0]["processor_usage"] = net_param[0], net_param[1], net_param[2]
            new_system["networks"][1]["bandwidth"], new_system["networks"][1]["latency"], new_system["networks"][1]["processor_usage"] = net_param[3], net_param[4], net_param[5]
            new_system["networks"][0]["size"] = 32768
            new_system["processing_mode"] = "no-overlap" # roofline, no-overlap
            system_filename = util.generateSystemFileNameString(new_system)
        
            # offloading
            new_arch = copy.deepcopy(arch_base)
            new_arch["weight_offload"] = False
            new_arch["activations_offload"] = False
            new_arch["optimizer_offload"] = False
            arch_filename = util.generateArchFileNameString(new_arch)
            output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
            assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
            exec_output = util.parseJSON(output_dir + system_filename)
            job_stats[f"{per_hbm_bw_GBps} GBps"].append(exec_output["Batch total time"])
    pprint.pprint(job_stats)
    mem_to_net_str = [f"{x[0]}:{x[1]}" for x in mem_to_net_split]
    x_ = {"label": "Memory-Network Ratio", "data": mem_to_net_str, "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    # plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "")
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")
    
def analyzeMemUsed():
    # physical dimensions
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    
    # hbm
    num_local_hbm = 1 # 0, 1, 2, 4, 6
    per_hbm_capacity_GB = 16
    per_hbm_bw_GBps = 500 # 300, 600, 1000, 2000
    local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
    local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
    local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
    
    # pic
    num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
    per_pic_bw_GBps = 2048 # 2033, 2048
    max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
    mem_ratio, net_ratio = 1, 9
    
    # sys
    new_system = copy.deepcopy(system_base)
    new_system["mem1"]["GiB"], new_system["mem1"]["GBps"] = local_hbm_capacity_GB, local_hbm_bw_GBps
    new_system["mem2"]["GiB"], new_system["mem2"]["GBps"] = 1000, mem_ratio*per_pic_bw_GBps
    new_system["networks"][0]["bandwidth"], new_system["networks"][0]["latency"], new_system["networks"][0]["processor_usage"] = net_ratio*per_pic_bw_GBps, 1e-5, 0.15
    new_system["networks"][1]["bandwidth"], new_system["networks"][1]["latency"], new_system["networks"][1]["processor_usage"] = net_ratio*per_pic_bw_GBps, 1e-5, 0.15
    new_system["networks"][0]["size"] = 32768
    new_system["processing_mode"] = "no-overlap" # roofline, no-overlap
    system_filename = util.generateSystemFileNameString(new_system)
    
    # arch_params
    arch_params = [(64,2,16,2), (128,2,32,2), 
                  (256,2,32,4), (512,4,32,4),(1024,4,32,8), (2048,8,32,8), (4096,8,32,16)] # (8,2,2,2), (16,2,4,2), (32,2,8,2), 
    
    job_stats = defaultdict(list)
    for num_proc, tensor_par, pipe_par, data_par in arch_params:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        new_arch["weight_offload"] = False
        new_arch["activations_offload"] = False
        new_arch["optimizer_offload"] = False
        arch_filename = util.generateArchFileNameString(new_arch)
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/" + gpu + "/"
        assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
        exec_output = util.parseJSON(output_dir + system_filename)
        job_stats[f"Local"].append(float(exec_output["Mem tier1 capacity used"].split(" ")[0]))
        job_stats[f"Remote"].append(float(exec_output["Mem tier2 capacity used"].split(" ")[0]))
    network_sizes = [str(x[0]) for x in arch_params]
    pprint.pprint(job_stats)
    x_ = {"label": "Network Size", "data": network_sizes, "log": None, "limit": None}
    y_ = {"label": "Memory Usage Per GPU (GB)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiColStackedBarChart(x = x_, y = y_, path = "", title=f"{num_local_hbm} HBM")

def main():
    # analyzeMemNetRatio()
    # analyzePICBandwidth()
    # analyzeHBMBandwidth()
    analyzeMemUsed()
    
        
if __name__ == '__main__':
    main()
    
# job_stats["FW Pass"].append(exec_output["Batch FW time"])
# job_stats["BW Pass"].append(exec_output["Batch BW time"])
# job_stats["Optim Step"].append(exec_output["Batch optim time"])
# job_stats["FW Recompute"].append(exec_output["Batch recompute overhead"] + exec_output["Batch recomm overhead"])
# job_stats["PP Comm"].append(exec_output["Batch bubble overhead"] + exec_output["Batch PP comm time on link"])
# job_stats["TP Comm"].append(exec_output["Batch TP comm time on link"])
# job_stats["DP Comm"].append(exec_output["Batch DP comm time on link"])
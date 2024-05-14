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
# model_base = utilities.parseJSON(model_base_filename)
# arch_base = utilities.parseJSON(arch_base_filename)

def analyzeMemBandwidth():
    gpu = "h100_80g_nvl8"
    model = "gpt3-175B"
    arch = "3072_t4_p64_d12_mbs4_full"
    job_stats = collections.defaultdict(list)
    mem1_GBps_list = [100, 500, 1000]
    for mem1_GBps in mem1_GBps_list:
        new_system = copy.deepcopy(system_base)
        new_system["mem1"]["GBps"] = mem1_GBps
        filename = util.generateSystemFileNameString(new_system)
        output_dir = OUTPUT_DIRECTORY + model + "/" + arch + "/" + gpu + "/"
        exec_output = util.parseJSON(output_dir + filename)
        job_stats["1"].append(exec_output["total_time"])
    x_ = {"label": "Local Memory Bandwidth", "data": mem1_GBps_list, "log": None, "limit": None}
    y_ = {"label": "Total Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")

def main():
    print("[ANALYSIS] Starting analysis ...")
    try:
        opts, args = getopt.getopt(sys.argv[1:],"he:c:",["exp="])
    except getopt.GetoptError:
        print('python3 analysis.py -e <experiment>')
        sys.exit(2)
    exp_type = ""
    for opt, arg in opts:
        if opt == '-h':
            print('python3 analysis.py -e <experiment>')
            sys.exit()
        elif opt in ("-e", "--exp"):
            exp_type = arg
    if exp_type == "mem_bw":
        analyzeMemBandwidth()
        
if __name__ == '__main__':
    main()
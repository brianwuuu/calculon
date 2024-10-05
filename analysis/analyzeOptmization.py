"""
Analysis script for memory disaggregation simulations.
"""

import sys, os
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(scripts_path)
import util
import pprint
import plot_util
from collections import defaultdict
from resource_optimization import optimize_mem_net, baseline_mem_net
from config_generation import get_confile_filenames


####################################################################################################
# Analysis Parameters 
####################################################################################################

print("[Analysis] Start ...")
BASE_DIRECTORY = "/Users/bwu/src/calculon/"
OUTPUT_DIRECTORY = BASE_DIRECTORY + "temp/"
SYSTEM_DIRECTORY = BASE_DIRECTORY + "systems/"
MODEL_DIRECTORY = BASE_DIRECTORY + "models/"
ARCH_DIRECTORY = BASE_DIRECTORY + "examples/"
EXECUTION_DIRECTORY = BASE_DIRECTORY + "execution/"

def get_config_str(configs : tuple):
    model_str = ((configs[0].split('/')[-1]).split('.'))[0]
    arch_str = ((configs[1].split('/')[-1]).split('.'))[0]
    sys_str = configs[2].split('/')[-1]
    return model_str, arch_str, sys_str

def analyzeIterTime():
    gpu = "h100"
    workloads = [
                 "megatron-126M", 
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "gpt3-13B",
                 "gpt3-175B"
                 ]
    mems = ["HBM2E"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048]
    
    job_stats = defaultdict(list)
    for workload in workloads:
        for mem in mems:
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["SiP Optim"].append(exec_output["Batch total time"])

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["Baseline Optim"].append(exec_output["Batch total time"])

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2.2,2), bbox_to_anchor=(0.4,0.5))

def analyzeEfficency():
    gpu = "h100"
    workloads = [
                 "megatron-126M", 
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "gpt3-13B",
                 "gpt3-175B"
                 ]
    mems = ["HBM2E"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048]
    
    job_stats = defaultdict(list)
    for workload in workloads:
        for mem in mems:
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["SiP Optim"].append(exec_output["Total efficiency"])

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["Baseline Optim"].append(exec_output["Total efficiency"])

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2.2,2), bbox_to_anchor=(0.4,0.5))


def analyzeEfficiency():
    gpu = "h100"
    workloads = [
                 "megatron-126M", 
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "gpt3-13B",
                 "gpt3-175B"
                 ] 
    mems = ["HBM2E"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048]
    
    job_stats = defaultdict(list)
    for workload in workloads:
        for mem in mems:
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["SiP Optim"].append(exec_output["Total efficiency"])

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["Baseline Optim"].append(exec_output["Total efficiency"])

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2.2,2), bbox_to_anchor=(0.4,0.5))

if __name__ == '__main__':
    analyzeIterTime()
    # analyzeEfficency()
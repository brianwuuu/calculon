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
from table import get_mem_info, get_workload_info, get_cu_info


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
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n530M": "megatron-530M",
        "Meg\n1B": "megatron-1B",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
        # "GPT3\n13B": "gpt3-13B",
        # "Meg\n1T": "megatron-1T",
    }
    mems = ["HBM2E"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [337.5] # 2048, 337.5
    worktype = "inference"
    
    job_stats = defaultdict(list)
    for workload in workloads.values():
        for mem in mems:
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps, worktype=worktype)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    norm_time = exec_output["Batch total time"]
                    job_stats["SiPAM"].append(exec_output["Batch total time"]/norm_time)

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["Baseline"].append(exec_output["Batch total time"]/norm_time)

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads.keys(), "log":None, "limit": None}
    y_ = {"label": "Norm. Iteration Time", "data": job_stats, "log":None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(3,2), bbox_to_anchor=(0.69,0.75), ncol=1)

def analyzeEfficency():
    gpu = "h100"
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
    }
    mems = ["HBM2E"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5
    worktype = "training"
    efftype = "Compute efficiency"
    
    job_stats = defaultdict(list)
    for workload in workloads.values():
        for mem in mems:
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps,worktype=worktype)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    # norm_eff = exec_output[efftype]
                    job_stats["Optimized"].append(exec_output[efftype] / 1)

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    job_stats["Baseline"].append(exec_output[efftype] / 1)

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads.keys(), "log":None, "limit": None}
    y_ = {"label": "System Efficiency (%)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2.4,1.8), bbox_to_anchor=(0.05,0.98), ncol=2)

def analyzeGPUHour():
    gpu = "h100"
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n530M": "megatron-530M",
        "Meg\n1B": "megatron-1B",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
        # "GPT3\n13B": "gpt3-13B",
        # "Meg\n1T": "megatron-1T",
    }
    mems = ["HBM2E"]
    datatypes = ["float16"]
    worktype = "training"
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [337.5]
    
    job_stats = defaultdict(list)
    for workload in workloads.values():
        for mem in mems:
            mem_info = get_mem_info(mem)
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps, worktype=worktype)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    norm_time = exec_output["Batch total time"]
                    optim_num_gpu = arch_params[0][0]
                    optim_per_gpu_mem_cap_GB = mem_params[0]["mem1_GB"]
                    optim_num_mu_per_gpu = optim_per_gpu_mem_cap_GB // mem_info['cap_GB']
                    optim_per_gpu_mem_bw_GBps = mem_params[0]["mem1_GBps"]
                    # job_stats["Optimized"].append((exec_output["Batch total time"])*(optim_num_mu_per_gpu/5))
                    job_stats["SiPAM"].append((exec_output["Batch total time"])*(optim_num_gpu)/3600)

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(baseline_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    base_num_gpu = arch_params[0][0]
                    base_per_gpu_mem_cap_GB = mem_params[0]["mem1_GB"]
                    base_num_mu_per_gpu = base_per_gpu_mem_cap_GB // mem_info['cap_GB']
                    base_per_gpu_mem_bw_GBps = mem_params[0]["mem1_GBps"]
                    # job_stats["Baseline"].append((exec_output["Batch total time"])*(base_num_mu_per_gpu/5))
                    job_stats["Baseline"].append((exec_output["Batch total time"])*(base_num_gpu)/3600)
                    

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads.keys(), "log":None, "limit": None}
    y_ = {"label": "Total CU Hours", "data": job_stats, "log":10, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(3,2), bbox_to_anchor=(0.01,0.75), ncol=1)
    
def analyzeResourceUsage():
    gpu = "h100"
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
    }
    mems = ["HBM2E"]
    datatypes = ["float16"]
    worktype = "inference"
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048]
    
    job_stats = defaultdict(lambda: defaultdict(list))
    for workload in workloads.values():
        for mem in mems:
            mem_info = get_mem_info(mem)
            for datatype in datatypes:
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps, worktype=worktype)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_num_gpu = arch_params[0][0]
                    optim_per_gpu_mem_cap_GB = mem_params[0]["mem1_GB"]
                    optim_num_mu_per_gpu = optim_per_gpu_mem_cap_GB // mem_info['cap_GB']
                    optim_per_gpu_mem_bw_GBps = mem_params[0]["mem1_GBps"]
                    job_stats["CU"]["Optimized"].append(optim_num_gpu)
                    # job_stats["MU"]["Optimized"].append(optim_num_mu_per_gpu)
                    job_stats["BW"]["Optimized"].append(optim_per_gpu_mem_bw_GBps)

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    base_num_gpu = arch_params[0][0]
                    base_per_gpu_mem_cap_GB = mem_params[0]["mem1_GB"]
                    base_num_mu_per_gpu = base_per_gpu_mem_cap_GB // mem_info['cap_GB']
                    base_per_gpu_mem_bw_GBps = mem_params[0]["mem1_GBps"]
                    job_stats["CU"]["Baseline"].append(base_num_gpu)
                    # job_stats["MU"]["Baseline"].append(base_num_mu_per_gpu)
                    job_stats["BW"]["Baseline"].append(base_per_gpu_mem_bw_GBps)
                    

    pprint.pprint(job_stats)
    workload_stats = {stat:list(workloads.keys()) for stat in ["CU", "MU", "BW"]}
    y_label = {label:f'{label} Count' for label in ["CU", "MU", "BW"]}
    x_ = {"label": "Workloads", "data": workload_stats, "log":None, "limit": None}
    y_ = {"label": y_label, "data": job_stats, "log": 10, "limit": None}
    # plot_util.plotMultiColStackedBarChart(x=x_, y=y_, fig_size=(2.2,2), bbox_to_anchor=(0.4,0.5))
    plot_util.plotMultiColBarSubChart(x=x_, y=y_, fig_dim=(2,1), fig_size=(2.4,2), bbox_to_anchor=(0.4,0.5))

def analyzeArithmeticIntensity():
    gpu = "h100"
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n530M": "megatron-530M",
        "Meg\n1B": "megatron-1B",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
        # "GPT3\n13B": "gpt3-13B",
        # "Meg\n1T": "megatron-1T",
    }
    mems = ["HBM2E"]
    datatypes = ["float16"]
    worktype = "inference"
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [337.5]
    
    job_stats = defaultdict(list)
    for workload in workloads.values():
        workload_info = get_workload_info(workload)
        arithmetic_intensity = workload_info[worktype]["ai"]
        job_stats["Arithmetic Intensity"].append(arithmetic_intensity)
        for mem in mems:
            for datatype in datatypes:
                cu_info = get_cu_info(gpu, datatype)
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps, worktype=worktype)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_operational_intensity = cu_info["matrix"] / (mem_params[0]["mem1_GBps"] * 1e9)
                    job_stats["SiPAM Compute Intensity"].append(optim_operational_intensity)

                    mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, workload, mem, datatype, **args)
                    baseline_operational_intensity = cu_info["matrix"] / (mem_params[0]["mem1_GBps"] * 1e9)
                    job_stats["Baseline Compute Intensity"].append(baseline_operational_intensity)
    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads.keys(), "log":None, "limit": None}
    y_ = {"label": "FLOPs/Byte", "data": job_stats, "log": None, "limit": (0, 1100)}
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(3,2), bbox_to_anchor=(0,0.65), ncol=1)
    # plot_util.plotMultiScatterChart(x=x_, y=y_, fig_size=(3.6,3.27), bbox_to_anchor=(0,0.5))
    
if __name__ == '__main__':
    # analyzeIterTime()
    # analyzeEfficency()
    # analyzeResourceUsage()
    analyzeGPUHour()
    # analyzeArithmeticIntensity()
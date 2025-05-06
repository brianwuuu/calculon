"""
Analysis script for memory disaggregation simulations.
"""

import sys, os, getopt
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(scripts_path)
import util
import pprint
import plot_util
from collections import defaultdict
from resource_optimization import optimize_mem_net, baseline_mem_net
from config_generation import get_confile_filenames, generate_input_str
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
    # --- Hardware Parameters ---
    gpu = "h100_80g_nvl8" # "h100_80g_nvl8", "b100_80g", "a100_80g"
    workloads = [
                 "megatron-126M",
                 "megatron-530M",
                 "megatron-1B",
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "anthropic-52B",
                 "chinchilla-64B",
                #  "turing-530B",
                 "gpt3-13B",
                 "gpt3-175B",
                 ]
    mems = ["HBM2E"]
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for workload, mem, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [workloads, mems, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        args = dict(workload=workload, gpu=gpu, mem=mem, seq_len=seq_len,
                total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm,
                per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
        input_str = generate_input_str("sipam", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        job_stats["SiPAM"].append(exec_output["total_time_aggregate"])

        input_str = generate_input_str("baseline", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        job_stats["Baseline"].append(exec_output["total_time_aggregate"])

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "Norm. Iteration Time", "data": job_stats, "log":10, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)

def analyzeMemoryUsage():
    # --- Hardware Parameters ---
    gpu = "h100_80g_nvl8" # "h100_80g_nvl8", "b100_80g", "a100_80g"
    workloads = [
                 "megatron-126M",
                 "megatron-530M",
                 "megatron-1B",
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "anthropic-52B",
                 "chinchilla-64B",
                #  "turing-530B",
                 "gpt3-13B",
                 "gpt3-175B",
                 ]
    mems = ["HBM2E"]
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for workload, mem, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [workloads, mems, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        args = dict(workload=workload, gpu=gpu, mem=mem, seq_len=seq_len,
                total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm,
                per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
        input_str = generate_input_str("sipam", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        mem_cap_sipam = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        job_stats["SiPAM"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/mem_cap_sipam)

        input_str = generate_input_str("baseline", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        mem_cap_baseline = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        job_stats["Baseline"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/mem_cap_baseline)

    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "% Memory Usage", "data": job_stats, "log":None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)


def analyzeGPUHour():
    # --- Hardware Parameters ---
    gpu = "h100_80g_nvl8" # "h100_80g_nvl8", "b100_80g", "a100_80g"
    workloads = [
                 "megatron-126M",
                 "megatron-530M",
                 "megatron-1B",
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "megatron-1T",
                 "anthropic-52B",
                 "chinchilla-64B",
                #  "turing-530B",
                 "gpt3-13B",
                 "gpt3-175B",
                 ]
    mems = ["HBM2E"]
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for workload, mem, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len in util.cartesian_product(
        [workloads, mems, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens]):
        for max_batch_size in max_batch_sizes:
            args = dict(workload=workload, gpu=gpu, mem=mem, seq_len=seq_len,
                    total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm,
                    per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                    datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
            input_str = generate_input_str("sipam", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str] 
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            job_stats["SiPAM"].append(exec_output["total_time_aggregate"]*exec_output["num_procs"]/3600)

            input_str = generate_input_str("baseline", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            job_stats["Baseline"].append(exec_output["total_time_aggregate"]*exec_output["num_procs"]/3600)
            
    pprint.pprint(job_stats)
    x_ = {"label": "Workloads", "data": workloads, "log":None, "limit": None}
    y_ = {"label": "Total CU Hours", "data": job_stats, "log":10, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(3,2), bbox_to_anchor=(0.01,0.75), ncol=1)

def analyzeArithmeticIntensity():
    gpu = "h100"
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n530M": "megatron-530M",
        "Meg\n1B": "megatron-1B",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "ANth\n52B": "anthropic-52B",
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
    
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"he:",["exp_id="])
    except getopt.GetoptError:
        print('[HELP] python3 analyzeWorkload.py -e <experiment_name>')
        sys.exit(2)
    exp_name = ""
    for opt, arg in opts:
        if opt == '-h':
            print('python3 generate_experiment.py -e <experiment_name>')
            sys.exit()
        elif opt in ("-e"):
            exp_name = str(arg)
    if exp_name == "iter":
        analyzeIterTime()
    elif exp_name == "mem":
        analyzeMemoryUsage()
    elif exp_name == "hour":
        analyzeGPUHour()
    elif exp_name == "ai":
        analyzeArithmeticIntensity()
    else:
        raise Exception("[Error] Invalid Experiment String")
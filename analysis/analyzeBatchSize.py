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

sys_map = {
    "a100_80g": "A100",
    "h100_80g_nvl8": "H100",
    "b100_192g": "B100"
}

def analyzeIterTime():
    # --- Hardware Parameters ---
    compute_sys = [
                   ("a100_80g", "HBM2", 96), 
                #    ("h100_80g_nvl8", "HBM2E", 96), 
                #    ("b100_192g", "HBM3", 120),
                #    ("b100_192g", "HBM2", 120),
                #    ("b100_192g", "HBM4", 120),
                #    ("b100_192g", "HBM2E", 120),
                   ]
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                #  "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "gpt3-175B",
                #  "gpt3-13B",
                 "megatron-1T",
                 ]
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2**i for i in range(int(8).bit_length(), int(8192).bit_length())] # [2048], [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        for sys in compute_sys:
            args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                    total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                    per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                    datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
            input_str = generate_input_str("sipam", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            sipam_time = exec_output["total_time_aggregate"]
            # job_stats["SiPAM"].append(sipam_time)
            job_stats[f"{sys_map[sys[0]]}-SiPAM"].append(sipam_time)

            input_str = generate_input_str("baseline", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            baseline_time = exec_output["total_time_aggregate"]
            # job_stats["Baseline"].append(baseline_time)
            job_stats[f"{sys_map[sys[0]]}-Baseline"].append(baseline_time)

            # job_stats[f"{sys_map[sys[0]]}-Speedup"].append(baseline_time / sipam_time)

    pprint.pprint(job_stats)
    x_ = {"label": "Batch Size", "data": max_batch_sizes, "log":2, "limit": None}
    y_ = {"label": "Norm. Iteration Time", "data": job_stats, "log":None, "limit": None}
    # plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,1.5), bbox_to_anchor=(0.25,1), ncol=3)

def analyzeEfficiency():
    # --- Hardware Parameters ---
    compute_sys = [
                #    ("a100_80g", "HBM2", 96), 
                #    ("h100_80g_nvl8", "HBM2E", 96), 
                   ("b100_192g", "HBM3", 120),
                #    ("b100_192g", "HBM2", 120),
                #    ("b100_192g", "HBM4", 120),
                #    ("b100_192g", "HBM2E", 120),
                   ]
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                #  "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "gpt3-175B",
                #  "gpt3-13B",
                 "megatron-1T",
                 ]
    
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2**i for i in range(int(8).bit_length(), int(8192).bit_length())] # [2048], [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        for sys in compute_sys:
            args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                    total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                    per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                    datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
            input_str = generate_input_str("sipam", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            sipam_eff = exec_output["system_efficiency_aggregate"]
            # job_stats["SiPAM"].append(sipam_eff)
            job_stats[f"{sys_map[sys[0]]}-SiPAM"].append(sipam_eff * 100)

            input_str = generate_input_str("baseline", **args)
            output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
            assert(os.path.isfile(output_file)), output_file
            exec_output = util.parse_JSON(output_file)
            baseline_eff = exec_output["system_efficiency_aggregate"]
            # job_stats["Baseline"].append(baseline_eff)
            job_stats[f"{sys_map[sys[0]]}-Baseline"].append(baseline_eff * 100)

            # job_stats[f"{sys_map[sys[0]]}-Speedup"].append(baseline_time / sipam_time)

    pprint.pprint(job_stats)
    x_ = {"label": "Batch Size", "data": max_batch_sizes, "log":2, "limit": None}
    y_ = {"label": "System Efficiency (%)", "data": job_stats, "log":None, "limit": None}
    # plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,1.5), bbox_to_anchor=(0.25,1), ncol=3)

def analyzeMemoryUsage():
    # --- Hardware Parameters ---
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                 "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "gpt3-175B",
                #  "gpt3-13B",
                #  "megatron-1T",
                 ]
    gpu = "b100_192g" # "h100_80g_nvl8", "b100_192g", "a100_80g"
    mems = ["HBM3"]
    total_length_mm = 120
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2**i for i in range(int(8).bit_length(), int(8192).bit_length())] # [2**i for i in range(int(8).bit_length(), int(2048).bit_length())] # [2048]
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
        job_stats["SiPAM"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_sipam)

        input_str = generate_input_str("baseline", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        mem_cap_baseline = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        job_stats["Baseline"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_baseline)

    pprint.pprint(job_stats)
    x_ = {"label": "Batch Size", "data": max_batch_sizes, "log":None, "limit": None}
    y_ = {"label": "% Memory Usage", "data": job_stats, "log":None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)


def analyzeGPUHour():
    # --- Hardware Parameters ---
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                #  "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "gpt3-175B",
                #  "gpt3-13B",
                 "megatron-1T",
                 ]
    gpu = "a100_80g" # "h100_80g_nvl8", "b100_192g", "a100_80g"
    mems = ["HBM2"]
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2**i for i in range(int(8).bit_length(), int(8192).bit_length())] # [2**i for i in range(int(8).bit_length(), int(2048).bit_length())] # [2048]
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
    x_ = {"label": "Batch Size", "data": max_batch_sizes, "log":2, "limit": None}
    y_ = {"label": "Total GPU Hours", "data": job_stats, "log":None, "limit": (0,200)}
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,1.5), bbox_to_anchor=(0.25,1), ncol=3)

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
    elif exp_name == "eff":
        analyzeEfficiency()
    elif exp_name == "mem":
        analyzeMemoryUsage()
    elif exp_name == "hour":
        analyzeGPUHour()
    else:
        raise Exception("[Error] Invalid Experiment String")
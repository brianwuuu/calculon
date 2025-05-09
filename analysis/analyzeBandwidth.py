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
                   ("h100_80g_nvl8", "HBM2E", 96), 
                   ("b100_192g", "HBM3", 120),
                #    ("b100_192g", "HBM2", 120),
                #    ("b100_192g", "HBM4", 120),
                #    ("b100_192g", "HBM2E", 120),
                   ]
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                 "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "megatron-1T",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "turing-530B",
                #  "gpt3-13B",
                #  "gpt3-175B",
                 ]
    per_pic_length_mm = 8
    per_pic_bws_GBps = [64, 128, 256, 512, 1024, 2048,]
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "inference"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs_map = {"megatron-126M":8, "megatron-1B":16, "megatron-5B":8, "gpt3-175B": 256, "megatron-1T":2048}
    
    job_stats = defaultdict(list)
    for sys, workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [compute_sys, workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        max_num_procs = max_num_procs_map[workload]
        args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
        sys_str = "_".join([str(sys_map[sys[0]]), sys[1]])
        input_str = generate_input_str("sipam", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        # job_stats[sys_str].append(exec_output["total_time"])
        job_stats[sys_str].append(exec_output["total_time_aggregate"])

    pprint.pprint(job_stats)
    bw_density_Gbps_per_mm = [bw*8/per_pic_length_mm/1e3 for bw in per_pic_bws_GBps]
    # x_ = {"label": "Bandwidth Density (Tbps/mm)", "data": bw_density_Gbps_per_mm, "log": 2, "limit": None}
    x_ = {"label": "Per I/O Bandwidth (Gbps)", "data": [x*8 for x in per_pic_bws_GBps], "log": 2, "limit": None}
    y_ = {"label": "Iteration Time (s)", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,1.5), bbox_to_anchor=(0.25,1), ncol=3)


def analyzeGPUHour():
    # --- Hardware Parameters ---
    compute_sys = [("a100_80g", "HBM2", 96), 
                   ("h100_80g_nvl8", "HBM2E", 96), 
                   ("b100_192g", "HBM3", 120),
                   ("b100_192g", "HBM2", 120),
                   ("b100_192g", "HBM4", 120),
                   ("b100_192g", "HBM2E", 120),
                   ]
    workloads = [
                #  "megatron-126M",
                #  "megatron-530M",
                #  "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "megatron-1T",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "turing-530B",
                #  "gpt3-13B",
                 "gpt3-175B",
                 ]
    per_pic_length_mm = 8
    per_pic_bws_GBps = [32, 64, 128, 256, 512, 1024, 2048]
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 4096
    
    job_stats = defaultdict(list)
    for sys, workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [compute_sys, workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
        sys_str = "_".join([str(sys_map[sys[0]]), sys[1]])
        input_str = generate_input_str("sipam", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        job_stats[sys_str].append(exec_output["total_time_aggregate"]*exec_output["num_procs"]/3600)
            
    pprint.pprint(job_stats)
    x_ = {"label": "Per I/O Bandwidth (GBps)", "data": per_pic_bws_GBps, "log": 2, "limit": None}
    y_ = {"label": "Total GPU Hour", "data": job_stats, "log": None, "limit": None}
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0.49,0.75), ncol=1)
    
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
    elif exp_name == "hour":
        analyzeGPUHour()
    else:
        raise Exception("[Error] Invalid Experiment String")
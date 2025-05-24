"""
Analysis script for memory disaggregation simulations.
"""

import sys, os, getopt
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(scripts_path)
import util
import pprint
import pathlib
import plot_util
from collections import defaultdict
from resource_optimization import optimize_mem_net, baseline_mem_net
from config_generation import get_confile_filenames, generate_input_str
from table import get_mem_info, get_workload_info, get_cu_info


####################################################################################################
# Analysis Parameters 
####################################################################################################

print("[Analysis] Start ...")
BASE_DIRECTORY = f"{pathlib.Path(__file__).parent.resolve()}" + "/../"
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

def analyzeIterationTime():
    ### Number of GPUs fixed at 128
    # --- Hardware Parameters ---
    compute_sys = [("a100_80g", "HBM2", 96), 
                   ("a100_80g", "HBM2E", 96), 
                   ("a100_80g", "HBM3", 96),
                   ("a100_80g", "HBM3E", 96),
                #    ("a100_80g", "HBM4", 96),
                   ("h100_80g_nvl8", "HBM2", 96), 
                   ("h100_80g_nvl8", "HBM2E", 96), 
                   ("h100_80g_nvl8", "HBM3", 96),
                   ("h100_80g_nvl8", "HBM3E", 96),
                #    ("h100_80g_nvl8", "HBM4", 96),
                   ("b100_192g", "HBM2", 120), 
                   ("b100_192g", "HBM2E", 120), 
                   ("b100_192g", "HBM3", 120),
                   ("b100_192g", "HBM3E", 120),
                #    ("b100_192g", "HBM4", 120)
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
    per_pic_bws_GBps = [2048] # 2048, 337.5 = 4050 / 12 (4050 = total H100 bandwidth), 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "inference"
    max_batch_sizes = [2048] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_procs = 128
    
    job_stats = defaultdict(list)
    for sys, workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [compute_sys, workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
        input_str = generate_input_str("sipam", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        mem_GB = float(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        mem_cap_sipam = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        norm_time = exec_output["total_time_aggregate"]
        # job_stats[f'{sys[1]}-S'].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_sipam)
        # job_stats["SiPAM"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_sipam)
        job_stats["SiPAM"].append(exec_output["total_time_aggregate"]/norm_time)
        # job_stats["SiPAM"].append(exec_output["total_time_aggregate"]*exec_output["num_procs"]/3600)

        input_str = generate_input_str("baseline", **args)
        output_file = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")[input_str]
        assert(os.path.isfile(output_file)), output_file
        exec_output = util.parse_JSON(output_file)
        mem_GBps = float(output_file.split("/")[-1].split("_")[3].split("GBps")[0])
        mem_GB = float(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        mem_cap_baseline = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
        # job_stats[f'{sys[1]}-B'].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_baseline)
        # job_stats["Baseline"].append((exec_output["proc_mem_tier1_cap_req"] + exec_output["proc_mem_tier2_cap_req"])/(1024**3)/mem_cap_baseline)
        job_stats["Baseline"].append(exec_output["total_time_aggregate"]/norm_time)
        # job_stats["Baseline"].append(exec_output["total_time_aggregate"]*exec_output["num_procs"]/3600)

    pprint.pprint(job_stats)
    # compute_sys_str = [" ".join([str(sys_map[sys[0]]), sys[1]]) for sys in compute_sys]
    compute_sys_str = [sys[1] for sys in compute_sys]
    # compute_sys_str = ["A100", "H100", "B100"]
    x_ = {"label": "Compute System", "data": compute_sys_str, "log":None, "limit": None}
    y_ = {"label": "Iteration Time (s)", "data": job_stats, "log":None, "limit": None}
    plot_util.plotMultiColBarChart(x=x_, y=y_, fig_size=(3.5,1.2), bbox_to_anchor=(0.7,0.60), ncol=1)
    # plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2.5,1.5), bbox_to_anchor=(0.49,0.75), ncol=1)
    
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
        analyzeIterationTime()
    else:
        raise Exception("[Error] Invalid Experiment String")
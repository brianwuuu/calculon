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
from config_generation import generate_input_str

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
                   ("b100_192g", "HBM3", 120),
                   ("h100_80g_nvl8", "HBM2E", 96), 
                   ("a100_80g", "HBM2", 96), 
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
                #  "megatron-1T",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "turing-530B",
                #  "gpt3-13B",
                #  "gpt3-175B",
                 "megatron-1T",
                 ]
    per_pic_length_mm = 8
    per_pic_bws_GBps = [2048]
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # --- Workload Parameters ---
    datatypes = ["float16"]
    worktype = "training"
    max_batch_sizes = [4096] # [2**i for i in range(int(2048).bit_length())]
    seq_lens = [2048]
    max_num_processors = [128, 256, 512, 1024] # 16, 32, 64, 128, 256, 512, 1024
    
    exp_type = "sipam" # "baseline", "sipam"
    exp_map = {"sipam": "SiPAM", "baseline": "Baseline"}
    cache_dict = util.parse_JSON(OUTPUT_DIRECTORY + "cache.json")
    data_text = [[None] * len(max_num_processors) for _ in range(len(compute_sys))]
    for workload, datatype, per_pic_bw_GBps, mem_add_lat_ns, net_lat_ns, seq_len, max_batch_size in util.cartesian_product(
        [workloads, datatypes, per_pic_bws_GBps, mem_add_lats_ns, net_lats_ns, seq_lens, max_batch_sizes]):
        for i, sys in enumerate(compute_sys):
            for j, max_num_procs in enumerate(max_num_processors):
                args = dict(workload=workload, gpu=sys[0], mem=sys[1], seq_len=seq_len,
                        total_length_mm=sys[2], per_pic_length_mm=per_pic_length_mm,
                        per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                        datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, max_num_procs=max_num_procs)
                input_str = generate_input_str(exp_type, **args)
                if input_str not in cache_dict:
                    data_text[i][j] = None
                else:
                    output_file = cache_dict[input_str]
                    assert(os.path.isfile(output_file)), output_file
                    exec_output = util.parse_JSON(output_file)
                    iter_time = int(exec_output['total_time_aggregate'])
                    mem_usage = int((exec_output['proc_mem_tier1_cap_req'])/(1024**3))
                    num_procs = int(exec_output['num_procs'])
                    mem_cap = int(output_file.split("/")[-1].split("_")[5].split("GB")[0])
                    data_text[i][j] = f"{num_procs}GPUs\n{mem_usage}GB\n/{mem_cap}GB\n{iter_time}"
                    print(sys, max_num_procs, data_text[i][j])

    pprint.pprint(data_text)
    compute_sys_str = ["".join([str(sys_map[sys[0]]), "\n", sys[1]]) for sys in compute_sys]
    x_ = {"label": "# Available GPUs", "data": max_num_processors, "log": None, "limit": None}
    y_ = {"label": "Compute Gen", "data": compute_sys_str, "log": None, "limit": None}
    plot_util.plotLabeledHeatMap(x=x_, y=y_, fig_size=(2.3,2.3), data_text=data_text, workload=util.upper(workload), exp_type=exp_map[exp_type])

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
    else:
        raise Exception("[Error] Invalid Experiment String")
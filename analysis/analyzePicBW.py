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
BASE_DIRECTORY = "/path/to/calculon/"
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

def analyzePicBW():
    gpu = "b100"
    workloads = {
        "126M": "megatron-126M",
        "530M": "megatron-530M",
        "1B": "megatron-1B",
        "5B": "megatron-5B", 
        "22B": "megatron-22B", 
        "40B": "megatron-40B",
        "52B": "anthropic-52B",
        "64B": "chinchilla-64B",
        "GPT3-175B": "gpt3-175B",
        # "GPT3\n13B": "gpt3-13B",
        "Meg\n1T": "megatron-1T",
    }
    mems = ["HBM3"]
    datatypes = ["float16"]
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [25, 50, 100, 200, 400, 800, 1600] # 2048, 337.5
    mem_add_lat_ns = 60
    net_lat_ns = 20
    worktype = "training"
    
    job_stats = defaultdict(list)
    for workload in workloads.values():
        for mem in mems:
            for datatype in datatypes:
                norm_time = None
                for per_pic_bw_GBps in per_pic_bws_GBps:
                    args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, per_pic_bw_GBps=per_pic_bw_GBps,
                                worktype=worktype, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns)
                    mem_params, net_params, model_params, arch_params = optimize_mem_net(gpu, workload, mem, datatype, **args)
                    optim_files = get_confile_filenames(gpu, mem_params, net_params, model_params, arch_params, **args)
                    model,arch_filename,system_filename = get_config_str(optim_files[0])
                    output_dir = OUTPUT_DIRECTORY + model + "/" + arch_filename + "/"
                    assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
                    exec_output = util.parseJSON(output_dir + system_filename)
                    # job_stats["SiPAM"].append(exec_output["Batch total time"]/norm_time)
                    if not norm_time: norm_time = exec_output["Batch total time"]
                    job_stats[workload].append(exec_output["Batch total time"]/norm_time)

    pprint.pprint(job_stats)
    x_ = {"label": "Per PIC Bandwidth (GBps)", "data": [str(x) for x in per_pic_bws_GBps], "log":None, "limit": None}
    y_ = {"label": "Norm. Iteration Time", "data": job_stats, "log":None, "limit": (0.75, 1.02)}
    plot_util.plotMultiLineChart(x=x_, y=y_, fig_size=(2,2), bbox_to_anchor=(0,1), ncol=5)

if __name__ == '__main__':
    analyzePicBW()
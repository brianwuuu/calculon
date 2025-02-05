"""
File generating script for memory disaggregation experiments.
"""
import sys, getopt, pprint
import utilities
from config_generation import setup_experiment, setup_optim_experiment, generate_optim_configs
from resource_optimization import optimize_mem_net, baseline_mem_net

####################################################################################################
# Simulation Parameters 
####################################################################################################

def generate_sipam_experiment():
    model_params = [{'model': "GPT3-175B", 'hidden': 4096, 'attn_size':128, 'num_blocks':24}] # GPT3-175B: (12288,128,96)
    arch_params = [
                   (1,1,1,1)
                #    (8,2,2,2), (16,2,4,2), (32,2,8,2), 
                #    (64,2,16,2), (128,2,32,2), (256,2,32,4), 
                #    (512,4,32,4),(1024,4,32,8), (2048,8,32,8), (4096,8,32,16)
                   ]
    total_length_mm = 96
    per_hbm_length_mm = 12
    per_pic_length_mm = 8
    per_pic_bw_GBps = 2048 # 2033, 2048
    
    num_local_hbms = [0] # 0,1,2,4,6
    per_hbm_bw_GBps = 500
    per_hbm_capacity_GB = 16
    worktype = "training"
    
    mem_params, net_params = [], []
    for num_local_hbm in num_local_hbms:
        local_hbm_length_mm = num_local_hbm * per_hbm_length_mm
        local_hbm_capacity_GB = num_local_hbm * per_hbm_capacity_GB
        local_hbm_bw_GBps = num_local_hbm * per_hbm_bw_GBps
        num_pic = (total_length_mm - local_hbm_length_mm) // per_pic_length_mm
        max_sip_bw_GBps = num_pic * per_pic_bw_GBps # 24.4 TBps
        
        # ratio
        mem_to_net_split = [(x, num_pic - x) for x in range(1, num_pic)]
        for mem_ratio, net_ratio in mem_to_net_split:
            mem_params.append((local_hbm_capacity_GB, local_hbm_bw_GBps, 1000, mem_ratio*per_pic_bw_GBps))
            net_params.append((net_ratio*per_pic_bw_GBps, 1e-5, 0.15, net_ratio*per_pic_bw_GBps, 1e-5, 0.15))
            assert((net_ratio + mem_ratio) * per_pic_bw_GBps == max_sip_bw_GBps)
    setup_experiment(mem_params, net_params, model_params, arch_params, gpu="h100", worktype=worktype)

def generate_simple_experiment():
    model_params = [{'model': "gpt3-13B"}] # GPT3-175B: (12288,128,96)
    arch_params = [(1,1,1,1)]
    mem_params, net_params = [{}], [{}]
    worktype = "inference"
    setup_experiment(mem_params, net_params, model_params, arch_params, gpu="h100", exp_name="simple", worktype=worktype)
        
def generate_mem_net_experiment():
    gpu = "b100"
    workloads = [
                 "megatron-126M",
                 "megatron-530M",
                 "megatron-1B",
                 "megatron-5B", 
                 "megatron-22B", 
                 "megatron-40B",
                 "anthropic-52B",
                 "chinchilla-64B",
                 "gpt3-175B",
                 "gpt3-13B",
                 "megatron-1T",
                 ]
    mems = ["HBM3"]
    datatypes = ["float16"]
    worktype = "training"
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [25, 50, 100, 200, 400, 800, 1600] # 2048, 337.5 = 4050 / 12, 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # optimized experiments
    mem_params, net_params, model_params, arch_params = [], [], [], []
    for per_pic_bw_GBps, workload, mem, datatype, mem_add_lat_ns, net_lat_ns in utilities.cartesian_product(
        [per_pic_bws_GBps, workloads, mems, datatypes, mem_add_lats_ns, net_lats_ns]):
        args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, 
                    per_pic_bw_GBps=per_pic_bw_GBps, worktype=worktype, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns)
        mem_param, net_param, model_param, arch_param = optimize_mem_net(gpu, workload, mem, datatype, **args)
        mem_params.extend(mem_param)
        net_params.extend(net_param)
        model_params.extend(model_param)
        arch_params.extend(arch_param)
    setup_experiment(mem_params, net_params, model_params, arch_params, gpu=gpu, worktype=worktype)

    # baseline experiments
    mem_params, net_params, model_params, arch_params = [], [], [], []
    for workload, mem, datatype in utilities.cartesian_product([workloads, mems, datatypes]):
        args = dict(worktype=worktype)
        mem_param, net_param, model_param, arch_param = baseline_mem_net(gpu, workload, mem, datatype, **args)
        mem_params.extend(mem_param)
        net_params.extend(net_param)
        model_params.extend(model_param)
        arch_params.extend(arch_param)
    setup_experiment(mem_params, net_params, model_params, arch_params, gpu=gpu, exp_name="baseline", worktype=worktype) 


def generate_optim_experiment():
    gpu = "h100"
    workloads = [
                 "megatron-126M",
                #  "megatron-530M",
                #  "megatron-1B",
                #  "megatron-5B", 
                #  "megatron-22B", 
                #  "megatron-40B",
                #  "anthropic-52B",
                #  "chinchilla-64B",
                #  "gpt3-175B",
                #  "gpt3-13B",
                #  "megatron-1T",
                 ]
    mems = ["HBM2"]
    datatypes = ["float16"]
    worktype = "training"
    max_batch_size = 2048
    num_iter = 10
        
    total_length_mm = 96
    per_pic_length_mm = 8
    per_pic_bws_GBps = [1600] # 2048, 337.5 = 4050 / 12, 25, 50, 100, 200, 400, 800, 1600
    mem_add_lats_ns = [60] # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9
    net_lats_ns = [20]
    
    # optimized experiments
    optim_config_files = []
    for per_pic_bw_GBps, workload, mem, datatype, mem_add_lat_ns, net_lat_ns in utilities.cartesian_product(
        [per_pic_bws_GBps, workloads, mems, datatypes, mem_add_lats_ns, net_lats_ns]):
        args = dict(total_length_mm=total_length_mm, per_pic_length_mm=per_pic_length_mm, 
                    per_pic_bw_GBps=per_pic_bw_GBps, mem_add_lat_ns=mem_add_lat_ns, net_lat_ns=net_lat_ns,
                    datatype=datatype, worktype=worktype, max_batch_size=max_batch_size, num_iter=num_iter)
        optim_config = generate_optim_configs(gpu, workload, mem, **args)
        optim_config_files.append(optim_config)
    setup_optim_experiment(optim_config_files, exp_name="optim")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"he:",["exp_id="])
    except getopt.GetoptError:
        print('[HELP] python3 generate_experiment.py -e <experiment_name>')
        sys.exit(2)
    exp_name = ""
    for opt, arg in opts:
        if opt == '-h':
            print('python3 generate_experiment.py -e <experiment_name>')
            sys.exit()
        elif opt in ("-e"):
            exp_name = str(arg)
    if exp_name == "simple":
        generate_simple_experiment()
    elif exp_name == "sipam":
        generate_sipam_experiment()
    elif exp_name == "mem_net":
        generate_mem_net_experiment()
    elif exp_name == "optim":
        generate_optim_experiment()
    else:
        raise Exception("[Error] Invalid Experiment String")
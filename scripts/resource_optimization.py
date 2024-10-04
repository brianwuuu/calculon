import numpy as np
import utilities
from table import get_workload_info, get_cu_info, get_mem_info

def optimize_mem_net(gpu, workload, mem, datatype, **kwargs):
    # variables
    total_length_mm = kwargs["total_length_mm"]
    per_pic_length_mm = kwargs["per_pic_length_mm"]
    per_pic_bw_GBps = kwargs["per_pic_bw_GBps"]
    
    workload_info = get_workload_info(workload)
    cu_info = get_cu_info(gpu, datatype)
    mem_info = get_mem_info(mem)
    
    req_mem_bw_per_gpu_GBps = cu_info["matrix"] / workload_info["ai"] / 1e9
    num_req_mu_per_gpu = int(np.ceil(req_mem_bw_per_gpu_GBps / mem_info["bw_GBps"]))
    per_gpu_mem_cap_GB = num_req_mu_per_gpu * mem_info["cap_GB"]
    per_gpu_mem_bw_GBps = num_req_mu_per_gpu * mem_info["bw_GBps"]
    num_gpu = int(np.ceil(workload_info["size_GB"] / per_gpu_mem_cap_GB))
    num_gpu = utilities.nearest_pow_of_2(num_gpu)
    par_params = utilities.get_par_params(num_gpu)
    
    num_mem_pic_per_gpu = int(np.ceil(per_gpu_mem_bw_GBps / per_pic_bw_GBps))
    num_net_pic_per_gpu = (total_length_mm - (per_pic_length_mm * num_mem_pic_per_gpu)) // per_pic_length_mm # round down
    net_bw_GBps = num_net_pic_per_gpu * per_pic_bw_GBps
    
    params = dict(
              num_gpu=num_gpu,
              arithmetic_intensity=workload_info["ai"],
              workload_size_GB=workload_info["size_GB"],
              req_mem_bw_per_gpu_GBps=req_mem_bw_per_gpu_GBps, 
              num_req_mu_per_gpu=num_req_mu_per_gpu, 
              per_gpu_mem_cap_GB=per_gpu_mem_cap_GB,
              per_gpu_mem_bw_GBps=per_gpu_mem_bw_GBps,
              num_mem_pic_per_gpu=num_mem_pic_per_gpu,
              num_net_pic_per_gpu=num_net_pic_per_gpu,
              net_bw_GBps=net_bw_GBps,
              par_params=par_params
              )

    mem_params = [{"mem1_GB": per_gpu_mem_cap_GB, 
                   "mem1_GBps": per_gpu_mem_bw_GBps, 
                   "mem1_ns": mem_info["lat_ns"], 
                   "mem2_GB": per_gpu_mem_cap_GB, 
                   "mem2_GBps": per_gpu_mem_bw_GBps, 
                   "mem2_ns": mem_info["lat_ns"]}]
    net_params = [{"net1_GBps": net_bw_GBps, 
                   "net1_ns": 0, 
                   "net2_GBps": net_bw_GBps, 
                   "net2_ns": 0
                   }]
    model_params = [{'model': workload}] # [(hidden, attn_size, num_blocks)]
    arch_params = [(num_gpu, par_params[0], par_params[1], par_params[2])] # [(workload, num_procs, tensor_par, pipe_par, data_par)]
    return mem_params, net_params, model_params, arch_params

def baseline_mem_net(gpu, workload, mem, datatype, **kwargs):
    workload_info = get_workload_info(workload)
    mem_info = get_mem_info(mem)
    num_hbm_per_gpu = 5 # 5/6 working HBMs
    mem_cap_GB = num_hbm_per_gpu * mem_info['cap_GB']
    mem_bw_GBps = num_hbm_per_gpu * mem_info['bw_GBps']
    num_gpu = int(np.ceil(workload_info["size_GB"] / mem_cap_GB))
    num_gpu = utilities.nearest_pow_of_2(num_gpu)
    par_params = utilities.get_par_params(num_gpu)

    mem_params = [{"mem1_GB": mem_cap_GB, 
                   "mem1_GBps": mem_bw_GBps, 
                   "mem1_ns": mem_info["lat_ns"], 
                   "mem2_GB": mem_cap_GB, 
                   "mem2_GBps": mem_bw_GBps, 
                   "mem2_ns": mem_info["lat_ns"]}]
    net_params = [{}]
    model_params = [{'model':workload}]
    arch_params = [(num_gpu, par_params[0], par_params[1], par_params[2])]
    return mem_params, net_params, model_params, arch_params
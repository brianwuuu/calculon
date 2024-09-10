"""
Analysis script for memory disaggregation simulations.
"""

import sys, os
import pprint, copy
import plot_util
import util
import collections

####################################################################################################
# Analysis Parameters 
####################################################################################################

# Directory Setup
print("[Analysis] Start ...")
BASE_DIRECTORY = "/Users/bwu/src/calculon/"
OUTPUT_DIRECTORY = BASE_DIRECTORY + "temp/"
SYSTEM_DIRECTORY = BASE_DIRECTORY + "systems/"
MODEL_DIRECTORY = BASE_DIRECTORY + "models/"
ARCH_DIRECTORY = BASE_DIRECTORY + "examples/"
EXECUTION_DIRECTORY = BASE_DIRECTORY + "execution/"

# Base file setup
gpu = "h100_80g_nvl8" # "h100_inf_nvl8" # "h100_80g_nvl8"
model = "gpt3-175B"
arch = "4096_t8_p64_d8_mbs4_full" # "3072_t4_p64_d12_mbs4_full"
system_base_filename = SYSTEM_DIRECTORY + gpu + ".json"
model_base_filename = MODEL_DIRECTORY + model + ".json"
arch_base_filename = ARCH_DIRECTORY + arch + ".json"
system_base = util.parseJSON(system_base_filename)
model_base = util.parseJSON(model_base_filename)
arch_base = util.parseJSON(arch_base_filename)

def analyzeModelSize():
    # Models
    model_params = [
              (24576,192,128,"1T"), (32768,205,160,"2T"), (40960,213,192,"4T"), (50176,224,224,"7T"),
              (60416,236,256,"11T"), (70656,245,288,"18T"), (81920,256,320,"26T"), (94208,268,352,"37T"),
              (106496,277,384,"53T"), (119808,288,416,"72T"), (134144,299,448,"96T"), (148480,309,480,"128T")
              ] # hidden, attn size, # blocks
    # Architectures
    arch_params = [(4096,8,32,16)]
    arch = "4096_t8_p32_d16_mbs4_wo_ao_oo_full"
    # arch = "16384_t8_p32_d64_mbs4_wo_ao_oo_full"
    
    sys_params = [
                #   (256,50), (512,50), (1024,50), (2048,50),
                #   (256,100), (512,100), (1024,100), (2048,100)
                (500,3072,10000,128,"PCIe"), (500,3072,10000,900,"NVLink"), (500,3072,10000,6100,"SiP")
                ] # mem1 GB, GBps; mem2 GB, GBps; 
   
    job_stats = collections.defaultdict(list)
    model_sizes = []
    for model_param in model_params:
        new_model = copy.deepcopy(model_base)
        new_model["seq_size"] = 8
        new_model["hidden"] = model_param[0]
        new_model["attn_size"] = model_param[1]
        new_model["num_blocks"] = model_param[2]
        new_model["feedforward"] = 4 * model_param[0]
        new_model["attn_heads"] = model_param[2]
        model_sizes.append(model_param[3]) 
        model_filename = util.generateModelFileNameString(new_model)
        output_dir = OUTPUT_DIRECTORY + model_filename + "/" + arch + "/" + gpu + "/"
        for sys_param in sys_params:
            new_system = copy.deepcopy(system_base)
            new_system["mem1"]["GiB"] = sys_param[0]
            new_system["mem1"]["GBps"] = sys_param[1]
            new_system["mem2"]["GiB"] = sys_param[2]
            new_system["mem2"]["GBps"] = sys_param[3]
            new_system["networks"][0]["size"] = 8
            new_system["processing_mode"] = "no_overlap" # roofline, no-overlap
            system_filename = util.generateSystemFileNameString(new_system)
            # assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
            if os.path.isfile(output_dir + system_filename):
                exec_output = util.parseJSON(output_dir + system_filename)
                job_stats[sys_param[4]].append(exec_output["Batch total time"])
            else:
                print("[Error] File not found: {}".format(output_dir + system_filename))
                job_stats[sys_param[4]].append(0)
    pprint.pprint(job_stats)
    x_ = {"label": "Model Sizes", "data": model_sizes, "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": None, "limit": None}
    # plot_util.plotMultiColBarChart(x = x_, y = y_, path = "")
    fig_path = "/Users/bwu/Downloads/model_sizes.png"
    plot_util.plotMultiLineChart(x = x_, y = y_, path = "")
    
def analyzeNetworkSize():
    # Models
    model_params = [
            #   (24576,192,128,"1T"), (32768,205,160,"2T"), (40960,213,192,"4T"), (50176,224,224,"7T"),
            #   (60416,236,256,"11T"), (70656,245,288,"18T"), (81920,256,320,"26T"), (94208,268,352,"37T"),
            #   (106496,277,384,"53T"), (119808,288,416,"72T"), (134144,299,448,"96T"), (148480,309,480,"128T")
            (24576,192,128)
              ] # hidden, attn size, # blocks, name
    model_filename = "148480h_593920ff_8ss_480ah_309as_480nb" # "24576h_98304ff_8ss_128ah_192as_128nb"
    # Architectures
    arch_params = [
                # (4096,8,32,16)
                # (8,2,2,2), (16,2,4,2), (32,2,8,2), (64,2,16,2), (128,2,32,2), (256,2,32,4), (512,4,32,4),
                # (2048,8,32,8), (4096,8,32,16), (8192,8,32,32), (16384,8,32,64), (32768,8,32,128)
                (1024,4,64,4), (2048,4,64,8), (4096,8,64,8), (8192,16,64,8), (16384,16,64,16), (32768,16,64,32)
        ]
    # arch = "4096_t8_p32_d16_mbs4_wo_ao_oo_full" # "4096_t8_p32_d16_mbs4_full" # 4096_t8_p32_d16_mbs4_wo_ao_oo_full
    # arch = "16384_t8_p32_d64_mbs4_wo_ao_oo_full"
    
    # Systems
    sys_params = [
                    # (256,50), (512,50), (1024,50), (2048,50),
                    # (256,100), (512,100), (1024,100), (2048,100)
                    # (500,3072,10000,128,"PCIe"), (500,3072,10000,900,"NVLink"), (500,3072,10000,6100,"SiP")
                    (1000,3072,10000000,128, "PCIe"), (1000,3072,10000000,900, "NVLink"), (1000,3072,10000000,6100, "SiP")
                ] # GB, GBps
   
    job_stats = collections.defaultdict(list)
    model_sizes = []
    
    # for model_param in model_params:
    #     new_model = copy.deepcopy(model_base)
    #     new_model["seq_size"] = 8
    #     new_model["hidden"] = model_param[0]
    #     new_model["attn_size"] = model_param[1]
    #     new_model["num_blocks"] = model_param[2]
    #     new_model["feedforward"] = 4 * model_param[0]
    #     new_model["attn_heads"] = model_param[2]
    #     model_sizes.append(model_param[3]) 
    #     model_filename = util.generateModelFileNameString(new_model)
    network_sizes = []
    for num_proc, tensor_par, pipe_par, data_par in arch_params:
        new_arch = copy.deepcopy(arch_base)
        new_arch["num_procs"] = num_proc
        new_arch["tensor_par"] = tensor_par
        new_arch["pipeline_par"] = pipe_par
        new_arch["data_par"] = data_par
        new_arch["tensor_par_net"] = 0
        new_arch["pipeline_par_net"] = 1
        new_arch["data_par_net"] = 1
        # offloading
        new_arch["weight_offload"] = True
        new_arch["activations_offload"] = True
        new_arch["optimizer_offload"] = True
        arch_filename = util.generateArchFileNameString(new_arch)
        network_sizes.append(num_proc)
        output_dir = OUTPUT_DIRECTORY + model_filename + "/" + arch_filename + "/" + gpu + "/"
        for sys_param in sys_params:
            new_system = copy.deepcopy(system_base)
            new_system["mem1"]["GiB"] = sys_param[0]
            new_system["mem1"]["GBps"] = sys_param[1]
            new_system["mem2"]["GiB"] = sys_param[2]
            new_system["mem2"]["GBps"] = sys_param[3]
            new_system["networks"][0]["size"] = 8
            new_system["processing_mode"] = "no_overlap" # roofline, no-overlap
            system_filename = util.generateSystemFileNameString(new_system)
            assert(os.path.isfile(output_dir + system_filename)), output_dir + system_filename
            if os.path.isfile(output_dir + system_filename):
                exec_output = util.parseJSON(output_dir + system_filename)
                job_stats[sys_param[4]].append(exec_output["Batch total time"])
            else:
                job_stats[sys_param[4]].append(0)
    pprint.pprint(job_stats)
    # x_ = {"label": "Model Sizes", "data": model_sizes, "log": None, "limit": None}
    x_ = {"label": "Network Sizes", "data": network_sizes, "log": None, "limit": None}
    y_ = {"label": "Execution Time (s)", "data": job_stats, "log": 2, "limit": None}
    fig_path = "/Users/bwu/Downloads/exe_time.png"
    plot_util.plotMultiLineChart(x = x_, y = y_, path = fig_path)
    
def main():
    analyzeModelSize()
    # analyzeNetworkSize()
        
if __name__ == '__main__':
    main()
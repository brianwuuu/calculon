import sys, os
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(scripts_path)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from table import get_mem_info, get_workload_info, get_cu_info
from resource_optimization import optimize_mem_net, baseline_mem_net

mpl.rcParams['font.family'] = "serif"
mpl.rcParams['hatch.linewidth'] = 0.2
mpl.use('tkagg')

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # ['darkcyan', 'lime', 'darkred','deeppink', 'blueviolet',  "silver", 'black']
mcolor_cycle = list(mcolors.TABLEAU_COLORS.values())
bar_color_cycle = ["lightblue", "cadetblue", "darkseagreen", "darkcyan", "olive", "slategray", "midnightblue", "darkslateblue"]
line_color_cycle = ["steelblue", "firebrick", "yellowgreen", "mediumpurple", "darkseagreen", "darkcyan",]
mark_cycle = ['s', 'x', "o", '+', 'v', '1', 'd', 'p', ".", "^", "<", ">", "1", "2", "3", "8", "P"]
line_styles = ["solid", "dashed", "dashdot", "dotted",]
bar_hatches = [ "////////" ,"\\\\\\\\\\\\\\\\", "xxxxxxxx", "........", "||||" , "o", "O", ".", "*", "/" , "\\" , "|" , "+" , "++++" ]
markersize_arg = 6
legend_fontsize = 6
tick_fontsize = 6
label_fontsize = 6
plot_linewidth = 1
plotfont = {'fontname':'Times'}

def analyzeRoofline():
    workloads = {
        "Meg\n126M": "megatron-126M",
        "Meg\n5B": "megatron-5B", 
        "Meg\n22B": "megatron-22B", 
        "Meg\n40B": "megatron-40B",
        "Anth\n52B": "anthropic-52B",
        "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
        # "Meg\n1T": "megatron-1T",
        # "GPT3\n13B": "gpt3-13B"
    }
    gpu = "h100"
    mems = ["HBM2E"]
    datatype = "float16"
    worktype = "training"
    
    cu_info = get_cu_info(gpu, datatype)
    peak_performance = cu_info['matrix']
    ai_range = np.logspace(1, 5, 1000)
    
    fig, ax = plt.subplots(1, figsize=(2.8,1.65),dpi=200)
    for mem in mems:
        mem_info = get_mem_info(mem)
        mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, "gpt3-175B", mem, datatype, worktype=worktype)
        mem_bw = mem_params[0]['mem1_GBps'] * 1e9
        
        mem_bound = mem_bw * ai_range
        compute_bound = np.ones_like(ai_range) * peak_performance
        
        roofline = np.minimum(mem_bound, compute_bound)
        ax.plot(ai_range, roofline, linewidth=plot_linewidth,
                label=f'{mem}')
    
    for name, workload in workloads.items():
        workload_info = get_workload_info(workload)    
        workload_ai = workload_info[worktype]['ai']
        idx = np.searchsorted(ai_range, workload_ai)
        if idx == len(ai_range): idx -= 1
        max_performance = roofline[idx]
        plt.vlines(x=workload_ai, ymin=0, ymax=max_performance,
                   color='red', linestyle='dashed', linewidth=0.5)
    
    plt.hlines(y=peak_performance, 
               color='black', xmin=333, xmax=1e5,
               linestyle='-', linewidth=plot_linewidth)
    
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=label_fontsize)
    ax.set_ylabel("Performance (FLOPs/s)", fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    ax.grid(which='both', linestyle='--', linewidth=0.1)
    plt.legend(fontsize=legend_fontsize, ncol=1)
    plt.tight_layout()
    plt.show()

def analyzeOptimSteps():
    workloads = {
        # "Meg\n126M": "megatron-126M",
        # "Meg\n5B": "megatron-5B", 
        # "Meg\n22B": "megatron-22B", 
        # "Meg\n40B": "megatron-40B",
        # "Anth\n52B": "anthropic-52B",
        # "Chin\n64B": "chinchilla-64B",
        "GPT3\n175B": "gpt3-175B",
        # "Meg\n1T": "megatron-1T",
        # "GPT3\n13B": "gpt3-13B"
    }
    gpu = "h100"
    mems = ["HBM2E"]
    datatype = "float16"
    worktype = "training"
    
    cu_info = get_cu_info(gpu, datatype)
    peak_performance = cu_info['matrix']
    ai_range = np.logspace(1, 5, 1000)
    
    fig, ax = plt.subplots(1, figsize=(2.8,1.65),dpi=200)
    
    # Plot slanted roof for memory bandwidth
    for mem in mems:
        mem_info = get_mem_info(mem)
        mem_params, net_params, model_params, arch_params = baseline_mem_net(gpu, "gpt3-175B", mem, datatype, worktype=worktype)
        mem_bw = 1160 * 1e9
        
        mem_bound = mem_bw * ai_range
        compute_bound = np.ones_like(ai_range) * peak_performance
        
        roofline = np.minimum(mem_bound, compute_bound)
        ax.plot(ai_range, roofline, linewidth=plot_linewidth,
                label=f'{mem}')
    
    # Plot vertical lines for workloads
    for name, workload in workloads.items():
        workload_info = get_workload_info(workload)    
        workload_ai = workload_info[worktype]['ai']
        idx = np.searchsorted(ai_range, workload_ai)
        if idx == len(ai_range): idx -= 1
        max_performance = roofline[idx]
        plt.vlines(x=workload_ai, ymin=0, ymax=max_performance,
                   color='red', linestyle='dashed', linewidth=0.5)
    
    plt.hlines(y=peak_performance, 
               color='black', xmin=862.31, xmax=1e5,
               linestyle='-', linewidth=plot_linewidth)
    
    ax.set_ylim(1e13, 2e15)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=label_fontsize)
    ax.set_ylabel("Performance (FLOPs/s)", fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    ax.grid(which='both', linestyle='--', linewidth=0.1)
    # plt.legend(fontsize=legend_fontsize, ncol=1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyzeRoofline()
    # analyzeOptimSteps()
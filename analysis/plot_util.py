# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

mpl.rcParams['font.family'] = "serif"
mpl.rcParams['hatch.linewidth'] = 0.5
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

def getCDF(data):
    count, bins_count = np.histogram(data, bins=100) 
    pdf = count / sum(count) 
    cdf = np.cumsum(pdf) 
    return cdf, bins_count[1:]

def getCDFv2(data):
    sorted_data = np.sort(data)
    # cumulative_data = np.cumsum(sorted_data) / np.sum(sorted_data)
    cumulative_data = np.array(range(len(data)))/float(len(data))
    return sorted_data, cumulative_data

def plotCDF(data, name=""):
    cdf, bins_count = getCDF(data)
    # plotting CDF
    plt.plot(bins_count[1:], cdf, label=name) 
    plt.legend()
    plt.show()
    plt.close()

def plotViolin(x, name=""):
    fig, ax = plt.subplots(1, figsize=(4,4),dpi=200)
    ax.violinplot(x["data"].values(), points=60, widths=0.4, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5)
    ax.set_xticks([y + 1 for y in range(len(x['data']))],
                  labels= [key for key in x['data'].keys()])
    ax.set_xlabel("Reconf Delay (s)", fontsize=label_fontsize)
    ax.set_ylabel(x["label"], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    # ax.set_yscale('log', base=10)
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    plt.show()
    plt.close()

def plotMultiCDF(x, path=""):
    print("[ANALYSIS] Plotting CDF for " + x["label"])
    fig, ax = plt.subplots(1, figsize=(4,4),dpi=200)
    for label, data in x['data'].items():
        x_, y_ = getCDFv2(data)
        ax.plot(x_, y_, 
                label=label,
                markerfacecolor='none', 
                markersize=markersize_arg,
                color=next(ax._get_lines.prop_cycler)['color'],
                linewidth=plot_linewidth
                )
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel("CDF", fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    if x["log"]: ax.set_xscale('log',base=x["log"])
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    # plt.legend(bbox_to_anchor= (0, 1.05), loc='lower left', ncol=6, fontsize=12)
    plt.legend(fontsize=legend_fontsize, ncol=1)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()


def plotBarChart(x, y, path=""):
    print("[ANALYSIS] Plotting bar chart for " + y["label"] + " vs " + x["label"])
    fig, ax = plt.subplots(1, figsize=(7,2.3),dpi=200)
    ax.bar(x["data"], y["data"], width=0.5, color="darkcyan")
    # for i, key in enumerate(y["data"]):
        # ax.text(i-0.25, 1.05*v, str(round(v,2)), color='darkcyan', fontweight='bold', fontsize=5)
    # ax.axhline(y = 88.52, color = 'r', linestyle = '--', linewidth=0.5)
    # ax.axhline(y = 600, color = 'r', linestyle = '--', linewidth=0.5)
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
    if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
    if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
    if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
    ax.set_xticks(list(x["data"]), labels=x["data"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") # , rotation=45, ha="right"
    ax.tick_params(axis='x', labelsize=label_fontsize-2)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.3)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.3)
    plt.tight_layout()
    # plt.legend(fontsize=legend_fontsize, ncol=2, loc="upper left")
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

def plotMultiColBarChart(x, y, path="", fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting multi-column bar chart for " + y["label"] + " vs " + x["label"])
    # plt.style.use('ggplot')
    num_pairs = len(x["data"])
    ind = np.arange(num_pairs)
    width = 0.3
    fig, ax = plt.subplots(1, figsize=fig_size, dpi=200)
    for i, parameter in enumerate(y["data"].keys()):
        ax.bar(ind+i*width, y["data"][parameter], label=parameter, width=width)
    ax.set_xticklabels(x["data"], fontsize=label_fontsize, rotation=45, ha="right")
    x_ticks_loc = [coord + width for coord in range(len(x['data']))]
    ax.set_xticks(x_ticks_loc)
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    if 'title' in kwargs: ax.set_title(kwargs['title'], y=1.15, pad=-10, fontsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
    if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
    if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
    if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
    if "sci" in y.keys() and y["sci"]: ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if "sci" in x.keys() and x["sci"]: ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.2)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.2)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.2)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.2)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'], loc='lower left', fontsize=legend_fontsize, ncol=1)
    if path: plt.savefig(path, dpi=200, transparent=False, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plotMultiColStackedBarChart(x, y, log=False, path=""):
    print("[ANALYSIS] Plotting multi-stacked bar chart for " + y["label"] + " vs " + x["label"])
    fig, ax = plt.subplots(1, figsize=(3.5,2.3),dpi=200)
    bottom = np.zeros(len(x["data"]))
    for i, (label, data) in enumerate(y["data"].items()):
        ax.bar(x["data"], data, width=0.4, label=label, bottom=bottom)
        bottom += data
    if y["log"]: ax.set_yscale('log',base=y["log"])
    if x["log"]: ax.set_xscale('log',base=x["log"])
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.3)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.3)
    # ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") # , rotation=45, ha="right"
    ax.grid(axis='y', linestyle='--')
    plt.legend(bbox_to_anchor= (0.35, 0.45),loc="lower left", ncol=2, fontsize=legend_fontsize)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

def plotStackedBarSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting bar chart for " + y["label"] + " vs " + x["label"])
    plt.style.use('ggplot')
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel() if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
    for index, param in enumerate(y["data"].keys()):
        ax = axes[index] if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
        bottom = np.zeros(len(x["data"]))
        for i, (label, data) in enumerate(y["data"][param].items()):
            ax.bar(x['data'], data, label=label, bottom=bottom, 
                   linewidth=0.3,
                   width = 0.3,
                #    color=mcolor_cycle[i],
                   edgecolor="black")
            bottom += data
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15,) # rotation=45,
        # ax.set_xlabel(x["label"], fontsize=label_fontsize)
        ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.set_title(param, fontsize=label_fontsize)
        ax.yaxis.offsetText.set_fontsize(label_fontsize)
        if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
        if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        if "sci" in y.keys() and y["sci"]: ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if "sci" in x.keys() and x["sci"]: ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.1)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.1)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.1)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.1)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'],loc="lower left", ncol=1, fontsize=legend_fontsize)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

def plotMultiColStackedBarSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting multi-stacked bar chart for " + y["label"] + " vs " + x["label"])
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel() if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
    for index, param in enumerate(y["data"].keys()):
        ax = axes[index] if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
        width = 0.3
        ind = np.arange(len(x['data'][param]))
        for i, s_param in enumerate(y['data'][param].keys()):
            bottom = np.zeros(len(x["data"][param]))
            for j, (label, data) in enumerate(y["data"][param][s_param].items()):
                label_str = f"{s_param}-{label}"
                ax.bar(ind+i*width, data, label=label_str, width=width, bottom=bottom, linewidth=0.6,
                       color=mcolor_cycle[i],
                       edgecolor="white", hatch=bar_hatches[j%len(y["data"][param][s_param])])
                bottom += data
        ax.set_xticklabels(x["data"][param], fontsize=label_fontsize,  ha="right")
        x_ticks_loc = [x + width for x in range(6)] # [0.3, 1.3, 2.3, 3.3, 4.3, 5.3]
        ax.set_xticks(x_ticks_loc)
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.set_title(param, fontsize=label_fontsize)
        ax.yaxis.offsetText.set_fontsize(label_fontsize)
        if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
        if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        if "sci" in y.keys() and y["sci"]: ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if "sci" in x.keys() and x["sci"]: ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.1)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.1)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.1)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.1)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'],loc="lower left", ncol=3, fontsize=legend_fontsize)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

# Ploting function for multi-line chart 
def plotMultiLineChart(x, y, path="", fig_dim=(1,1), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting multiline chart to " + path)
    # plt.style.use(['ggplot'])
    fig, ax = plt.subplots(1, figsize=fig_size, dpi=200)
    for i, param in enumerate(y["data"].keys()):
        ax.plot(x["data"], y['data'][param], 
                label=param, 
                ls=line_styles[i%len(line_styles)],  # line_styles[i%len(line_styles)], 
                marker=mark_cycle[i%len(y['data'])], 
                markersize=markersize_arg,
                fillstyle="none",
                # color= next(ax._get_lines.prop_cycler)['color'], # line_color_cycle[i],
                linewidth=plot_linewidth,
                # alpha=0.9 # transparency
                )
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center") 
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    if "title" in kwargs: ax.set_title(kwargs["title"], y=1.15, pad=-10, fontsize=label_fontsize)
    if y["log"]: ax.set_yscale('log',base=y["log"])
    if x["log"]: ax.set_xscale('log',base=x["log"])
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=kwargs["bbox_to_anchor"], loc='lower left', ncol=1, fontsize=legend_fontsize)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

# Ploting function for line subplot
def plotLineSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5)):
    print("[ANALYSIS] Plotting multiline chart to " + path)
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    for index, param in enumerate(y["data"].keys()):
        i, j = index // 3, index % 3
        ax = axes[i, j]
        ax.plot(x["data"], y['data'][param], 
                label=param, 
                ls=line_styles[i%len(line_styles)],  # line_styles[i%len(line_styles)], 
                marker="o", #mark_cycle[i%4], 
                markerfacecolor='none', 
                markersize=markersize_arg,
                fillstyle=None,
                color= next(ax._get_lines.prop_cycler)['color'], # line_color_cycle[i],
                linewidth=plot_linewidth,
                alpha=0.5 # transparency
                )
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), ) # rotation=45, ha="right"
        # ax.yaxis.offsetText.set_fontsize(label_fontsize)
        if y["log"]: ax.set_yscale('log',base=y["log"])
        if x["log"]: ax.set_xscale('log',base=x["log"])
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
        ax.legend(fontsize=legend_fontsize, ncol=1)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

# Ploting function for multi-line subplot
def plotMultiLineSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting multiline chart to " + path)
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel() if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
    for index, param in enumerate(y["data"].keys()):
        ax = axes[index] if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
        for ind, s_param in enumerate(y['data'][param].keys()):
            ax.plot(x["data"][param], y['data'][param][s_param], 
                    label=s_param, 
                    ls=line_styles[ind%len(line_styles)],  # line_styles[i%len(line_styles)], 
                    marker=mark_cycle[ind%len(mark_cycle)], 
                    markerfacecolor='none', 
                    markersize=markersize_arg,
                    fillstyle=None,
                    # color= next(ax._get_lines.prop_cycler)['color'], # line_color_cycle[i],
                    linewidth=plot_linewidth,
                    )
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        if index == 0: ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.set_title(param, y=1.15, pad=-10, fontsize=label_fontsize)
        # ax.set_xticklabels(ax.get_xticklabels(), ) # rotation=45, ha="right"
        # ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
        ax.yaxis.offsetText.set_fontsize(label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
        if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        if "sci" in y.keys() and y["sci"]: ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if "sci" in x.keys() and x["sci"]: ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    # plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'], loc='lower left', fontsize=legend_fontsize, ncol=4)
    if path: plt.savefig(path, dpi=200, transparent=False, bbox_inches='tight')
    else: plt.show()
    plt.close()

# Ploting function for multi-line subplot
def plotMultiScatterSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting multiline chart to " + path)
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel() if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
    for index, param in enumerate(y["data"].keys()):
        ax = axes[index] if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
        for ind, s_param in enumerate(y['data'][param].keys()):
            ax.scatter(x["data"][param], y['data'][param][s_param], 
                    label=s_param, 
                    # marker=mark_cycle[ind%len(mark_cycle)], 
                    alpha=0.8
                    )
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        if index == 0: ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.set_title(param, y=1, pad=-10, fontsize=label_fontsize)
        # ax.set_xticklabels(ax.get_xticklabels(), ) # rotation=45, ha="right"
        # ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
        ax.yaxis.offsetText.set_fontsize(label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
        if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        if "sci" in y.keys() and y["sci"]: ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if "sci" in x.keys() and x["sci"]: ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
        # ax.legend(fontsize=legend_fontsize, ncol=1)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'], loc='lower left', fontsize=legend_fontsize, ncol=1)
    if path: plt.savefig(path, dpi=200, transparent=False, bbox_inches='tight')
    else: plt.show()
    plt.close()

# Ploting function for multi-scatter chart 
def plotMultiScatterChart(x, y, path=""):
    print("[ANALYSIS] Plotting multiscatter chart to " + path)
    # plt.style.use(['ggplot'])
    fig, ax = plt.subplots(1, figsize=(2.5,2.5),dpi=200)
    for i, param in enumerate(y["data"].keys()):
        ax.scatter(x["data"], y['data'][param], 
                label=param, 
                marker="o", #mark_cycle[i%4],  
                color=next(ax._get_lines.prop_cycler)['color'], # line_color_cycle[i],
                )
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), ) # rotation=45, ha="right"
    # ax.yaxis.offsetText.set_fontsize(label_fontsize)
    if y["log"]: ax.set_yscale('log',base=y["log"])
    if x["log"]: ax.set_xscale('log',base=x["log"])
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    # plt.legend(bbox_to_anchor= (0, 1.05), loc='lower left', ncol=6, fontsize=12)
    plt.legend(fontsize=legend_fontsize, ncol=1)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()


# Ploting function for multi-line subplot
def plotAreaSubChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting area chart to " + path)
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel() if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
    for index, param in enumerate(y["data"].keys()):
        ax = axes[index] if not (fig_dim[0] == 1 and fig_dim[1] == 1) else axes
        ax.stackplot(x["data"][param], y['data'][param].values(), labels=y['data'][param].keys(), alpha=0.6)
        for vline_x in kwargs['reconf_times'][param]:
            ax.vlines(vline_x, ymin=-0.6, ymax=-0.2, linewidth=0.5, color="red", alpha=0.6)
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.set_title(param, y=1, fontsize=label_fontsize)
        # ax.set_xticklabels(ax.get_xticklabels()) # rotation=45, ha="right"
        # ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if y["log"]: ax.set_yscale('log',base=y["log"])
        if x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        ax.grid(which='major', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax.grid(which='major', axis='y', linestyle='--',linewidth=0.4)
        ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.4)
    plt.tight_layout()
    # plt.legend()
    # plt.legend(bbox_to_anchor=kwargs['bbox_to_anchor'], loc='lower left', fontsize=legend_fontsize, ncol=4) # (-1.4, -0.39) (-0.3, 3.95)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()

# Ploting function for multi-line subplot
def plotHeatMapChart(x, y, path="", fig_dim=(3,3), fig_size=(2.5,2.5), **kwargs):
    print("[ANALYSIS] Plotting heatmap chart to " + path)
    assert('matrices' in kwargs)
    fig, axes = plt.subplots(fig_dim[0], fig_dim[1], figsize=fig_size,dpi=200)
    axes = axes.ravel()
    for index, param in enumerate(kwargs['matrices'].keys()):
        ax = axes[index]
        im = ax.imshow(kwargs['matrices'][param], cmap='viridis')
        cbar = fig.colorbar(im, ax=ax, location='right', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        ax.set_xlabel(x["label"], fontsize=label_fontsize)
        ax.set_ylabel(y["label"], fontsize=label_fontsize)
        ax.set_xticks(np.arange(len(x['data'])), labels=x['data'])
        ax.set_yticks(np.arange(len(y['data'])), labels=y['data'])
        ax.tick_params(axis='x', labelsize=label_fontsize)
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.set_title(param, fontsize=label_fontsize)
        if y["log"]: ax.set_yscale('log',base=y["log"])
        if x["log"]: ax.set_xscale('log',base=x["log"])
        if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
        if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
        # add text
        for i in range(len(x['data'])):
            for j in range(len(y['data'])):
                text = ax.text(j, i, round(kwargs['matrices'][param][i][j], 3),ha="center", va="center", color="w", fontsize=tick_fontsize)
    plt.tight_layout()
    # plt.legend(bbox_to_anchor=(-1.4, -0.4), loc='lower left', fontsize=legend_fontsize, ncol=4) # (-1.4, -0.39) (-0.3, 3.95)
    if path: plt.savefig(path, dpi=200, transparent=True)
    else: plt.show()
    plt.close()
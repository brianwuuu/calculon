import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams['font.family'] = "serif"

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # ['darkcyan', 'lime', 'darkred','deeppink', 'blueviolet',  "silver", 'black']
mcolor_cycle = list(mcolors.TABLEAU_COLORS.values())
bar_color_cycle = ["lightblue", "cadetblue", "darkseagreen", "darkcyan", "olive", "slategray", "midnightblue", "darkslateblue"]
line_color_cycle = ["steelblue", "firebrick", "yellowgreen", "mediumpurple", "darkseagreen", "darkcyan",]
mark_cycle = ['s', 'x','v', 'd', '+', 's', 'x','v','1', 'p', ".", "o", "^", "<", ">", "1", "2", "3", "8", "P"]
line_styles = ["dashed","solid",  "dashdot", "dotted",]
bar_hatches = ["xx", "..", "//"]
markersize_arg = 2
legend_fontsize = 5
tick_fontsize = 6
label_fontsize = 6
plot_linewidth = 1.2
plotfont = {'fontname':'Times'}

def plotCDF(data, name):
    # getting data of the histogram 
    count, bins_count = np.histogram(data, bins=100) 
    # finding the PDF of the histogram using count values 
    pdf = count / sum(count) 
    # using numpy np.cumsum to calculate the CDF 
    # We can also find using the PDF values by looping and adding 
    cdf = np.cumsum(pdf) 
    # plotting CDF
    plt.plot(bins_count[1:], cdf, label=name) 
    plt.legend()
    plt.show()
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
    if path: plt.savefig(path, dpi=200)
    else: plt.show()
    plt.close()

def plotMultiColBarChart(x, y, path=""):
    print("[ANALYSIS] Plotting multi-column bar chart for " + y["label"] + " vs " + x["label"])
    num_pairs = len(x["data"])
    ind = np.arange(num_pairs)
    width = 0.4
    fig, ax = plt.subplots(1, figsize=(7,2.3),dpi=200)
    for i, parameter in enumerate(y["data"].keys()):
        ax.bar(ind+i*width, y["data"][parameter], label=parameter, width=width, color=bar_color_cycle[i%len(y["data"].keys())])
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    if "limit" in y.keys() and y["limit"]: ax.set_ylim(*y['limit'])
    if "limit" in x.keys() and x["limit"]: ax.set_xlim(*x['limit'])
    if "log" in y.keys() and y["log"]: ax.set_yscale('log',base=y["log"])
    if "log" in x.keys() and x["log"]: ax.set_xscale('log',base=x["log"])
    x_ticks_loc = [x + 0.2 for x in range(len(y["data"][list(y["data"].keys())[0]]))] # [0.3, 1.3, 2.3, 3.3, 4.3, 5.3]
    ax.set_xticks(x_ticks_loc)
    ax.set_xticklabels(x["data"], fontsize=5, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=6)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.3)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.3)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.3)
    plt.tight_layout()
    plt.legend(fontsize=legend_fontsize, ncol=2, loc="upper left")
    if path: plt.savefig(path, dpi=200)
    else: plt.show()
    plt.close()

# Ploting function for multi-line chart 
def plotMultiLineChart(x, y, path=""):
    print("[ANALYSIS] Plotting multiline chart to " + path)
    # plt.style.use(['ggplot'])
    fig, ax = plt.subplots(1, figsize=(3,2.3),dpi=200)
    for i, param in enumerate(y["data"].keys()):
        ax.plot(x["data"], y['data'][param], 
                label=param, 
                ls=line_styles[i%len(line_styles)],  # line_styles[i%len(line_styles)], 
                marker="o", #mark_cycle[i%4], 
                markerfacecolor='none', 
                markersize=markersize_arg,
                color=line_color_cycle[i],
                linewidth=plot_linewidth
                )
    ax.set_xlabel(x["label"], fontsize=label_fontsize)
    ax.set_ylabel(y["label"], fontsize=label_fontsize)
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)
    ax.yaxis.offsetText.set_fontsize(label_fontsize)
    if y["log"]: ax.set_yscale('log',base=y["log"])
    if x["log"]: ax.set_xscale('log',base=x["log"])
    ax.grid(which='major', axis='x', linestyle=':', linewidth=0.7)
    ax.grid(which='minor', axis='x', linestyle=':', linewidth=0.7)
    ax.grid(which='major', axis='y', linestyle='--',linewidth=0.7)
    ax.grid(which='minor', axis='y', linestyle='--',linewidth=0.7)
    plt.tight_layout()
    # plt.legend(bbox_to_anchor= (0, 1.05), loc='lower left', ncol=6, fontsize=12)
    plt.legend(fontsize=legend_fontsize, ncol=1)
    if path: plt.savefig(path, dpi=200)
    else: plt.show()
    plt.close()
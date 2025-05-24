import numpy as np
import matplotlib.pyplot as plt

# Hardware configurations: (label, peak_flops [GFLOPs], bandwidth [GB/s])
hardware_specs = [
    ("A100", 312e12, 2048e9),
    ("H100", 1000e12, 3072e9),
    ("B100", 3500e12, 8000e9),
]

colors = {
    "A100": "#66c2a5",  # teal
    "H100": "#fc8d62",  # salmon
    "B100": "#8da0cb",  # purplish blue
}

# Workload arithmetic intensities to annotate (FLOPs/Byte)
workload_ais = [
    (54, "CNN", "training"),
    (162, "DLRM", "training"),
    # (158, "BERT", "training"),
    # (351, "Ant", "training"),
    # (404, "Meg", "training"),
    # (840, "GPT3", "training"),
    (978, "GPT3-13B\nPF_Proj", "inference"),
    (1755, "GPT3-175B\nPF_Proj", "inference"),
    # (1024, "Llama2\nPF_Proj", "inference"),
    (114, "Llama2\nPF_MM", "inference"),
    (1, "Llama2\nDC_Proj", "inference"),
    # (0.99, "Llama2\nDC_MM", "inference"),
    (2, "GPT-2", "inference"),
]

# X-axis range (arithmetic intensity)
ai_range = np.logspace(-1, 5, 1000)

# Plot setup
fig, ax = plt.subplots(1, figsize=(2.8,1.6), dpi=300)

# Set fontsize
fontsize = 4

# Plot each roofline
for label, peak_flops, bandwidth in hardware_specs:
    roofline = np.minimum(bandwidth * ai_range, peak_flops)
    ax.plot(ai_range, roofline, linewidth=1.1, label=f'{label}', color=colors[label])

# Plot workload vertical lines
for ai,name,type in workload_ais:
    ax.axvline(ai, color="red", linestyle='--', linewidth=0.8)
    # ax.text(ai, ax.get_ylim()[0] * 1.5, f'{ai} FLOPs/Byte', rotation=90, color='red', ha='right', va='bottom', fontsize=6)

# Log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=fontsize)
ax.set_ylabel('Performance (FLOPs/s)', fontsize=fontsize)
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)

# Limits and grid
ax.set_xlim(0.1, 1e5)
ax.set_ylim(1e11, max(h[1] for h in hardware_specs) * 10)
ax.grid(True, which='both', linestyle='--', linewidth=0.1)

# Legend
ax.legend(fontsize=fontsize)
plt.tight_layout()
plt.show()

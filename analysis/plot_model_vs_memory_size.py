import matplotlib.pyplot as plt

def compute_model_size(num_params, datatype='float32'):
    """ 
    Does not account for activation memory yet
    
    Assume m_params = N, m_grad = N, m_opt = 2N
    """
    # Map datatype to number of bytes
    dtype_to_bytes = {
        'int8': 1,
        'float16': 2,
        'float32': 4,
        'float64': 8
    }

    # Get the number of bytes per parameter
    if datatype not in dtype_to_bytes:
        raise ValueError(f"Unsupported datatype: {datatype}. Supported datatypes are: {list(dtype_to_bytes.keys())}")

    num_bytes_per_param = dtype_to_bytes[datatype]

    # Compute memory size in bytes
    memory_size_bytes = num_params * 10**9 * 4 * num_bytes_per_param

    # Convert to gigabytes
    memory_size_GB = memory_size_bytes / 1024**3  # 1 GB = 1024^3 bytes

    return memory_size_GB

# Workloads with (year, num_params in billion, name)
workloads = [((2018, 6), 0.117, "GPT1"), # GPT1
            ((2018, 10), 0.340, "BERT"),  # BERT
            ((2019, 2), 1.5, "GPT2"), # GPT2
            ((2020, 5), 175, "GPT3"), # GPT3
            ((2021, 10), 530, "Megatron"), # Megatron-Turing
            ((2021, 12), 1200, "GLaM"), # GLaM
            ((2022, 4), 540, "PaLM"), # PaLM
            ((2023, 2), 65, "LLaMA"), # LLaMA 
            ((2023, 3), 1760, "GPT4"), # GPT4
            ((2023, 8), 1085, "PanGu"), # PanGu
            ((2023, 11), 314, "Grok 1"), # Grok 1
            ((2024, 12), 671, "DeepSeek-V3"), # DeepSeek-V3
            ((2025, 2), 2700, "Grok 3"), # Grok-3
            ]

# GPU memory sizes in GB
gpu_memory_sizes = {
    "A100": 40,
    "H100": 80,
    "B100": 192
}

colors = {
    "A100": "#66c2a5",  # teal
    "H100": "#fc8d62",  # salmon
    "B100": "#8da0cb",  # purplish blue
}

model_names = [x[2] for x in workloads]
model_sizes_GB = [compute_model_size(x[1]) for x in workloads]

# Set fontsize
fontsize = 4

# Plotting
plt.figure(figsize=(2.8,1.6), dpi=300)
bars = plt.bar(model_names, model_sizes_GB, color='floralwhite', edgecolor='black') 
# Add horizontal lines for GPU memory sizes
for label, size in gpu_memory_sizes.items():
    plt.axhline(y=size, linestyle='--', linewidth=0.8, label=f'{label} ({size} GB)', color=colors[label])

plt.yscale('log')

# Labels and styling
plt.ylabel("Model Size (GB)", fontsize=5)
# plt.xlabel("Model", fontsize=6)
plt.tick_params(axis='x', labelsize=fontsize)
plt.tick_params(axis='y', labelsize=fontsize)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.2)

plt.legend(fontsize=4)

plt.tight_layout()
plt.show()

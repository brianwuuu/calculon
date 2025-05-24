import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def year_month_to_float(year_month):
    year, month = year_month
    return year + (month - 1) / 12.0

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

# Input 1: Workloads with (year, num_params in billion, name)
workloads = [((2018, 6), 0.117, "GPT1"), # GPT1
            ((2018, 10), 0.340, "BERT"),  # BERT
            ((2019, 2), 1.5, ""), # GPT2
            ((2020, 5), 175, "GPT3"), # GPT3
            ((2021, 10), 530, ""), # Megatron-Turing
            ((2021, 12), 1200, "GLaM"), # GLaM
            ((2022, 4), 540, "PaLM"), # PaLM
            ((2023, 2), 65, "LLaMA"), # LLaMA 
            ((2023, 3), 1760, "GPT4"), # GPT4
            ((2023, 8), 1085, ""), # PanGu
            ((2023, 11), 314, "Grok 1"), # Grok 1
            ((2024, 12), 671, "DeepSeek-V3"), # DeepSeek-V3
            ((2025, 2), 2700, "Grok 3"), # Grok-3
            ]

# Input 2: GPU memory sizes (in GB)
gpu_memories = {
    "A100": 40,
    "H100": 80,
    "B100": 192,
}

# Organize data per GPU
years = []
ratios_by_gpu = {gpu: [] for gpu in gpu_memories}
years_by_gpu = {gpu: [] for gpu in gpu_memories}

for year, num_params, name in workloads:
    model_size = compute_model_size(num_params, datatype='float32')
    for gpu, mem_size in gpu_memories.items():
        ratio = model_size / mem_size
        ratios_by_gpu[gpu].append(ratio)
        years_by_gpu[gpu].append(year_month_to_float(year))

# Plotting
colors = {"A100": "#1f77b4", "H100": "#2ca02c", "B100": "#d62728"}
fig, ax = plt.subplots(figsize=(3,2), dpi=300)

for gpu in gpu_memories:
    x = np.array(years_by_gpu[gpu])
    y = np.array(ratios_by_gpu[gpu])
    
    # Scatter plot
    ax.bar(x, y, label=f'{gpu}', color=colors[gpu])
    # fit = np.polyfit(x, np.log(y), 1)
    # plt.plot(x, np.exp(fit[0] * x + fit[1]), color='red', linestyle='--')

# Set Y-axis to log scale
# plt.yscale('log')

# Labels and aesthetics
ax.set_xlabel("Year", fontsize=6)
ax.set_ylabel("Model Size / GPU Memory Size", fontsize=6)
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)

ax.legend()
ax.grid(True, linestyle='--', linewidth=0.2)

plt.tight_layout()
plt.show()

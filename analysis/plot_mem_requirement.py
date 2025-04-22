import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('ggplot')

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

# Sample data: ((year, month), num_params billions)
llm_data_raw = [((2018, 6), 0.117, "GPT1"), # GPT1
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
# https://www.techpowerup.com/gpu-specs/?sort=released
gpu_hbm_data_raw = [((2018, 3), 32),
                   ((2020, 6), 40),
                   ((2020, 11), 80),
                   ((2021, 6), 80),
                   ((2022, 11), 80),
                   ((2023, 3), 80),
                   ((2023, 3), 96),
                   ((2023, 1), 128),
                   ((2023, 12), 192),
                   ((2024, 11), 141),
                   ((2024, 12), 288),
                ]

llm_data_processed = [(year_month_to_float(date), compute_model_size(n, "float32"), name) for date, n, name in llm_data_raw]
gpu_memory_data_processed = [(year_month_to_float(date), size_GB) for date, size_GB in gpu_hbm_data_raw]

# Unpack the data into separate lists
llm_years, llm_sizes, llm_names = zip(*llm_data_processed)
gpu_years, gpu_sizes = zip(*gpu_memory_data_processed)

# Convert to numpy arrays for easier manipulation
llm_years = np.array(llm_years)
llm_sizes = np.array(llm_sizes)
gpu_years = np.array(gpu_years)
gpu_sizes = np.array(gpu_sizes)

# Plotting
plt.figure(figsize=(6,3.6))

# Plot LLM data
plt.scatter(llm_years, llm_sizes, color='blue', label='LLM Model Size (GB)')
llm_fit = np.polyfit(llm_years, np.log(llm_sizes), 1)
plt.plot(llm_years, np.exp(llm_fit[0] * llm_years + llm_fit[1]), color='blue', linestyle='--')

# Add label to LLM data:
for year, size, name in zip(llm_years, llm_sizes, llm_names):
    plt.text(year, size, name, fontsize=10, ha='right', va='bottom', rotation=0)

# Plot GPU memory data
plt.scatter(gpu_years, gpu_sizes, color='red', label='GPU Memory Size (GB)')
gpu_fit = np.polyfit(gpu_years, np.log(gpu_sizes), 1)
plt.plot(gpu_years, np.exp(gpu_fit[0] * gpu_years + gpu_fit[1]), color='red', linestyle='--')

# Set Y-axis to log scale
plt.yscale('log')

# plt.xlabel('Year', fontsize=12)
plt.ylabel('Size (GB)', fontsize=12)
plt.legend(fontsize=12)

# Increase tick label font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot with a grid
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
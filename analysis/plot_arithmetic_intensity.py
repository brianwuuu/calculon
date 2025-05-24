import matplotlib.pyplot as plt

# TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings (Fig 16)
# LLM Inference Unveiled: Survey and Roofline Model Insights (Table 1)
# AI and Memory Wall (Fig 3)
# NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing (Fig 4)
data = [
    (54, "CNN", "training"),
    (162, "DLRM", "training"),
    (158, "BERT", "training"),
    # (351, "Ant", "training"),
    # (404, "Meg", "training"),
    (840, "GPT3", "training"),
    (978, "GPT3-13B\nPF_Proj", "inference"),
    (1755, "GPT3-175B\nPF_Proj", "inference"),
    (1024, "Llama2\nPF_Proj", "inference"),
    (114, "Llama2\nPF_MM", "inference"),
    (1, "Llama2\nDC_Proj", "inference"),
    (0.99, "Llama2\nDC_MM", "inference"),
    (2, "GPT-2", "inference"),
]

# Extracting values
arithmetic_intensity = [item[0] for item in data]
model_names = [item[1] for item in data]
labels = [item[2] for item in data]

# Define colors for training and inference
colors = ["#1f77b4" if label == "training" else "#ff7f0e" for label in labels]  # Blue for training, Orange for inference

# Define hardware compute intensity
a100_intensity = 152 
h100_intensity = 333 
b100_intensity = 437.5 

# Plot bar chart
plt.figure(figsize=(6,3.6))
plt.bar(model_names, arithmetic_intensity, color=colors, edgecolor='black')

# Set y-axis to log scale
plt.yscale('log')

# Add horizontal line for hardware compute intensity
plt.axhline(y=a100_intensity, color='red', linestyle='--', linewidth=2)
plt.axhline(y=h100_intensity, color='red', linestyle='--', linewidth=2)
plt.axhline(y=b100_intensity, color='red', linestyle='--', linewidth=2)

# Increase tick label font size
plt.xticks(fontsize=10) # rotation=45, ha='right'
plt.yticks(fontsize=12)

# Labels and title
plt.ylabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=12)
plt.grid(axis='y', which="both", ls="--", alpha=0.5)

# Show plot
plt.tight_layout()
plt.show()

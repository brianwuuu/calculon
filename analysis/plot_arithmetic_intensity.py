import matplotlib.pyplot as plt

# TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings
data = [
    (162, "DLRM", "training"),
    (158, "BERT", "training"),
    (351, "Ant", "training"),
    (404, "Meg", "training"),
    (840, "GPT3", "training"),
    (1024, "Llama2\nPF_Proj", "inference"),
    (114, "Llama2\nPF_MM", "inference"),
    (1, "Llama2\nDC_Proj", "inference"),
    (0.99, "Llama2\nDC_MM", "inference"),
]

# Extracting values
arithmetic_intensity = [item[0] for item in data]
model_names = [item[1] for item in data]
labels = [item[2] for item in data]

# Define colors for training and inference
colors = ["#1f77b4" if label == "training" else "#ff7f0e" for label in labels]  # Blue for training, Orange for inference

# Define hardware compute intensity
hardware_intensity = 333 

# Plot bar chart
plt.figure(figsize=(6,3.6))
plt.bar(model_names, arithmetic_intensity, color=colors, edgecolor='black')

# Set y-axis to log scale
plt.yscale('log')

# Add horizontal line for hardware compute intensity
plt.axhline(y=hardware_intensity, color='red', linestyle='--', linewidth=2)

# Increase tick label font size
plt.xticks(fontsize=10) # rotation=45, ha='right'
plt.yticks(fontsize=12)

# Labels and title
plt.ylabel("Arithmetic Intensity (FLOPS/Byte)", fontsize=12)
plt.grid(axis='y', which="both", ls="--", alpha=0.5)

# Show plot
plt.tight_layout()
plt.show()

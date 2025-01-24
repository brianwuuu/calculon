import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data for the first radar chart
categories = ['Capacity', 'Latency', 'Execution Time', 'Bandwidth', 'Power']
values_1 = [2, 5, 3, 3, 3]

# Data for the second radar chart
values_2 = [5, 3, 4, 4, 4]

# Data for the third radar chart
values_3 = [5, 3, 3, 3, 3]

num_vars = len(categories)

# Create a list of angles for the categories
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop for both data sets

# Append the first value to close the radar chart loop
values_1 += values_1[:1]
values_2 += values_2[:1]
values_3 += values_3[:1]

# Initialize the spider chart
fig, ax = plt.subplots(figsize=(2.5,2.5), subplot_kw=dict(polar=True), dpi=200)

# Draw one axis per category and add the labels
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw the lines around the chart with straight lines for y-axis grid
ax.set_rlabel_position(0)
plt.xticks(angles[:-1], categories)

# Fix the y-axis scale (radial axis)
ax.set_ylim(0, 5)  # This sets the range from 0 to 5 regardless of the dataset

# Disable the radial circular grid lines
ax.grid(False)

# Remove the outermost circular contour (spine)
ax.spines['polar'].set_visible(False)

# Draw radial lines manually as straight lines from the center to the edge
for angle in angles[:-1]:  # Exclude the duplicate angle at the end
    ax.plot([angle, angle], [0, 5], color='gray', linewidth=0.5)

# Draw concentric circles as straight segments for each value level
for i in range(1, 6):
    ax.plot(angles, [i] * len(angles), color='gray', linewidth=0.5, linestyle='-')

# Draw y-labels
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], color="grey", size=8)

# Plot the first dataset
ax.plot(angles, values_1, linewidth=2, linestyle='solid', label='H100 Compute System')
ax.fill(angles, values_1, 'b', alpha=0.1)

# Plot the second dataset
ax.plot(angles, values_2, linewidth=2, linestyle='solid', label='SiPAM Memory Pooling System', color='r')
ax.fill(angles, values_2, 'r', alpha=0.1)

# Plot the third dataset
ax.plot(angles, values_3, linewidth=2, linestyle='solid', label='Prior Memory Pooling System', color='g')
ax.fill(angles, values_3, 'g', alpha=0.1)

ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)

plt.tight_layout()
# Add a legend
plt.legend(loc='lower left', bbox_to_anchor=(0.1, -0.3), fontsize=5)

plt.show()
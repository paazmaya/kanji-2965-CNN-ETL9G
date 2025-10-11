"""
Create comprehensive training comparison visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load CNN training data
with open('models/training_progress.json', 'r') as f:
    cnn_data = json.load(f)

# Load RNN training data from terminal output (manually extracted)
rnn_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
rnn_train_loss = [8.0239, 7.9272, 3.1385, 0.6611, 0.3706, 0.2714, 0.2237, 0.1975, 0.1776, 0.1630, 0.1537, 0.1424, 0.1368, 0.1313, 0.1274, 0.1233, 0.1205, 0.1180, 0.1156, 0.1124, 0.1116, 0.1111]
rnn_val_loss = [8.0234, 6.8822, 0.5625, 0.2334, 0.1422, 0.1159, 0.1018, 0.0972, 0.0809, 0.0881, 0.1024, 0.0833, 0.0779, 0.0912, 0.0772, 0.0733, 0.0812, 0.0685, 0.0717, 0.0707, 0.0945, 0.0790]
rnn_val_acc = [0.02, 0.50, 84.40, 93.26, 96.01, 96.67, 97.16, 97.33, 97.80, 97.70, 97.35, 97.80, 98.01, 97.69, 98.07, 98.17, 97.95, 98.40, 98.27, 98.26, 97.76, 98.24]

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CNN vs RNN Training Comparison', fontsize=16, fontweight='bold')

# Plot 1: Training Loss Comparison
ax1.plot(cnn_data['epochs'], cnn_data['train_loss'], 'b-', label='CNN Train Loss', linewidth=2)
ax1.plot(rnn_epochs, rnn_train_loss, 'r-', label='RNN Train Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Validation Loss Comparison
ax2.plot(cnn_data['epochs'], cnn_data['val_loss'], 'b-', label='CNN Val Loss', linewidth=2)
ax2.plot(rnn_epochs, rnn_val_loss, 'r-', label='RNN Val Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Validation Accuracy Comparison
ax3.plot(cnn_data['epochs'], cnn_data['val_acc'], 'b-', label='CNN Val Accuracy', linewidth=2)
ax3.plot(rnn_epochs, rnn_val_acc, 'r-', label='RNN Val Accuracy', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Accuracy (%)')
ax3.set_title('Validation Accuracy Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 100)

# Plot 4: Training Efficiency (Accuracy vs Time)
# Calculate cumulative training time (approximate)
cnn_time_per_epoch = 9  # minutes
rnn_time_per_epoch = 60  # minutes

cnn_cumulative_time = [i * cnn_time_per_epoch / 60 for i in cnn_data['epochs']]  # hours
rnn_cumulative_time = [i * rnn_time_per_epoch / 60 for i in rnn_epochs]  # hours

ax4.plot(cnn_cumulative_time, cnn_data['val_acc'], 'b-', label='CNN', linewidth=2, marker='o', markersize=4)
ax4.plot(rnn_cumulative_time, rnn_val_acc, 'r-', label='RNN', linewidth=2, marker='s', markersize=4)
ax4.set_xlabel('Training Time (hours)')
ax4.set_ylabel('Validation Accuracy (%)')
ax4.set_title('Training Efficiency: Accuracy vs Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add final performance annotations
ax4.annotate(f'CNN Final: {cnn_data["val_acc"][-1]:.1f}%\n@{cnn_cumulative_time[-1]:.1f}h', 
            xy=(cnn_cumulative_time[-1], cnn_data["val_acc"][-1]), 
            xytext=(cnn_cumulative_time[-1]+1, cnn_data["val_acc"][-1]-2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='blue'))

ax4.annotate(f'RNN Peak: {max(rnn_val_acc):.1f}%\n@{rnn_cumulative_time[rnn_val_acc.index(max(rnn_val_acc))]:.1f}h', 
            xy=(rnn_cumulative_time[rnn_val_acc.index(max(rnn_val_acc))], max(rnn_val_acc)), 
            xytext=(rnn_cumulative_time[rnn_val_acc.index(max(rnn_val_acc))]+1, max(rnn_val_acc)+1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('models/training_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a separate focused plot for the RNN training progression
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('RNN (Hybrid CNN-RNN) Training Progression', fontsize=16, fontweight='bold')

# RNN Loss curves
ax1.plot(rnn_epochs, rnn_train_loss, 'b-', label='Train Loss', linewidth=2, marker='o')
ax1.plot(rnn_epochs, rnn_val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('RNN Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Add annotations for key points
ax1.annotate('Fast convergence', xy=(3, 0.5625), xytext=(5, 2),
            arrowprops=dict(arrowstyle='->', color='green'),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# RNN Validation Accuracy
ax2.plot(rnn_epochs, rnn_val_acc, 'g-', label='Validation Accuracy', linewidth=3, marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_title('RNN Validation Accuracy')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(90, 100)

# Highlight peak performance
peak_epoch = rnn_val_acc.index(max(rnn_val_acc)) + 1
ax2.annotate(f'Peak: {max(rnn_val_acc):.2f}%\nEpoch {rnn_epochs[peak_epoch-1]}', 
            xy=(rnn_epochs[peak_epoch-1], max(rnn_val_acc)), 
            xytext=(rnn_epochs[peak_epoch-1]+2, max(rnn_val_acc)-0.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='orange'))

# Add horizontal line for CNN best performance
ax2.axhline(y=97.18, color='blue', linestyle='--', alpha=0.7, label='CNN Best (97.18%)')
ax2.legend()

plt.tight_layout()
plt.savefig('models/rnn/hybrid_cnn_rnn_training_curves_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training comparison visualizations created:")
print("1. models/training_comparison_comprehensive.png - Full CNN vs RNN comparison")
print("2. models/rnn/hybrid_cnn_rnn_training_curves_corrected.png - Corrected RNN curves")
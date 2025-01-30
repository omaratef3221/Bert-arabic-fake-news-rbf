import matplotlib.pyplot as plt
import numpy as np

def create_plots(total_loss_train_plot, 
                 total_loss_validation_plot, 
                 total_acc_train_plot, 
                 total_acc_validation_plot, 
                 epochs, 
                 plot_path):
    
    epochs = np.arange(1, epochs+1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(plot_path.split('/')[1].replace('.png', ''), fontsize=14, fontweight='bold')

    # Plot training and validation loss in one plot
    axs[0].plot(epochs, total_loss_train_plot, label='Train Loss', color='blue')
    axs[0].plot(epochs, total_loss_validation_plot, label='Validation Loss', color='red')
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(epochs, total_acc_train_plot, label='Train Accuracy', color='green')
    axs[1].plot(epochs, total_acc_validation_plot, label='Validation Accuracy', color='orange')
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)


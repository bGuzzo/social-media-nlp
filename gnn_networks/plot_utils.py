"""
This script provides a suite of utility functions for generating and saving high-resolution
visualizations of model performance metrics during both the training and testing phases.
It is designed to produce clear and informative plots that are essential for the analysis
and interpretation of the GNN model's behavior.

The core functionalities of this script include:

1.  **Training Performance Visualization**:
    - `plot_train_multiple_auc`: This function generates line plots that track the evolution
      of both the standard link prediction AUC and the custom Agnostic AUC (A-AUC) over
      the training steps. This allows for a direct comparison of the model's ability to
      learn the explicit graph structure versus the underlying semantic similarities.
    - `plot_train_loss`: This function creates a line plot of the training loss over time,
      providing a fundamental diagnostic for assessing the convergence and stability of the
      training process.

2.  **Testing Performance Visualization**:
    - `plot_test_res`: This function produces a bar chart that summarizes the final AUC and
      A-AUC scores for each item in the test set. It also includes the average scores,
      offering a concise and comparative view of the model's generalization performance.

3.  **High-Quality Output**:
   All plotting functions are configured to generate high-resolution
   images suitable for academic reports and presentations. They include detailed titles,
   labels, and legends to ensure that the plots are self-explanatory.

These visualization tools are integral to the research and development process, enabling a
deep and nuanced understanding of the model's learning dynamics and its final performance.
"""

import time
import matplotlib.pyplot as plt
import os
import numpy as np

# Default folder for saving the plots.
PLOT_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/plots"
# Default plot dimensions and resolution.
DEF_X_SIZE = 20
DEF_Y_SIZE = 8
DEF_DPI = 650

def plot_train_multiple_auc(
    model_name: str, 
    total_epochs: int,
    dataset_size: int,
    auc: list[float], 
    a_auc: list[float],
    a_auc_cos_tresh: float,
    plot_folder:str = PLOT_FOLDER,
    x_size: int = DEF_X_SIZE,
    y_size: int = DEF_Y_SIZE,
    dpi: int = DEF_DPI
) -> None:
    """
    Generates and saves a plot of training AUC and A-AUC scores over training steps.

    Args:
        model_name (str): The name of the model being trained.
        total_epochs (int): The total number of epochs.
        dataset_size (int): The size of the training dataset.
        auc (list[float]): A list of AUC scores for each training step.
        a_auc (list[float]): A list of A-AUC scores for each training step.
        a_auc_cos_tresh (float): The cosine similarity threshold used for A-AUC.
        plot_folder (str, optional): The folder to save the plot. Defaults to PLOT_FOLDER.
        x_size (int, optional): The width of the plot. Defaults to DEF_X_SIZE.
        y_size (int, optional): The height of the plot. Defaults to DEF_Y_SIZE.
        dpi (int, optional): The resolution of the plot. Defaults to DEF_DPI.
    """

    train_steps = list(range(1, len(auc) + 1))
    title = f"AUC, A-AUC vs. Train steps\nModel: {model_name}\n#Data points: {dataset_size}, #Epochs: {total_epochs}, #Train steps: {len(train_steps)}"
    
    fig, ax = plt.subplots(figsize=(x_size, y_size), dpi=dpi)

    ax.plot(train_steps, auc, marker='o', linestyle='-', label="AUC")
    ax.plot(train_steps, a_auc, marker='x', linestyle='--', label=f"A-AUC({a_auc_cos_tresh:.2f})")

    ax.set_xlabel("Train steps")
    ax.set_ylabel("AUC value")
    ax.set_title(title)
    ax.set_xticks(train_steps)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    plt.savefig(os.path.join(plot_folder, f"train_auc_{model_name}_{dataset_size}_{total_epochs}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"), dpi=dpi)
    plt.close(fig)

def plot_train_loss(
    model_name: str, 
    total_epochs: int,
    dataset_size: int,
    loss_list: list[float], 
    plot_folder:str = PLOT_FOLDER,
    x_size: int = DEF_X_SIZE,
    y_size: int = DEF_Y_SIZE,
    dpi: int = DEF_DPI
) -> None:
    """
    Generates and saves a plot of the training loss over training steps.

    Args:
        model_name (str): The name of the model being trained.
        total_epochs (int): The total number of epochs.
        dataset_size (int): The size of the training dataset.
        loss_list (list[float]): A list of loss values for each training step.
        plot_folder (str, optional): The folder to save the plot. Defaults to PLOT_FOLDER.
        x_size (int, optional): The width of the plot. Defaults to DEF_X_SIZE.
        y_size (int, optional): The height of the plot. Defaults to DEF_Y_SIZE.
        dpi (int, optional): The resolution of the plot. Defaults to DEF_DPI.
    """
    
    train_steps = list(range(1, len(loss_list) + 1))
    title = f"Loss vs. Train steps\nModel: {model_name}\n#Train data points: {dataset_size}, #Epochs: {total_epochs}, #Train steps: {len(train_steps)}"
    
    fig, ax = plt.subplots(figsize=(x_size, y_size), dpi=dpi)

    ax.plot(train_steps, loss_list, marker='o', linestyle='-', label="Loss", color="red")

    ax.set_xlabel("Train steps")
    ax.set_ylabel("Loss value")
    ax.set_title(title)
    ax.set_xticks(train_steps)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    plt.savefig(os.path.join(plot_folder, f"train_loss_{model_name}_{dataset_size}_{total_epochs}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"), dpi=dpi)
    plt.close(fig)

def plot_test_res(
    model_name: str, 
    train_num_steps: int,
    train_total_epochs: int,
    train_dataset_size: int,
    auc_list: list[float], 
    a_auc_list: list[float],
    a_auc_cos_tresh: float,
    plot_folder:str = PLOT_FOLDER,
    x_size: int = DEF_X_SIZE,
    y_size: int = DEF_Y_SIZE,
    dpi: int = DEF_DPI
) -> None:
    """
    Generates and saves a bar chart of test AUC and A-AUC scores.

    Args:
        model_name (str): The name of the model being tested.
        train_num_steps (int): The total number of training steps.
        train_total_epochs (int): The total number of training epochs.
        train_dataset_size (int): The size of the training dataset.
        auc_list (list[float]): A list of AUC scores for each test item.
        a_auc_list (list[float]): A list of A-AUC scores for each test item.
        a_auc_cos_tresh (float): The cosine similarity threshold used for A-AUC.
        plot_folder (str, optional): The folder to save the plot. Defaults to PLOT_FOLDER.
        x_size (int, optional): The width of the plot. Defaults to DEF_X_SIZE.
        y_size (int, optional): The height of the plot. Defaults to DEF_Y_SIZE.
        dpi (int, optional): The resolution of the plot. Defaults to DEF_DPI.
    """
    
    title = f"Test AUC, A-AUC vs. Test items\nModel: {model_name}\n#Train data points: {train_dataset_size}, #Epochs: {train_total_epochs}, #Train steps: {train_num_steps}\n#Test data points: {len(auc_list)}"

    x = np.arange(len(auc_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(x_size, y_size), dpi=dpi)

    ax.bar(x - width/2, auc_list, width, label="AUC", color="skyblue")
    ax.bar(x + width/2, a_auc_list, width, label=f"A-AUC({a_auc_cos_tresh:.2f})", color="lightcoral")

    # Calculate and plot averages
    avg_auc = np.mean(auc_list)
    avg_a_auc = np.mean(a_auc_list)

    ax.axhline(y=avg_auc, color='royalblue', linestyle='--', label=f'Avg. AUC ({avg_auc:.3f})')
    ax.axhline(y=avg_a_auc, color='crimson', linestyle='--', label=f'Avg. A-AUC ({avg_a_auc:.3f})')

    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    
    plt.savefig(os.path.join(plot_folder, f"test_auc_{len(auc_list)}_{model_name}_{train_dataset_size}_{train_total_epochs}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"), dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    # This block serves as a demonstration of the plotting functions.
    # It creates and saves example plots for training and testing metrics.
    model_name = "MyModel"
    total_epochs = 10
    dataset_size = 1000
    auc = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94]
    a_auc = [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.86, 0.87, 0.88, 0.89]
    a_auc_cos_tresh = 0.8

    plot_train_multiple_auc(model_name, total_epochs, dataset_size, auc, a_auc, a_auc_cos_tresh, plot_folder=PLOT_FOLDER)
    plot_test_res(
        model_name=model_name, 
        train_num_steps=len(auc), 
        train_total_epochs=total_epochs, 
        train_dataset_size=dataset_size, 
        auc_list=auc, 
        a_auc_list=a_auc, 
        a_auc_cos_tresh=a_auc_cos_tresh, 
        plot_folder=PLOT_FOLDER
    )

    print("Example plots generated and saved.")

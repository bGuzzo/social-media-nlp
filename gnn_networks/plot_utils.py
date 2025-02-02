import time
import matplotlib.pyplot as plt
import os

import numpy as np

PLOT_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/plots"
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

    train_steps = list(range(1, len(auc) + 1))
    title = "AUC, A-AUC vs. Train steps" + \
            f"\nModel: {model_name}" + \
            f"\n#Data points: {dataset_size}, #Epochs: {total_epochs}, #Train steps: {len(train_steps)}"
    
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
    
    train_steps = list(range(1, len(loss_list) + 1))
    title = "Loss vs. Train steps" + \
            f"\nModel: {model_name}" + \
            f"\n#Train data points: {dataset_size}, #Epochs: {total_epochs}, #Train steps: {len(train_steps)}"
    
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
    
    title = "Test AUC, A-AUC vs. Test items" + \
            f"\nModel: {model_name}" + \
            f"\n#Train data points: {train_dataset_size}, #Epochs: {train_total_epochs}, #Train steps: {train_num_steps}" + \
            f"\n#Test data points: {len(auc_list)}"


    # n_items = len(item_names)
    x = np.arange(len(auc_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(x_size, y_size), dpi=dpi)  # Set figure size and DPI

    ax.bar(x - width/2, auc_list, width, label="AUC", color="skyblue")
    ax.bar(x + width/2, a_auc_list, width, label=f"A-AUC({a_auc_cos_tresh:.2f})", color="lightcoral")

    # Calculate and plot averages
    avg_auc = np.mean(auc_list)
    avg_a_auc = np.mean(a_auc_list)

    ax.axhline(y=avg_auc, color='royalblue', linestyle='--', label=f'Avg. AUC ({avg_auc:.3f})')
    ax.axhline(y=avg_a_auc, color='crimson', linestyle='--', label=f'Avg. A-AUC ({avg_a_auc:.3f})')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()  # Adjust layout to prevent labels from overlapping
    
    plt.savefig(os.path.join(plot_folder, f"test_auc_{len(auc_list)}_{model_name}_{train_dataset_size}_{train_total_epochs}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"), dpi=dpi)
    plt.close(fig) # Close the figure to free memory


# Test only
if __name__ == "__main__":
    # Test the function:
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

    print("Plot shown and saved. Continuing with code execution...")

    # Simulate some other work happening while the plot is displayed
    import time
    time.sleep(5)  # Wait for 5 seconds
    print("Continuing with other tasks...")
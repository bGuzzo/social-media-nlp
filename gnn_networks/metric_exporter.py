"""
This script provides a set of utility functions for the systematic exportation of model
performance metrics to JSON files. It is designed to capture a comprehensive snapshot
of the model's state and performance at both the training and testing phases, ensuring
that all relevant data is preserved for subsequent analysis and reporting.

The primary functionalities of this script are:

1.  **Training Metric Exportation (`dump_train_metric`)**:
    - This function is responsible for collecting and saving a wide range of metrics
      generated during the model training process.
    - It captures not only the performance metrics such as AUC, A-AUC, and loss per epoch,
      but also contextual information like the model name, total epochs, and dataset size.
    - The collected data is serialized into a structured JSON format and saved with a
      timestamped filename, creating a permanent and detailed record of each training run.

2.  **Testing Metric Exportation (`dump_test_metric`)**:
    - This function is tailored for the exportation of metrics from the model evaluation phase.
    - It records the performance of the trained model on the test dataset, including the
      list of AUC and A-AUC scores for each test item, as well as their averages.
    - Similar to the training metric dump, it also includes metadata about the training
      process to provide full context for the evaluation results.

3.  **Generic JSON Dumping (`__dump_json`)**:
    - A private helper function that handles the low-level details of serializing a Python
      dictionary to a JSON file, ensuring a consistent and readable format.

By systematically exporting these metrics, this script plays a crucial role in maintaining
the traceability and reproducibility of the experimental results, which is a cornerstone of
rigorous academic research.
"""

import json
import os
import time

# Default output folder for the JSON metric files.
JSON_METRIC_OUT_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/metric outputs"

def __dump_json(
    json_obj: dict,
    json_file_name: str,
    json_metric_folder_ump: str = JSON_METRIC_OUT_FOLDER
) -> None:
    """
    Serializes a dictionary to a JSON file and saves it to the specified folder.

    Args:
        json_obj (dict): The dictionary to be serialized.
        json_file_name (str): The name of the output JSON file.
        json_metric_folder_ump (str, optional): The folder where the JSON file will be saved.
                                                Defaults to JSON_METRIC_OUT_FOLDER.
    """
    json_str = json.dumps(json_obj, indent=4)
    json_file_path = os.path.join(json_metric_folder_ump, json_file_name)
    with open(json_file_path, "w") as json_file:
        json_file.write(json_str)

def dump_train_metric(
    model_name: str, 
    total_epochs: int,
    train_dataset_size: int,
    train_num_steps: int,
    auc_epoch_list: list[float], 
    a_auc_epoch_list: list[float],
    loss_epoch_list: list[float],
    a_auc_cos_tresh: float,
    json_metric_folder_dump: str = JSON_METRIC_OUT_FOLDER
) -> None:
    """
    Exports a comprehensive set of training metrics to a JSON file.

    Args:
        model_name (str): The name of the model being trained.
        total_epochs (int): The total number of epochs.
        train_dataset_size (int): The size of the training dataset.
        train_num_steps (int): The total number of training steps.
        auc_epoch_list (list[float]): A list of average AUC scores for each epoch.
        a_auc_epoch_list (list[float]): A list of average A-AUC scores for each epoch.
        loss_epoch_list (list[float]): A list of average loss values for each epoch.
        a_auc_cos_tresh (float): The cosine similarity threshold used for A-AUC.
        json_metric_folder_dump (str, optional): The folder to save the JSON file.
                                                    Defaults to JSON_METRIC_OUT_FOLDER.
    """
    json_to_dump = {
        "mode": "TRAINING",
        "model_name": model_name,
        "train_total_epochs": total_epochs,
        "train_dataset_size": train_dataset_size,
        "train_num_steps": train_num_steps,
        "train_auc_epoch_list": auc_epoch_list,
        "train_a_auc_epoch_list": a_auc_epoch_list,
        "train_loss_epoch_list": loss_epoch_list,
        "train_auc_epoch_final": auc_epoch_list[-1],
        "train_auc_epoch_max": max(auc_epoch_list),
        "train_a_auc_epoch_final": a_auc_epoch_list[-1],
        "train_a_auc_epoch_max": max(a_auc_epoch_list),
        "train_loss_epoch_final": loss_epoch_list[-1],
        "train_loss_epoch_min": min(loss_epoch_list),
        "train_a_auc_cos_tresh": a_auc_cos_tresh
    }
    json_file_name = f"train_metric_{model_name}_{train_dataset_size}_{total_epochs}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    __dump_json(json_obj=json_to_dump, json_file_name=json_file_name, json_metric_folder_ump=json_metric_folder_dump)

def dump_test_metric(
    model_name: str, 
    total_epochs: int,
    train_dataset_size: int,
    train_num_steps: int,
    test_dataset_size: int,
    auc_test_list: list[float],
    a_auc_test_list: list[float],
    a_auc_cos_tresh: float,
    json_metric_folder_dump: str = JSON_METRIC_OUT_FOLDER
) -> None:
    """
    Exports a comprehensive set of testing metrics to a JSON file.

    Args:
        model_name (str): The name of the model being tested.
        total_epochs (int): The total number of epochs used for training.
        train_dataset_size (int): The size of the training dataset.
        train_num_steps (int): The total number of training steps.
        test_dataset_size (int): The size of the test dataset.
        auc_test_list (list[float]): A list of AUC scores for each test item.
        a_auc_test_list (list[float]): A list of A-AUC scores for each test item.
        a_auc_cos_tresh (float): The cosine similarity threshold used for A-AUC.
        json_metric_folder_dump (str, optional): The folder to save the JSON file.
                                                    Defaults to JSON_METRIC_OUT_FOLDER.
    """
    
    json_to_dump = {
        "mode": "TEST",
        "model_name": model_name,
        "train_total_epochs": total_epochs,
        "train_dataset_size": train_dataset_size,
        "train_num_steps": train_num_steps,
        "test_dataset_size": test_dataset_size,
        "test_auc_list": auc_test_list,
        "test_auc_avg": sum(auc_test_list) / len(auc_test_list),
        "test_a_auc_list": a_auc_test_list,
        "test_a_auc_avg": sum(a_auc_test_list) / len(a_auc_test_list),
        "test_a_auc_cos_tresh": a_auc_cos_tresh
    }
    json_file_name = f"test_metric_{model_name}_{test_dataset_size}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    __dump_json(json_obj=json_to_dump, json_file_name=json_file_name, json_metric_folder_ump=json_metric_folder_dump)

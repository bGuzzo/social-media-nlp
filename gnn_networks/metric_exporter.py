import json
import os
import time
JSON_METRIC_OUT_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/metric outputs"

def __dump_json(
    json_obj: dict,
    json_file_name: str,
    json_metric_folder_ump: str = JSON_METRIC_OUT_FOLDER
) -> None:
    json_str = json.dumps(json_obj, indent=4)
    json_file_path = os.path.join(json_metric_folder_ump, json_file_name)
    with open(json_file_path, "w") as json_file:
        json_file.write(json_str)
        json_file.close

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
    # Train parmas
    total_epochs: int,
    train_dataset_size: int,
    train_num_steps: int,
    # Test parmas
    test_dataset_size: int,
    auc_test_list: list[float],
    a_auc_test_list: list[float],
    a_auc_cos_tresh: float,
    json_metric_folder_dump: str = JSON_METRIC_OUT_FOLDER
) -> None:
    
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
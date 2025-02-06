import networkx as nx
import numpy as np
import sys, os
import logging
import community as co
import json
from tqdm import tqdm
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

JSON_METRICS_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_metrics"
JSON_DENSITY_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_metrics_density"
CSV_METRICS_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/csv_metrics/graph_metrics_dnesity.csv"

def __load_json_metrics(json_metric_folder: str = JSON_METRICS_FOLDER) -> list[dict]:
    dict_metric_list = []
    for filename in tqdm(os.listdir(json_metric_folder), desc="JSON metrics loop"):
        if filename.endswith(".json"):
            json_metric = json.load(open(os.path.join(json_metric_folder, filename)))
            dict_metric_list.append(json_metric)
    
    log.info(f"Loaded {len(dict_metric_list)} JSON metrics")
    return dict_metric_list


def __load_json_density(json_density_folder: str = JSON_DENSITY_FOLDER) -> list[dict]:
    dict_density_list = []
    for filename in tqdm(os.listdir(json_density_folder), desc="JSON metrics loop"):
        if filename.endswith(".json"):
            json_density = json.load(open(os.path.join(json_density_folder, filename)))
            dict_density_list.append(json_density)
    
    log.info(f"Loaded {len(dict_density_list)} JSON density")
    return dict_density_list

def __join_metrics(
    csv_metric_dict_list: list[dict], 
    csv_density_dict_list: list[dict]
) -> list[dict]:
    csv_dict_list = []
    for metric_dict in csv_metric_dict_list:
        filename = metric_dict["filename"]
        for density_dict in csv_density_dict_list:
            if density_dict["filename"] == filename:
                metric_dict["density"] = density_dict["density"]
                csv_dict_list.append(metric_dict)
                break
    return csv_dict_list
    
    

def __export_metrics_csv(
    csv_metric_dict_list: list[dict], 
    csv_out_file: str = CSV_METRICS_FILE
) -> None:
    with open(csv_out_file, "w", newline="") as csv_file:
        header = [
            "filename", 
            "num_nodes", 
            "num_edges",
            "average_degree", 
            "average_clustering", 
            "average_path_length",
            "modularity",
            "density",
            "is_connected"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_metric_dict_list)
        csv_file.flush()
        csv_file.close()
    log.info(f"Metrics exported to {csv_out_file}")
    
if __name__ == "__main__":
    dict_metric_list = __load_json_metrics()
    dict_density_list = __load_json_density()
    csv_dict_list = __join_metrics(dict_metric_list, dict_density_list)
    __export_metrics_csv(csv_dict_list)
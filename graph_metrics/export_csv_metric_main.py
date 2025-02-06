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
CSV_METRICS_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/csv_metrics/graph_metrics.csv"

def __load_json_metrics(json_metric_folder: str = JSON_METRICS_FOLDER) -> list[dict]:
    dict_metric_list = []
    for filename in tqdm(os.listdir(json_metric_folder), desc="JSON metrics loop"):
        if filename.endswith(".json"):
            json_metric = json.load(open(os.path.join(json_metric_folder, filename)))
            dict_metric_list.append(json_metric)
    
    log.info(f"Loaded {len(dict_metric_list)} JSON metrics")
    return dict_metric_list

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
    __export_metrics_csv(dict_metric_list)
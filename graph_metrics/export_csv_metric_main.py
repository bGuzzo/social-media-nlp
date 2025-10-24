"""
This script aggregates graph metrics from multiple JSON files into a single CSV file.
It is designed to facilitate the analysis of graph metrics across a dataset of graphs by
consolidating distributed JSON metric files into a unified tabular format.

The script performs the following main operations:
1.  **Load JSON Metrics**: It reads a directory of JSON files, where each file contains
    the computed metrics for a single graph.
2.  **Load JSON Density**: It separately loads graph density metrics from another
    directory of JSON files.
3.  **Join Metrics**: It joins the general metrics and the density metrics based on the
    filename, creating a comprehensive record for each graph.
4.  **Export to CSV**: The aggregated and joined metrics are then written to a single CSV
    file, with a header row defining the metric fields.

This script is a crucial component of the data analysis pipeline, enabling efficient
comparison and statistical analysis of the topological properties of the graph dataset.
"""

import sys, os
import logging
import json
from tqdm import tqdm
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Directory containing the JSON files with general graph metrics.
JSON_METRICS_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_metrics"
# Directory containing the JSON files with graph density metrics.
JSON_DENSITY_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_metrics_density"
# The output CSV file where the aggregated metrics will be saved.
CSV_METRICS_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/csv_metrics/graph_metrics_dnesity.csv"

def __load_json_metrics(json_metric_folder: str = JSON_METRICS_FOLDER) -> list[dict]:
    """
    Loads all JSON metric files from a specified folder into a list of dictionaries.

    Args:
        json_metric_folder (str, optional): The path to the folder containing the JSON metric files.
                                            Defaults to JSON_METRICS_FOLDER.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents the metrics of a graph.
    """
    dict_metric_list = []
    for filename in tqdm(os.listdir(json_metric_folder), desc="JSON metrics loop"):
        if filename.endswith(".json"):
            json_metric = json.load(open(os.path.join(json_metric_folder, filename)))
            dict_metric_list.append(json_metric)
    
    log.info(f"Loaded {len(dict_metric_list)} JSON metrics")
    return dict_metric_list


def __load_json_density(json_density_folder: str = JSON_DENSITY_FOLDER) -> list[dict]:
    """
    Loads all JSON density files from a specified folder into a list of dictionaries.

    Args:
        json_density_folder (str, optional): The path to the folder containing the JSON density files.
                                             Defaults to JSON_DENSITY_FOLDER.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents the density of a graph.
    """
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
    """
    Joins the general graph metrics with the density metrics based on the filename.

    Args:
        csv_metric_dict_list (list[dict]): A list of dictionaries with general graph metrics.
        csv_density_dict_list (list[dict]): A list of dictionaries with graph density metrics.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the combined metrics for a graph.
    """
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
    """
    Exports a list of metric dictionaries to a CSV file.

    Args:
        csv_metric_dict_list (list[dict]): A list of dictionaries containing the aggregated graph metrics.
        csv_out_file (str, optional): The path to the output CSV file. Defaults to CSV_METRICS_FILE.
    """
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
    # This script block orchestrates the loading, joining, and exporting of graph metrics.
    # It first loads the general and density metrics from their respective JSON folders,
    # then joins them into a single data structure, and finally exports the result to a CSV file.
    dict_metric_list = __load_json_metrics()
    dict_density_list = __load_json_density()
    csv_dict_list = __join_metrics(dict_metric_list, dict_density_list)
    __export_metrics_csv(csv_dict_list)

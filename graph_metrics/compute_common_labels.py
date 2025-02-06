import networkx as nx
import numpy as np
import sys, os
import logging
import community as co
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

JSON_DATASET_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
JOSN_OUT_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_common_nodes"


def __json_to_graph(json_graph: dict) -> nx.Graph:
    
    graph = nx.Graph()
    node_idx_map = {}

    for node in json_graph["nodes"]:
        node_idx_map[node["id"]] = node["label"]
        graph.add_node(node["label"])
    
    for edge in json_graph["edges"]:
        source_node_id = edge["source"]
        target_node_id = edge["target"]
        graph.add_edge(node_idx_map[source_node_id], node_idx_map[target_node_id])

    log.info(f"Build NetworX graph from JSON with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def __compue_common_nodes(curr_graph: nx.Graph, remaining_graphs: list[nx.Graph]) -> int:
    common_nodes = 0
    for cur_node in curr_graph.nodes:    
        for other_graph in remaining_graphs:
            
            if curr_graph == other_graph:
                # log.info("Found same graph, skipping")
                continue
            
            if cur_node in other_graph.nodes:
                common_nodes += 1
                break
                
    log.info(f"Found {common_nodes} common nodes")
    return common_nodes

def main():
    graphs_map: dict[str, nx.Graph] = {}
    for filename in tqdm(os.listdir(JSON_DATASET_DIR), desc="JSON graphs loop"):
            if filename.endswith(".json"):
                log.info(f"Loading NetworkX graphs, processing {filename}")
                json_graph = json.load(open(os.path.join(JSON_DATASET_DIR, filename)))
                graph = __json_to_graph(json_graph)
                graphs_map[filename] = graph
    log.info(f"Loaded {len(graphs_map)} graphs")
    
    for key, graph in tqdm(graphs_map.items(), total=len(graphs_map), desc="Graphs loop"):
        log.info(f"Computing common nodes count for {key}")
        common_nodes_count = __compue_common_nodes(graph, list(graphs_map.values()))
        log.info(f"Common nodes count for {key}: {common_nodes_count}")
        json_dict = {
            "filename": key,
            "common_nodes_count": common_nodes_count
        }
        json_str = json.dumps(json_dict, indent=4)
        json_out_filename = "common_nodes_count_" + key
        with open(os.path.join(JOSN_OUT_DIR, json_out_filename), "w") as file:
            file.write(json_str)
            file.close()
        log.info(f"Dict {json_dict} exported to {json_out_filename}")
        

if __name__ == "__main__":
    main()





# def __export_metrics_json(filename: str) -> None:
#     log.info(f"Processing {filename}")
    
#     filepath = os.path.join(JSON_DATASET_DIR, filename)
#     json_graph = json.load(open(filepath))
#     graph = __json_to_graph(json_graph)
#     metrics = export_metrics(graph)
#     log.info(f"Metrics for {filename}: {metrics}")
    
#     graph_name = filename.split(".")[0]
#     metric_file_name = f"{graph_name}_metrics.json"
#     metrics["filename"] = filename
#     metric_json_path = os.path.join(JSON_METRIC_EXPORT_DIR, metric_file_name)
#     str_metrics = json.dumps(metrics, indent=4)
#     with open(metric_json_path, "w") as file:    
#         file.write(str_metrics)
#         file.close()
    
#     log.info(f"Metrics exported to {metric_json_path}")
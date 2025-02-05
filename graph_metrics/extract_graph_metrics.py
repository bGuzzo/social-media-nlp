import networkx as nx
import numpy as np
import sys, os
import logging
import community as co

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)


def export_metrics(graph: nx.Graph) -> dict[str, float]:

    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    # Handle empty graph case to avoid division by zero errors
    if num_nodes == 0:  
        avg_degree = 0
    else:
        degrees = dict(graph.degree())  # Get node degree map
        avg_degree = sum(degrees.values()) / num_nodes # Compute average

    if num_nodes <=1:
        avg_path_length = 0
    else:
        try:
            avg_path_length = nx.average_shortest_path_length(graph)
        except nx.NetworkXError: 
            # Throws exception when graph is not connected
            # Use infinity as average path lenght
            avg_path_length = float('inf')


    if num_nodes <= 2:
        avg_clustering = 0
    else:
        avg_clustering = nx.average_clustering(graph)

    if num_nodes <= 1 or num_edges == 0:
        modularity = 0
    else:
        try:           
            # Use Louvain method
            partition = co.best_partition(graph)  # Compute communities
            modularity = co.modularity(partition, graph)
        except Exception as e:
            log.error(f"Error calculating modularity: {e}. Modularity set to None")
            modularity = None

    results = {
        "num_edges": num_edges,
        "num_nodes": num_nodes,
        "average_degree": avg_degree,
        "average_clustering": avg_clustering,
        "average_path_length": avg_path_length,
        "modularity": modularity,
    }

    log.info(f"Graph metrics computed: {results}")
    return results


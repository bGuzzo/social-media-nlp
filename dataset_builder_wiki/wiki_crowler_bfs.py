import wikipedia
from wikipedia.wikipedia import WikipediaPage
import networkx as nx
import logging
import matplotlib.pyplot as plt
import json

# Import parent script
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Default  graph size
DEF_MAX_NODES = 1000

def json_dump(graph: nx.Graph):
    node_ids_counter = 0
    node_id_map = {str : int}
    
    nodes_list = []
    edges_list = []
    
    for node in graph.nodes:
        node_id_map[str(node)] = node_ids_counter
        nodes_list.append({
            "id": node_ids_counter,
            "label": node
        })
        node_ids_counter += 1
    
    for edge in graph.edges:
        edges_list.append({
            "source": node_id_map[str(edge[0])],
            "target": node_id_map[str(edge[1])]
        })
    
    json_graph = {
        "nodes": nodes_list,
        "edges": edges_list
    }
    
    return json.dumps(json_graph, indent=4)


def check_title(page_title: str) -> WikipediaPage:
    # Check if the page exists
    try:
        page = wikipedia.page(page_title)
    except wikipedia.exceptions.PageError as e:
        log.error(f"Error: Page '{page_title}' not found.")
        raise e
    except wikipedia.exceptions.DisambiguationError as e:
        log.error(f"Disambiguation Error for '{page_title}': {e.options}")
        raise e
    
    # Page found
    return page


# Breadth first wikipedia crowler
def crawl(
    graph: nx.Graph, 
    queue: list, 
    visited: set, 
    max_nodes: int
):
    if not queue:
        log.info("Found empty queue")
        return
    
    if len(graph.nodes) >= max_nodes:
        log.info(f"Reached graph size limit {len(graph.nodes)}")
        return
    
    page_title = queue.pop(0)
    
    try:
        page = check_title(page_title)
    except Exception:
        log.error(f"Error: Page '{page_title}' not found. Branch skipped. Actual queue size {len(queue)}")
        if (len(queue) > 0):
            # Not an exit condition!
            crawl(
                graph=graph, 
                queue=queue, 
                visited=visited, 
                max_nodes=max_nodes
            )
            return

    # Enqueue node child
    for link_title in page.links:
        
        if link_title in visited:
            log.info(f"Already visited {link_title}")
            continue
        
        if len(graph.nodes) >= max_nodes:
            log.info(f"Reached graph size limit {len(graph.nodes)}")
            return
        
        graph.add_node(link_title)
        graph.add_edge(page_title, link_title)
        visited.add(link_title)
        queue.append(link_title)
    
    crawl(
        graph=graph, 
        queue=queue, 
        visited=visited, 
        max_nodes=max_nodes
    )


def extract_graph(
    root_article_title: str, 
    max_nodes: int = DEF_MAX_NODES, 
) -> nx.Graph:
    check_title(root_article_title)
    
    # Inizialize graph
    graph = nx.Graph()
    graph.add_node(root_article_title)
    visited = set(root_article_title)
    queue = [root_article_title]
    

    crawl(
        graph=graph, 
        queue=queue, 
        visited=visited, 
        max_nodes=max_nodes
    )
    
    if not nx.is_connected(graph):
        log.error("Graph is not connected")
        raise Exception("Graph is not connected")
    
    log.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

# Test only
if __name__ == "__main__":
    # SGD as root article
    graph = extract_graph(root_article_title="Carbon dioxide in Earth's atmosphere")
    
    # Draw the graph (adjust layout as needed)
    plt.figure(figsize=(12, 12))
    nx.draw(graph, with_labels=True, node_size=300, font_size=8, node_color="skyblue")
    plt.title("Environmental sustainability")
    plt.show()
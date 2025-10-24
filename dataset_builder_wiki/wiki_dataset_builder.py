"""
This script orchestrates the creation of a graph dataset from Wikipedia articles.
It is designed to read a list of root article titles from a file and then, for each title,
initiate a web crawling process to build a graph of interconnected articles. The resulting
graphs are then saved in a structured JSON format.

The script is built with efficiency in mind, employing a multi-threaded approach to
parallelize the graph creation process. This is particularly important given that web
crawling and graph construction can be time-consuming tasks.

The main functionalities of this script are:

1.  **Article Title Loading**: It reads a list of Wikipedia article titles from a specified
    text file. This list serves as the set of root nodes for the graph construction process.

2.  **Multi-threaded Graph Construction**: The script uses a `ThreadPoolExecutor` to manage a
    pool of worker threads. Each thread is assigned the task of building a graph for a single
    root article. This parallel execution significantly reduces the overall time required to
    build the dataset.

3.  **Graph Extraction and Serialization**: For each article, it invokes the `extract_graph`
    function from the `wiki_crowler_bfs` module to perform the breadth-first search (BFS)
    crawling of Wikipedia and construct the graph. The resulting graph is then serialized
    to a JSON string using the `json_dump` function from the same module.

4.  **Data Persistence**: The JSON representation of each graph is saved to a separate file,
    creating a dataset of individual graph files that can be further processed or analyzed.

This script is the first step in the data pipeline of this project, responsible for the
acquisition and initial structuring of the raw data from Wikipedia.
"""

import multiprocessing
import os, sys
import logging
import concurrent.futures as futures
from wiki_crowler_bfs import extract_graph, json_dump

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

log.info(f"CPU cores available: {multiprocessing.cpu_count()}")

# Default parameters for dataset creation
DEF_MAX_NODES = 5000
FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"
WIKI_ARTICLES_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/wikipedia_articles.txt"

def __get_wiki_articles(file_name: str = WIKI_ARTICLES_FILE) -> list[str]:
    """
    Reads a list of Wikipedia article titles from a text file.

    Args:
        file_name (str, optional): The path to the text file. Defaults to WIKI_ARTICLES_FILE.

    Returns:
        list[str]: A list of Wikipedia article titles.
    """
    wiki_articles_title = []
    with open(file_name, "r") as file:
        for file_line in file:
            if not file_line.startswith("#"):
                title_str = file_line.strip()
                if title_str:
                    log.info(f"Reading file {file_name}, line: {title_str}")
                    wiki_articles_title.append(title_str)
    log.info(f"Loaded {len(wiki_articles_title)} Wikipedia articles titles")
    return wiki_articles_title

def __build_wiki_json_graph(
    title: str, 
    max_nodes: int = DEF_MAX_NODES, 
    folder_path: str = FOLDER_PATH
) -> None:
    """
    Builds a single Wikipedia graph in JSON format and saves it to a file.

    Args:
        title (str): The title of the root article.
        max_nodes (int, optional): The maximum number of nodes in the graph. Defaults to DEF_MAX_NODES.
        folder_path (str, optional): The folder to save the JSON file. Defaults to FOLDER_PATH.
    """
    log.info(f"Processing article {title}")
    file_name = title.replace(" ", "_")
    if os.path.isfile(f"{folder_path}/{file_name}.json"):
        log.info(f"File {file_name}.json already exists")
        return
    
    try:
        graph = extract_graph(root_article_title=title, max_nodes=max_nodes)
        json_graph = json_dump(graph)
    except Exception as e:
        log.error(f"Error creating graph for '{title}'. Skipped: {e}")
        return
    
    with open(f"{folder_path}/{file_name}.json", "w") as file:    
        file.write(json_graph)
    log.info(f"File {file_name}.json created")

def create_dataset(
    max_nodes: int = DEF_MAX_NODES, 
    folder_path: str = FOLDER_PATH,
    wiki_file_path: str = WIKI_ARTICLES_FILE
) -> None:
    """
    Orchestrates the creation of the entire dataset using multiple threads.

    Args:
        max_nodes (int, optional): The maximum number of nodes per graph. Defaults to DEF_MAX_NODES.
        folder_path (str, optional): The folder to save the JSON files. Defaults to FOLDER_PATH.
        wiki_file_path (str, optional): The path to the file with Wikipedia article titles. Defaults to WIKI_ARTICLES_FILE.
    """
    wiki_articles = __get_wiki_articles(file_name=wiki_file_path)
    
    with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_list = [executor.submit(__build_wiki_json_graph, title, max_nodes, folder_path) for title in wiki_articles]
        
        log.info("Awaiting all workers to complete")
        for future in futures.as_completed(future_list):
            future.result() # Block and wait for completion
            
    log.info("Execution completed")

if __name__ == "__main__":
    # This block initiates the dataset creation process.
    create_dataset()

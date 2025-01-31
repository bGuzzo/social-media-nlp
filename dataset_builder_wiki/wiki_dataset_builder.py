import multiprocessing
import os, sys, os.path
import logging
import concurrent.futures as futures
from wiki_crowler_bfs import extract_graph
from wiki_crowler_bfs import json_dump

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

log.info(f"CPU cores available {multiprocessing.cpu_count()}")

DEF_MAX_NODES = 5000

FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"
WIKI_ARTICLES_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/wikipedia_articles.txt" 

def __get_wiki_articles(file_name:str = WIKI_ARTICLES_FILE) -> list[str]:
    wiki_articles_title = []
    
    with open(file_name, "r") as file:
        for file_line in file:
            if not file_line.startswith("#"):
                title_str:str = file_line.strip()
                log.info(f"Reading file {file_name}, line: {title_str}")
                if title_str:
                    wiki_articles_title.append(title_str)
            
    log.info(f"Loaded {len(wiki_articles_title)} Wikipedia articles titles")
    return wiki_articles_title


def __build_wiki_json_graph(
    title:str, 
    max_nodes: int = DEF_MAX_NODES, 
    folder_path:str = FOLDER_PATH
) -> None:
    
    log.info(f"Processing article {title}")
    file_name = title.replace(" ", "_")
    if os.path.isfile(f"{folder_path}/{file_name}.json"):
        log.info(f"File {file_name}.json already exists")
        return
    
    try:
        graph = extract_graph(root_article_title=title, max_nodes=max_nodes)
        json_graph = json_dump(graph)
    except Exception:
        log.error(f"Error creating graph for '{title}'. Skipped.")
        return
    
    with open(f"{folder_path}/{file_name}.json", "w") as file:    
        file.write(json_graph)
        file.close()
        log.info(f"File {file_name}.json created")


def create_dataset(
    # wiki_articles: list[str] = DEF_WIKIPEDIA_ARTICLES, 
    max_nodes: int = DEF_MAX_NODES, 
    folder_path: str = FOLDER_PATH,
    wiki_file_path: str = WIKI_ARTICLES_FILE
) -> None:
    # Load wikipedia articles title
    wiki_articles: list[str] = __get_wiki_articles(file_name=wiki_file_path)
    
    # Thread list
    future_list: list[futures.Future] = []
    
    with futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for title in wiki_articles:
            log.info(f"Start async building graph for article {title}")
            future = executor.submit(__build_wiki_json_graph, title, max_nodes, folder_path)
            future_list.append(future)
            
    log.info("Await all workers to complete")
    for future in futures.as_completed(future_list):
        # Blocking wait
        future.result()
    log.info("Execution completed")


if __name__ == "__main__":
    create_dataset()
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

DEF_WIKIPEDIA_ARTICLES = [
    
    # Sustinability
    "Sustainable development",
    "Environmental sustainability",
    "Social sustainability",
    "Economic sustainability",
    "Sustainable agriculture",
    "Sustainable energy",
    "Sustainable transport",
    "Sustainable cities",
    "Sustainable consumption",
    "Sustainable production",
    "Circular economy",
    "Green building",
    "Renewable energy",
    "Climate change mitigation",
    "Biodiversity conservation",
    "Water conservation",
    "Waste management",
    "Environmental ethics",
    "Corporate social responsibility",
    "Sustainable development goals",
    
    # Poverty
    "Poverty",
    "Poverty threshold",
    "Poverty in the United States",
    "Cycle of poverty",
    "Feminization of poverty",
    "Extreme poverty",
    "Child poverty",
    "Poverty reduction",
    "List of countries by percentage of population living in poverty",
    "Causes of poverty",
    "Effects of poverty",
    "Poverty and health",
    "Poverty and education",
    "Poverty and crime",
    "Social safety net",
    "Welfare",
    "Food security",
    "Homelessness",
    "Millennium Development Goals",
    
    # Global warming
    "Climate change",
    "Global warming",
    "Effects of climate change",
    "Climate change mitigation",
    "Climate change adaptation",
    "Causes of climate change",
    "Greenhouse gas",
    "Carbon dioxide in Earth's atmosphere",
    "Methane",
    "Nitrous oxide",
    "Deforestation",
    "Fossil fuel",
    "Renewable energy",
    "Solar energy",
    "Wind power",
    "Hydropower",
    "Geothermal energy",
    "Biomass",
    "Nuclear power",
    "Climate change denial",
    
    # Food secutity
    "Food security",
    "Food sovereignty",
    "Global hunger",
    "Malnutrition",
    "Famine",
    "Right to food",
    "Food systems",
    "Sustainable agriculture",
    "Food waste",
    "Food desert",
    "Food bank",
    "World Food Programme",
    "Food and Agriculture Organization",
    "International Fund for Agricultural Development",
    "Global Food Security Index",
    "Food crisis",
    "Urban agriculture",
    "Community garden",
    "Vertical farming",
    "Genetically modified food controversies",
    
    # Health
    "Health",
    "Public health",
    "Global health",
    "Mental health",
    "Physical health",
    "Nutrition",
    "Disease",
    "Infectious disease",
    "Non-communicable disease",
    "Health care",
    "Universal health care",
    "Health economics",
    "Environmental health",
    "Occupational safety and health",
    "Epidemiology",
    "Biostatistics",
    "Health promotion",
    "Preventive healthcare",
    "Traditional medicine",
    "Telehealth"
]

FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"

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
    wiki_articles: list[str] = DEF_WIKIPEDIA_ARTICLES, 
    max_nodes: int = DEF_MAX_NODES, 
    folder_path:str = FOLDER_PATH
) -> None:
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
# Import parent script
import os, sys, os.path
import logging
from wiki_crowler_bfs import extract_graph
from wiki_crowler_bfs import json_dump

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

MAX_NODES = 2000

WIKIPEDIA_ARTICLES = [
    
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


if __name__ == "__main__":
    for title in WIKIPEDIA_ARTICLES:
        log.info(f"Processing article {title}")
        
        file_name = title.replace(" ", "_")
        if os.path.isfile(f"{FOLDER_PATH}/{file_name}.json"):
            log.info(f"File {file_name}.json already exists")
            continue
        
        try:
            graph = extract_graph(root_article_title=title, max_nodes=MAX_NODES)
            json_graph = json_dump(graph)
        except Exception as e:
            log.error(f"Error: Article '{title}' not found. Skipped.", e)
            continue
        
        with open(f"{FOLDER_PATH}/{file_name}.json", "w") as file:    
            file.write(json_graph)
            file.close()
            log.info(f"File {file_name}.json created")
import gc
import logging
import os
import re
import nltk
import PyPDF2
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download nltk
nltk.download("stopwords")
nltk.download("wordnet")

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def clean_text(text: str) -> str:
    """
    Cleans text data for training an LLM.

    Args:
        text: The input text string.

    Returns:
        A cleaned text string.
    """

    logger.info(f"Cleaning text, initial lenght {len(text)}")

    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove extra whitespace
    # text = " ".join(text.split())

    # Remove newlines and extra whitespace
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r" +", " ", text)

    # Remove any non-alphanumeric characters except for basic punctuation
    # (period, comma, question mark, exclamation mark, and hyphen)
    text = re.sub(r"[^a-zA-Z0-9.,?!'-]+", " ", text)

    # Remove double punctuation symbols
    # cleaned_text = re.sub(r'([.,!?;:])\1+', r'\1', cleaned_text)
    text = re.sub(r"([.,!?;:])\1+", r"\1", text)

    # Tokenize the text
    tokens = text.split()

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    cleaned_text = " ".join(tokens)

    logger.info(f"Cleaning text, final lenght {len(cleaned_text)}")
    return cleaned_text


def convert_pdf_to_txt(pdf_folder, txt_folder):
    """
    Converts all PDF documents in a folder to plain text files.

    Args:
        pdf_folder: The path to the folder containing the PDF documents.
        txt_folder: The path to the folder where the text files will be saved.
    """

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            logger.info(f"Processing file: {filename}")
            text_content = []
            pdf_path = os.path.join(pdf_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(txt_folder, txt_filename)

            with (
                open(pdf_path, "rb") as pdf_file,
                open(txt_path, "w", encoding="utf-8") as txt_file,
            ):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Clean the text (example cleaning steps)
                    text = text.replace("\n", " ")  # Remove newline characters
                    text = text.replace("\t", " ")  # Remove tab characters
                    text = text.replace("\r", " ")  # Remove carriage return characters
                    
                    # Replace multiple spaces with single space
                    text = re.sub(r"\s+", " ", text)

                    logger.info(
                        f"Page {page_num}/{len(pdf_reader.pages)} of {filename} successfully parsed"
                    )

                    # Write the cleaned text to the text file
                    text_content.append(text)

                full_text = " ".join(text_content)

                # Clean & pre-process text
                full_text = clean_text(text=full_text)

                txt_file.write(full_text)
                logger.info(
                    f"File {filename} successfully saved as plain text in {txt_filename}"
                )


if __name__ == "__main__":
    pdf_folder = "/home/bruno/Documents/GitHub/social-media-nlp/training/dataset_pdf"  # Replace with the actual path
    txt_folder = "/home/bruno/Documents/GitHub/social-media-nlp/training/dataset_txt"  # Replace with the actual path
    convert_pdf_to_txt(pdf_folder, txt_folder)

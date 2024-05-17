# use llamaparse to parse the policy

# Import necessary libraries
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document

# Load environment variables
load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")


def parse_policy(pdf_file_path: str, parsing_instruction: str = "") -> str:
    # Initialize the LlamaParse with the API key
    parser = LlamaParse(result_type="markdown", parsing_instruction=parsing_instruction)

    # Parse the PDF and get the output in markdown format
    file_extractor = {".pdf": parser}
    parsed_markdown = SimpleDirectoryReader(
        input_files=[pdf_file_path],
        file_extractor=file_extractor,
    ).load_data()
    return parsed_markdown


def save_markdown(parsed_document: Document, file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        for line in parsed_document:
            f.write(line.text + "\n")


if __name__ == "__main__":
    # PARSING_INSTRUCTION = """You will be given an insurance policy or a policy certificate.
    # There may be tables with important insurance data. Make sure to parse everything."""
    PDF_FILE_PATH = "./data/policies/57876675-Cargo.pdf"
    markdown = parse_policy(PDF_FILE_PATH)
    save_markdown(markdown, "parsed_policy.md")

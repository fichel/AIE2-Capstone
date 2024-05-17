import os
import glob
from .parse_pdf import parse_policy, save_markdown


def parse_all_policies(source_folder: str, target_folder: str):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List all PDF files in the source folder
    pdf_files = glob.glob(os.path.join(source_folder, "*.pdf"))

    # Sort files for consistent numbering
    pdf_files.sort()

    # Parse each PDF file and save the output
    for index, pdf_file in enumerate(pdf_files):
        # Parse the PDF file to markdown
        markdown_content = parse_policy(pdf_file)

        # Define the output filename
        output_filename = f"policy{index + 1:03d}.md"
        output_filepath = os.path.join(target_folder, output_filename)

        # Save the markdown content to the file
        save_markdown(markdown_content, output_filepath)
        print(f"Processed {pdf_file} -> {output_filepath}")


if __name__ == "__main__":
    SOURCE_FOLDER = "./data/policies"
    TARGET_FOLDER = "./data/finetuning/inputs"
    parse_all_policies(SOURCE_FOLDER, TARGET_FOLDER)

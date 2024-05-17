import os

INPUTS_DIR = "./data/finetuning/inputs"
OUTPUTS_DIR = "./data/finetuning/outputs"


def find_missing_outputs():
    missing_outputs = []

    for input_file in os.listdir(INPUTS_DIR):
        if input_file.endswith(".md"):
            base_name = input_file.split(".")[0]
            simplified_name = f"simplified-{base_name}.md"
            simplified_path = os.path.join(OUTPUTS_DIR, simplified_name)

            if not os.path.exists(simplified_path):
                missing_outputs.append(input_file)

    return missing_outputs


if __name__ == "__main__":
    missing = find_missing_outputs()
    print("Policies missing simplified counterparts:")
    for policy in missing:
        print(policy)

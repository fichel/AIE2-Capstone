import os
from synthetic_policy import (
    read_policy,
    generate_synthetic_policy,
    save_policy,
    SIMPLIFIED_POLICY_PROMPT_V2,
)

INPUTS_DIR = "./data/finetuning/inputs"
# OUTPUTS_DIR = "./data/finetuning/outputs"
OUTPUTS_DIR = "./data/finetuning/outputs3"


def generate_output(policy_path: str, output_path: str) -> None:
    # get policy name
    base_policy = read_policy(policy_path)
    base_policy_name = os.path.basename(policy_path).split(".")[0]

    # generate the simplified policy
    simplified_policy = generate_synthetic_policy(
        base_policy, SIMPLIFIED_POLICY_PROMPT_V2
    )

    # save the output
    simplified_policy_name = f"simplified-{base_policy_name}.md"
    simplified_policy_path = os.path.join(output_path, simplified_policy_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_policy(simplified_policy, simplified_policy_path)


if __name__ == "__main__":
    # for file_name in os.listdir(INPUTS_DIR):
    #     if file_name.endswith(".md"):
    #         base_policy_path = os.path.join(INPUTS_DIR, file_name)
    #         generate_output(base_policy_path, OUTPUTS_DIR)

    # test
    BASE_POLICY_PATH = "./data/finetuning/inputs/policy001.md"
    generate_output(BASE_POLICY_PATH, OUTPUTS_DIR)

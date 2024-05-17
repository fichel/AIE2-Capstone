import os
from synthetic_policy import (
    read_policy,
    generate_synthetic_policy,
    save_policy,
    SYNTHETIC_POLICY_PROMPT,
)

INPUTS_DIR = "./data/finetuning/inputs"


def generate_synthetic_policies(base_policy_path: str, num_policies: int = 5) -> None:
    base_policy = read_policy(base_policy_path)
    base_policy_name = os.path.basename(base_policy_path).split(".")[0]

    for i in range(1, num_policies + 1):
        synthetic_policy = generate_synthetic_policy(
            base_policy, SYNTHETIC_POLICY_PROMPT
        )
        synthetic_policy_name = f"{base_policy_name}-{i}.md"
        synthetic_policy_path = os.path.join(INPUTS_DIR, synthetic_policy_name)
        save_policy(synthetic_policy, synthetic_policy_path)


if __name__ == "__main__":
    for file_name in os.listdir(INPUTS_DIR):
        if file_name.endswith(".md") and "-" not in file_name:
            policy_path = os.path.join(INPUTS_DIR, file_name)
            generate_synthetic_policies(policy_path)

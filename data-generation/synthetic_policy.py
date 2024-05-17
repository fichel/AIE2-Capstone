from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SIMPLIFIED_POLICY_PROMPT = """
You are given an original policy and a simplifed policy template for a new policy. Your goal is to translate the policy in the format of the template.
- Modify the information in between square brackets '[]' on the template to match the policy.
- The information in between '<>' is optional.
- Do not add or change any items except for the ones that are already in the template.

Original Policy Text:
{policy}

Simplified Policy Template:
Congratulations! You got yourself a [Insurance Type] Policy for your [Insured object, ex: Property Type, Vehicle, etc] at [Address, if it makes sense]. Your policy number is [Policy Number].

We want to make sure you know what you're getting for your [Premium Amount] per [Premium Frequency], so we did our best to make this policy short and sweet.

Please take a few minutes to read through, and let us know if you have any questions. You can always change coverages, add valuable items, and more.

Who's covered?
This policy covers [Covered Individuals, property or object].

When?
This policy covers events that started after [Start Date], and before [End Date].

Against what?
We protect you against [Covered Perils]. There are important limitations, though, so please read on.

For how much?
We provide coverage up to a certain limit. Here is a quick overview of the limits you chose (and can change):

[Itemized Coverage Limits]

These amounts indicate the maximum we will reimburse you, in total, per year - even if the losses you suffer are larger.

So take a minute to check your policy [Hypothetical Scenario to Consider Coverage Adequacy].

If [Total Coverage Limit] isn't enough to cover everything, please increase your total coverage. <And if you own any valuable items worth more than [Per-Item Limit], be sure to add them to your policy so they're covered for their full amount.>

Used or new?
[Explanation of Replacement Cost Coverage]
"""

SIMPLIFIED_POLICY_PROMPT_V2 = """
You are given an original policy and a simplifed policy template for a new policy. Your goal is to translate the policy in the format of the template.
- Modify the information in between square brackets '[]' on the template to match the policy.
- The information in between '<>' is optional.
- If the premium amount is not present, just reference 'what you're getting for your premium'.
- If the total coverage limit is not present, just reference 'your coverage limits'.
- Do not add or change any items except for the ones that are already in the template.

Original Policy Text:
{policy}

Simplified Policy Template:
Congratulations! You got yourself a [Insurance Type] Policy for your [Insured object, ex: Property Type, Vehicle, etc] at [Address, if it makes sense]. Your policy number is **[Policy Number]**.

We want to make sure you know what you're getting for your **[Premium Amount] per [Premium Frequency]**, so we did our best to make this policy short and sweet.

Please take a few minutes to read through, and let us know if you have any questions. You can always change coverages, <add valuable items>, and more.

## Who's covered?
This policy covers **[Covered Individuals, property or object. If it's a vehicle, ]**.

## When?
This policy covers events that started after [Start Date], and before [End Date].

## Against what?
We protect you against [Covered Perils]. There are important limitations, though, so please read on.

## For how much?
We provide coverage up to a certain limit. Here is a quick overview of the limits you chose (and can change):

[Itemized Coverage Limits]

These amounts indicate the maximum we will reimburse you, in total, per year - even if the losses you suffer are larger.

So take a minute to check your policy [Hypothetical Scenario to Consider Coverage Adequacy].

If [Total Coverage Limit] isn't enough to cover everything, please increase your total coverage. <And if you own any valuable items worth more than [Per-Item Limit], be sure to add them to your policy so they're covered for their full amount.>

## Used or new?
[Explanation of Replacement Cost Coverage]
"""

SYNTHETIC_POLICY_PROMPT = """
Base Policy:
{policy}

Given the base policy above, generate a synthetic policy that is similar to the base policy.
Modify the following information, by generating variants of the original data for the insurer, the object being insured and dolar amounts:
- Policy number
- Insured name and address, including city and state
- Policy period start and end dates
- Producer/agent name and contact details
- Vehicle details like make, model, year, VIN, registration
- Property address and description
- Coverages, limits and deductibles
- Additional coverages or endorsements
- Premium amount

If any of the properties above don't appear in the policy, just skip them.
Do not attempt to modidy the insurance type, the name of the insurance carrier or any of the text related to the carrier.
Also avoid changing general text, like: ... THE POLICIES DESCRIBED HEREIN IS SUBJECT TO ALL THE TERMS ...
Do not add or subtract anything, only substitute the data above for other similar data.

Output your results following the same exact format of the Base Policy.
"""


def read_policy(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def save_policy(policy_content: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(policy_content)


def generate_synthetic_policy(base_policy: str, prompt: str) -> str:
    """
    Generate a synthetic policy based on the base policy.
    """
    synthetic_policy_prompt = ChatPromptTemplate.from_template(prompt)
    chat_model = ChatOpenAI(model="gpt-4", temperature=0.7)
    policy_chain = synthetic_policy_prompt | chat_model | StrOutputParser()
    return policy_chain.invoke({"policy": base_policy})


if __name__ == "__main__":
    POLICY_PATH = "./data/finetuning/inputs/policy001.md"
    OUTPUT_PATH = "./data/finetuning/inputs/policy001-2.md"
    policy = read_policy(POLICY_PATH)
    synthetic_policy = generate_synthetic_policy(policy, SYNTHETIC_POLICY_PROMPT)
    save_policy(synthetic_policy, OUTPUT_PATH)

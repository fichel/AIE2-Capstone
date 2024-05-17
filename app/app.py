# streamlit app that generates a clearPolicy certificate

import os
from tempfile import NamedTemporaryFile
import streamlit as st
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.language_models import LLM
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import pipeline, AutoTokenizer

# Load environment variables
load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# Load prompt template
messages = [
    {
        "role": "system",
        "content": """Please translate the following insurance policy into a simplified format, in markdown.
        Don't say anything else, just present the policy in a clear and easy-to-read format.
        """,
    },
]

POLICY_TEMPLATE = """\
[INSURANCE_POLICY]{INPUT}[END_INSURANCE_POLICY]
"""


@st.cache_resource
def load_model() -> LLM:
    """Load the HF Inference Endpoint"""
    llm = HuggingFaceEndpoint(
        endpoint_url="https://yq8wk1ab6ed6554i.us-east-1.aws.endpoints.huggingface.cloud",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        streaming=False,
        huggingfacehub_api_token=HF_TOKEN,
    )

    return llm


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


st.title("🔎 clearPolicy")
st.subheader("Generate a clearPolicy certificate for your policy")
st.divider()
file = st.file_uploader(label="Upload your policy certificate", type="pdf")
col1, col2 = st.columns(2)

if file:
    # load the model and cache it
    llm = load_model()
    clearpolicy = ChatHuggingFace(llm=llm)

    # display the policy
    with col1:
        pdf_viewer(file.getvalue())

    # parse the policy
    with NamedTemporaryFile(suffix=".pdf") as f:
        f.write(file.getvalue())
        f.flush()
        parsed_policy = parse_policy(f.name)
        with col1:
            st.write(parsed_policy[0])

    # generate the clearPolicy certificate
    policy_prompt = POLICY_TEMPLATE.format(INPUT=parsed_policy[0].get_text())
    messages += [
        {"role": "user", "content": policy_prompt},
    ]
    response = clearpolicy.invoke(messages)

    # display the clearPolicy certificate
    with col2:
        with st.container(border=True):
            st.markdown(response.content)

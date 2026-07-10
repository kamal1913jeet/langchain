import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_API_TOKEN")
if not hf_token:
    raise RuntimeError(
        "Hugging Face API token is not set. "
        "Please set HUGGINGFACEHUB_API_TOKEN or HF_API_TOKEN in your environment or .env."
    )

repo_id = os.environ.get("HF_REPO_ID", "gpt2")
task = os.environ.get("HF_TASK", "text-generation")
provider = os.environ.get("HF_PROVIDER") or None

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task=task,
    provider=provider,
    huggingfacehub_api_token=hf_token,
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("who is father of AI")
print(result.content)
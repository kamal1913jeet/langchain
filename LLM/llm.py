import os
from dotenv import load_dotenv

load_dotenv()

USE_HF = os.environ.get("USE_HF", "true").strip().lower() in {"1", "true", "yes", "on"}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def make_huggingface_llm():
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
    return ChatHuggingFace(llm=llm)


def make_openai_llm():
    from langchain_openai import OpenAI

    model_name = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in your environment or .env to use OpenAI.")
    return OpenAI(model=model_name)


def main():
    if USE_HF or not OPENAI_API_KEY:
        print("Using Hugging Face model. Set USE_HF=false and OPENAI_API_KEY to use OpenAI.")
        llm = make_huggingface_llm()
    else:
        print("Using OpenAI model.")
        llm = make_openai_llm()

    prompt = "What is the capital of India?"

    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        print("LLM invocation failed:", exc)
        return

    output = getattr(result, "content", result)
    print(output)


if __name__ == "__main__":
    main()
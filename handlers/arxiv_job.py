import requests
import arxiv
import os

import openai

from llama_index import ServiceContext

from llama_index.llms import OpenAI

from llama_index.indices.struct_store import JSONQueryEngine

openai.api_key = os.environ["OPENAI_API_KEY"]
DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_URL"]

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")

service_context = ServiceContext.from_defaults(
    llm=chatgpt, chunk_size=1024, embed_model=None
)

json_schema = {
    "title": "ArxivPaper",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "pdf_url": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "pdf_url", "summary"],
}


def fetch_arxiv_papers():
    search_query = f"abs:retrieval augmentaton"

    search = arxiv.Search(
        query=search_query, max_results=3, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []

    for result in arxiv.Client().results(search):
        papers.append(
            {
                "title": result.title,
                "pdf_url": result.pdf_url,
                "summary": result.summary,
            }
        )

    return papers


def generate_query(paper):
    return (
        f"Summarize and ensure the output to be: # {paper['title']} \n ### Summary: \n <summary_here> \n ### PDF: \n {paper['pdf_url']}'"
        "dont use special characters and make sure to use the correct spacing."
        "Break the summary in bullet lists, paragraphs using \n\n to better summarization and understanding."
    )

def lambda_handler():
    papers = fetch_arxiv_papers()

    for paper in papers:
        nl_query_engine = JSONQueryEngine(
            json_value=paper,
            json_schema=json_schema,
            service_context=service_context,
        )

        response = nl_query_engine.query(generate_query(paper))

        data = {"content": str(response)}

        request_response = requests.post(DISCORD_WEBHOOK, json=data)

        if request_response.status_code == 204:
            print("Successfully sent to discord")


if __name__ == "__main__":
    lambda_handler()

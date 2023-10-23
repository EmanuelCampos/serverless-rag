import requests
import datetime
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
        "title": {
            "type": "string",
            "description": "The title of the paper",
        },
        "pdf_url": {
            "type": "string",
            "description": "The url to the pdf of the paper",
        },
        "summary": {
            "type": "string",
            "description": "The summary of the paper",
        },
    },
    "required": ["title", "pdf_url", "summary"],
}


def fetch_arxiv_papers():
    today_date = datetime.date.today().strftime('%Y%m%d')
    last_week_date = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y%m%d')
    search_query = f"abs:retrieval augmentation AND submittedDate:[{last_week_date}0000 TO {today_date}2359]"

    search = arxiv.Search(
        query=search_query, max_results=50, sort_by=arxiv.SortCriterion.SubmittedDate
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

    print(papers)

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
            verbose=True
        )

        response = nl_query_engine.query(generate_query(paper))

        data = {"content": str(response)}

        request_response = requests.post(DISCORD_WEBHOOK, json=data)

        if request_response.status_code == 204:
            print("Successfully sent to discord")


if __name__ == "__main__":
    lambda_handler()

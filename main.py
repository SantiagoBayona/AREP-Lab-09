import os

import pinecone
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores.pinecone import Pinecone

openai_api_key = "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"

embeddings = OpenAIEmbeddings(api_key=openai_api_key)


@tool("SayHello", return_direct=True)
def say_hello(name: str) -> str:
    """
Answer when someone says hello
    :param name:username
    :return:Hello message
    """
    return f"Hello {name}!"


@tool("document_search_tool", return_direct=True)
def document_search_tool(query: str) -> str:
    """
Answer when someone asks about engineering
    :param query:
    :return:
    """
    docsearch = Pinecone.from_existing_index("taller9", embeddings)
    return docsearch.similarity_search(query)[0].page_content


def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_document():
    documents_data = []

    for document in os.listdir("documentos/"):
        if document.endswith('.txt'):
            file_path = os.path.join("documentos/", document)
            text = load_text_from_file(file_path)
            documents_data.append((document, text))

    for doc_name, doc_text in documents_data:
        Pinecone.from_texts(texts=[doc_text], embedding=embeddings, index_name="taller9")


def main():
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    llm = ChatOpenAI(temperature=0)
    tools = [
        say_hello,
        document_search_tool
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    process_document()
    print(agent.run("Hello! My name is Santiago"))
    print(agent.run("ingenieria electrica"))
    query = input("Input query: ")
    while query != "adiós":
        print(agent.run(query))
        query = input("Input query: ")


if __name__ == '__main__':
    main()

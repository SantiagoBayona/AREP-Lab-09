import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone


def main():
    pinecone.init(api_key="485409cd-40df-4fc6-b1cf-a8234af2b8b8", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    text = open("documentos/economia.txt", "r")
    pinecone.create_index("taller9", dimension=1536, metric="cosine")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='taller9')
    text = open("documentos/ingenieria-civil.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='taller9')
    text = open("documentos/ingenieria-sistemas.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='taller9')
    text = open("documentos/ingenieria-electrica.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='taller9')
    text = open("documentos/ingenieria-industrial.txt", "r")
    data = Pinecone.from_texts(texts=text.readlines(), embedding=embeddings, index_name='taller9')


def buscar():
    pinecone.init(api_key="485409cd-40df-4fc6-b1cf-a8234af2b8b8", environment="gcp-starter")
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index("taller9", embeddings)
    query = "Cuantos años de acreditación tiene ingeniería de industrial?"
    docs = docsearch.similarity_search(query)
    print(docs)


if __name__ == '__main__':
    main()

import os
import hashlib
import nltk
import pinecone
import requests
import mimetypes
import numpy as np
import glob
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TextSplitter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

load_dotenv()

# Get the Variables from the .env file
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

class PineconeManager:
    def __init__(self, api_key, environment, index_name):
        pinecone.init(
            api_key=api_key,
            environment=environment
        )
        self.index_name = index_name

    def list_indexes(self):
        return pinecone.list_indexes()

    def index_exists(self):
        active_indexes = self.list_indexes()
        return self.index_name in active_indexes

    def create_index(self, dimension, metric):
        if not self.index_exists():
            pinecone.create_index(name=self.index_name, dimension=dimension, metric=metric)

    def get_index(self):
        return pinecone.Index(index_name=self.index_name)

    def deinit(self):
        pinecone.deinit()

class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed_url = urlsplit(url)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)

    @staticmethod
    def extract_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(url, href)
                if URLHandler.is_valid_url(absolute_url):
                    links.append(absolute_url)

        return links

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = []

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.extend(links)

        return all_links
    
class Document:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}
        self.id = hashlib.sha256(self.content.encode()).hexdigest()

class SentenceAwareTextSplitter(TextSplitter):
    def __init__(self, chunk_size, add_start_index=False):
        super().__init__(chunk_size)
        self._add_start_index = add_start_index

    def split_text(self, text):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self._chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = ''
            current_chunk += ' ' + sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

class DocumentLoaderFactory:
    @staticmethod
    def get_loader(file_path_or_url):
        if file_path_or_url.endswith(".txt"):
            return UnstructuredTextLoader(file_path_or_url)
        elif file_path_or_url.endswith(".pdf"):
            return PdfDocumentLoader(file_path_or_url)
        elif file_path_or_url.endswith(".docx"):
            return WordDocumentLoader(file_path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {file_path_or_url}")
        
def load_and_split_documents(loader, text_splitter):
    raw_documents = loader.load_and_split(text_splitter=text_splitter)
    documents = []
    for i, raw_doc in enumerate(raw_documents):
        doc = Document(
            content=raw_doc.page_content,  # this is defined in the pdfparser library https://api.python.langchain.com/en/latest/_modules/langchain/document_loaders/parsers/pdf.html
            metadata={
                'source': loader.file_path,
                'chunk_number': i,
            },
        )
        documents.append(doc)
    return documents


def create_unique_id(doc):
    return hashlib.sha256(doc.content.encode()).hexdigest()

def load_documents_into_pinecone(pinecone_manager, embeddings, train):
    text_splitter = SentenceAwareTextSplitter(chunk_size=5000)

    file_paths = glob.glob("data/*.pdf")
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = load_and_split_documents(loader, text_splitter)

        ids_and_vectors = []
        for doc in tqdm(documents):
            # generate an embedding for the document
            embedded_content = embeddings.embed_documents([doc.content])[0]

            vector = {
                "id": doc.id,
                "values": embedded_content,
                "metadata": doc.metadata,
            }
            ids_and_vectors.append(vector)

        # add the documents to the Pinecone index
        pinecone_manager.get_index().upsert(ids_and_vectors)

        if train:
            print(f"Training the model with documents from {file_path}")
            if not pinecone_manager.index_exists():
                pinecone_manager.create_index(dimension=1536, metric="cosine")
            pinecone_manager.get_index().upsert(ids_and_vectors)
        else:
            print(f"Updating the model with documents from {file_path}")
            if pinecone_manager.index_exists():
                pinecone_manager.get_index().upsert(ids_and_vectors)

    return pinecone_manager.get_index()


def answer_questions(pinecone_index, embeddings, chat):
    messages = [
        SystemMessage(
            content='I want you to act as a document that I am having a conversation with. Your name is "AI '
                    'Assistant". You will provide me with answers from the given info from reference. If the answer '
                    'is not included, in reference say exactly "Hmm, I am not sure or give answer by yourself." and '
                    'stop after that. Refuse to answer any question not about the info in reference.')
    ]
    while True:
        question = input("Please enter a question (or 'quit' to stop): ")
        if question.lower() == 'quit':
            break

        query_vector = embeddings.embed_query(question)
        results = pinecone_index.query(queries=[query_vector], top_k=1)

        # Check if any results were returned
        if results.ids and results.ids[0]:
            # Assemble the document chunks
            results = sorted(results, key=lambda result: result.metadata['chunk_number'])
            document = "\n".join([result.values for result in results])
            main_content = question + "\n\nreference:\n"
            main_content += document + "\n\n"

            messages.append(HumanMessage(content=main_content))
            response = chat.generate_response([SystemMessage(content=question),
                                               AIMessage(content=main_content, role="assistant")])
            messages.pop()
            messages.append(HumanMessage(content=question))
            messages.append(AIMessage(content=response, role="assistant"))

            print(response)
        else:
            print("I'm sorry, but I couldn't find a relevant answer to your question.")

def main():
    # initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # initiatlize a chat model
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # create a PineconeManager instance
    pinecone_manager = PineconeManager(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME)

    # create the index if it doesn't already exist
    if not pinecone_manager.index_exists():
        pinecone_manager.create_index(dimension=1536, metric="cosine")

    # check if the user wants to train the model
    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    if train == 1:
        print("Updating the model with documents from data/test.pdf")
        # load or update documents in the Pinecone index
        load_documents_into_pinecone(pinecone_manager, embeddings, train)

    # start the question-answering loop
    answer_questions(pinecone_manager.get_index(), embeddings, chat)

    # deinitialize the index and client after you're done
    pinecone_manager.get_index().deinit()
    pinecone_manager.deinit()

if __name__ == "__main__":
    main()
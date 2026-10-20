import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text:str) -> str: 

    text = re.sub(r"\s+"," ",text)
                                     
    text = text.strip()  

    return text


class Textingestion:

    def __init__(self, chunk_size = 1000, chunk_overlap = 250):

        self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                       chunk_overlap = chunk_overlap,
                                                       separators=['\n\n','\n','.',' ',''])

    def clean(self, text:str) -> str:
        return clean_text(text)


    def process(self, path:str):

        docs = TextLoader(path,encoding = 'utf-8').load()

        for d in docs:
            d.page_content= self.clean(d.page_content)

        
        return self.splitter.split_documents (docs)


# Exploration run (only when executed directly, never on import)
if __name__ == "__main__":
    loader = TextLoader('C:\\Projets_rag_personnels\\data\\pdf_files\\text_files\\Notebook summary for RAG.txt',encoding='utf-8')

    text = loader.load()

    print(text[0].page_content[:300],'\n')
    print(f'Metadata: {text[0].metadata.get("source")}')


    cleaned_docs = []

    for doc in text:
        doc.page_content = clean_text(doc.page_content)

        cleaned_docs.append(doc)



    splitter = RecursiveCharacterTextSplitter(chunk_size =1000,
                                              chunk_overlap = 250,
                                              separators=['\n\n','\n','.',' ',''])


    chunks = splitter.split_documents(cleaned_docs)

    print(len(chunks),'\n')

    for i, chunk in enumerate(chunks[:6]):

        print(f'CHUNK:{i+1}')
        print(chunk.page_content)
        print(f'Metadata : {chunk.metadata} \n')

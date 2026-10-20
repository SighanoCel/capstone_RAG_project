import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter







text_to_cancel = r"Projet à venir #\d+\s*Espace réservé pour un futur projet\.\s*Tech 1\s*Tech 2\s*Tech 3"


def clean_web_text(text:str) -> str:

    text= re.sub(text_to_cancel, "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+',' ',text) # espaces mulitples en un seul espace
    text = re.sub(r'\n+', '\n',text)  # saut de ligne multiples en un seul saut
    text = re.sub(r'(Cookies?|Privacy Policy).*?$','',text, flags = re.I) # suppression de certain
                                             # éléments qu'on retrouve sur les pages web. comme 
                                             # cookies, privacy policy etc. donc on laisse 
                                             # seulement le texte utile.

    

    return text.strip()



class Webingestion:

    def __init__(self, chunk_size = 500, chunk_overlap = 100):

        self.splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                       chunk_overlap = chunk_overlap,
                                                       separators=['\n\n','\n','.',' ',''])

    def clean(self, text:str) -> str:
        return clean_web_text(text)


    def process(self, url:str):

        docs = WebBaseLoader(url).load()

        for d in docs:
            d.page_content= self.clean(d.page_content)

        
        return self.splitter.split_documents (docs)
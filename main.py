from typing import Union
from flask import Flask, request, jsonify, render_template
import requests
import xml.etree.ElementTree as ET
import json
from sentence_transformers import SentenceTransformer,util
import faiss
import numpy as np
import pandas as pd
from Bio import Entrez
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import ollama
import chromadb

app = Flask(__name__)


@app.get("/")
def read_root():

    Entrez.email = 'ynakilan@gmail.com'
    topics = ['mRNA' , 'Gene Editing' , 'Gut' , 'microbiome' , 'organ regeneration' , 'Epigenetics' , 'Autoimmune' , 'Cancer' , 'Stem Cell' , 'Neurodegenerative' ,'Biopsies' , 'Organoids']
    date_range = '("2012/03/01"[Date - Create] : "2025/02/10"[Date - Create])'

    queries = []
    if topics:
        topic_queries = ['{}[Title/Abstract]'.format(topic) for topic in topics]
        queries.append('(' + ' OR '.join(topic_queries) + ')')

    full_query = ' AND '.join(queries) + ' AND ' + date_range

    handle = Entrez.esearch(db='pubmed', retmax=3, term=full_query)
    record = Entrez.read(handle)
    id_list = record['IdList']

    final_list = []
    for ids in id_list:
        handle = Entrez.efetch(db='pubmed', id=ids, rettype='xml')
        record = Entrez.read(handle)
    
        for things in record['PubmedArticle']:
            title = things['MedlineCitation']['Article']['ArticleTitle']
            abstract = things['MedlineCitation']['Article']['Abstract']['AbstractText'] if 'Abstract' in things['MedlineCitation']['Article'] and 'AbstractText' in things['MedlineCitation']['Article']['Abstract'] else ''
            url = f"https://www.ncbi.nlm.nih.gov/pubmed/{ids}"
            final_list.append({"title":title, "abstract":abstract , "url":url})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 200)

    abstract_list=[]
    for things in final_list:
        if(things['abstract'] == ''):
            continue
        chunks = text_splitter.split_text(things['abstract'][0])
    
        for chunk in chunks:
            abstract_list.append({"title":things['title'] , "abstract":chunk , "url":things['url']})

    chroma_client = chromadb.Client()
    collection_name = "biomedical_papers"
    collection = chroma_client.create_collection(collection_name)

    for chunk in abstract_list:
        abst = chunk['abstract']
        embed = ollama.embeddings(model="nomic-embed-text", prompt=abst)["embedding"]
        collection.add(ids=[str(hash(abst))], embeddings=[embed],documents=[abst])

    
    query_text = "Tell me about most frequent mutation among cancer cells."
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query_text)["embedding"]
    results = collection.query(query_embeddings=[query_embedding],n_results=1)
    print(results)

    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"I am giving you a set of papers {results} and based on the given papers answer to the query that the user is asking. You need to analyse what the user wants and check in the given information and make sure to give the exact paper where you are getting the information from. I want you not to go beyond what context I have given."
            }],
            max_tokens=150
        )

    return results



#     # This is to find all the ids wrt the search term
#     search_term='Machine Learning'
#     base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
#     url = f'{base_url}?db=pubmed&term={search_term}&retmode=json&retmax=100'

#     response = requests.get(url)
#     data = response.json()
#     pubmed_ids = data['esearchresult']['idlist']

#     model = SentenceTransformer('all-MiniLM-L6-v2')


#     # This is to find all the abstracts wrt the pubmed_ids found above
#     abstract_list=[]
#     for pubmed_id in pubmed_ids:
#         base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
#         url = f'{base_url}?db=pubmed&id={pubmed_id}&retmode=xml'
#         response = requests.get(url)

#         if response.status_code != 200 or not response.text.strip():
#             print(f"Error fetching PubMed ID {pubmed_id}: Empty or invalid response")
#             abstract_list.append({"abstract": "Error: No data found"})
#             continue

#         xml_data = response.text
#         root = ET.fromstring(xml_data)
#         abstract_elements = root.findall('.//AbstractText')

#         if abstract_elements:
#             abstract_texts = []
#             for abstract_element in abstract_elements:
#                 if abstract_element.text is None:
#                     continue  # Skip if text is missing
#                 abstract_texts.append(abstract_element.text.strip())

#             if abstract_texts:  # Only add if there's valid text
#                 abstract = '\n'.join(abstract_texts)
#                 embeddings = model.encode([abstract])
#                 abstract_list.append({"pubmed_id": pubmed_id, "abstract": abstract , "embeddings":embeddings[0].tolist()})
#         else:
#             continue

#     embeddings=[]
#     for paper in abstract_list:
#         if 'embeddings'in paper:
#             embeddings.append(paper['embeddings'])
#         else:
#             continue
#     embeddings = np.array(embeddings).astype('float32')
#     d = embeddings.shape[1]  # Embedding dimension
#     index = faiss.IndexFlatL2(d)
#     index.add(embeddings)

#     faiss.write_index(index, 'ml_papers.index')

#     def search_faiss(query, index, model, papers, top_k=5):
#         query_embedding = model.encode(query).astype('float32')
#         D, I = index.search(np.array([query_embedding]), top_k)  # Get top_k results

#         results = [papers[i] for i in I[0]]  # Fetch the best-matching papers
#         return results

# # Load FAISS index
#     index = faiss.read_index('ml_papers.index')

# # Example query
#     query = "Machine Learning in DNA"
#     results = search_faiss(query, index, model, abstract_list)

# # Print top results
#     for idx, res in enumerate(results):
#         print(f"Rank {idx+1}: {res['abstract'][:300]}...\n")

    # return abstract_list

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
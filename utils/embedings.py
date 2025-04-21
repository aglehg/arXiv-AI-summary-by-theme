import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.lib as lib 
import json
import google.generativeai as genai  
import google.genai.types as types
from google import genai
import pandas as pd
        


def embed_papers(client,papers):
    """
    Embed papers grouped by industry without modifying the original DataFrame.
    """
    
    collection = lib.get_chroma_collection(name="paper_abstracts")
    print(f"✅ Collection '{collection.name}' contains {collection.count()} items.")

    # Build a temporary expanded list
    expanded = []

    for _, row in papers.iterrows():
        industry_list = row.get('industry_list')

        # Not classified yet
        if not industry_list or not isinstance(industry_list, list):
            continue

        for industry_info in industry_list:
            industry = industry_info.get('industry')
            if industry and industry not in ('N/A', 'Other'):
                expanded.append({
                    "id": row['id'],
                    "title": row['title'],
                    "abstract": row['abstract'],
                    "industry": industry
                })

    if not expanded:
        print("⚠️ No papers with industries found to embed.")
        return

    papers_expanded = pd.DataFrame(expanded)

    industries = papers_expanded['industry'].unique()
    print(f"🧠 Found {len(industries)} industries to embed.")

    for industry in industries:
        industry_papers = papers_expanded[papers_expanded['industry'] == industry]
        print(f"\\n🛠️ Embedding {len(industry_papers)} papers for industry: {industry}") 
        batch_embed_and_store(client,industry_papers, collection,industry)


def batch_embed_and_store(client,papers, collection, industry_name, batch_size=10):
    """
    Processes (id, abstract) papers, embeds them in batches, and stores them in a Chroma collection,
    tagging each entry specifically with a single industry.

    Args:
        papers (List[Dict]): List of dicts with 'id' and 'abstract' fields.
        collection (chroma.Collection): Target Chroma collection to store embeddings.
        industry_name (str): The specific industry to assign in metadata.
        batch_size (int): How many papers to process per batch.
    """

    if papers.empty:
        print(f"⚠️ No papers to embed for industry {industry_name}.")
        return

    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]

    for batch in batches:
        ids = batch['id'].apply(lambda x: f"{x}__{industry_name}").tolist()
        arxiv_ids = batch['id'].tolist()
        texts = batch['id'].tolist()


        print(f"🚀 Embedding {len(batch)} papers for {industry_name}: {ids}")

        embeddings = embed_texts(client,texts)

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"id": id_, "industry": industry_name} for id_ in arxiv_ids],
            ids=ids
        )


def embed_texts(client,texts):
    """
    Embeds a list of texts using Gemini's text-embedding-004 model.

    Args:
        texts (List[str]): The list of texts to embed.

    Returns:
        List[List[float]]: A list of embedding vectors.
    """
    response = client.models.embed_content(
        model='models/text-embedding-004',
        contents=texts,
        config=types.EmbedContentConfig(task_type='semantic_similarity')
    )
    embeddings = [e.values for e in response.embeddings]
    # print (response.embeddings)
   
    return embeddings


# Use for resume on failure 
# Run just the classification task from a previously saved enriched json file -----------------------------------------------
if __name__ == "__main__": 
    import pandas as pd

    config = lib.load_config()
    client = genai.Client(api_key=config['GOOGLE_API_KEY']) 
    papers = pd.DataFrame(lib.load_enriched_papers())
    print("Loaded papers:", len(papers))
    
    print(papers.head(10))
    embed_papers(client,papers)
# -------------------------------------------------------------------------
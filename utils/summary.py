# utils/summary.py
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import hdbscan
import numpy as np
from utils import lib 

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances


### 1. Top-level entry

def get_summaries(papers, OUT_DIR, industries=[]):
    '''
      Get list of summaries { industry, sections: [title, summary] } 
    '''

    all_results = []

    # Load intermediate results if present 
    if(os.path.exists(os.path.join(OUT_DIR, "industry_sections.json"))):
        with open(os.path.join(OUT_DIR, "industry_sections.json"), 'r') as f:
            all_results = json.load(f)
        print("Loaded intermediate results from file")
        

    collection = lib.get_chroma_collection(name="paper_abstracts")

    exclude = ['N/A', 'Other']

    # Create summary for all industries 
    if(len(industries) == 0):
        industries = lib.get_industries(papers)

    print(f"🧠 Found {len(industries)} industries to summarize.", industries)

    for industry in industries:
        if industry in exclude:
            continue

        # already done this industry? 
        if any(entry['industry'] == industry for entry in all_results):
            print(f"✅ Already done industry {industry}, skipping.")
            continue
        
        print(f"\n\nSummarizing industry: {industry} ")
        industry_summary = get_sections(papers, industry, collection)

        all_results.append({
            "industry": industry,
            "sections": industry_summary
        })
 
        # Save the all_results to a file
        with open(os.path.join(OUT_DIR, f"industry_sections.json"), 'w') as f:
            json.dump(all_results, f, indent=4)


    return all_results


### 2. Per industry

def get_sections(industry_papers, industry_name, collection):
    print("get_sections for industry:", industry_name)

    if(len(industry_papers) < 10):
        clustered_papers = industry_papers
        clustered_papers['cluster'] = -1
    else:
        clustered_papers = get_clusters(industry_papers, industry_name, collection)

    num_clusters = clustered_papers['cluster'].nunique()
    print(f"🧠 Found {len(clustered_papers)} papers for industry: {industry_name} with {num_clusters} clusters")    

    sections = []
    # Turn the long data on industry to wide data by matching to the current industry  
    clustered_papers[['relevanceScore', 'summary']] = clustered_papers.apply(
        lambda row: pd.Series(extract_industry_info(row, industry_name)),
        axis=1
    )

    for cluster_id, group in clustered_papers.groupby('cluster'):
        # print(group.head(10))
        section_papers = group[['id', 'summary', 'title']].to_dict(orient='records')
        print(f"Cluster {cluster_id} has {len(group)} papers ", group['id'].tolist())
        
        # If it's a large cluster sort by "relevance" and take the top 50
        # while currently this is largely unreliable as we improve the reliability of this score the pipeline is already prepared to use it 
        if len(group) > 50: 
            group = group.sort_values(by='relevanceScore', ascending=False).head(50)
            print(f"Cluster {cluster_id} has more than 50 papers, taking top 50 by relevance")
              
        section = get_summaries_batch(group, industry_name)
        print(f"Title: {section['title']}")
        section["papers"] = section_papers


        # "Noise", not an assigned cluster =>  we define the title 
        if cluster_id == -1: 
            section["title"] = "General Outlook"

        sections.append(section)

    print("returning sections ", sections)
    return sections


### 2.1 Get Results from chroma, if group is smaller than 10 just return the papers

def get_clusters(industry_papers, industry_name, collection, min_cluster_size=3):
    """
    assign a cluster label to the industry papers
    Args:
        industry_papers (pd.DataFrame): DataFrame of papers in the industry.
        industry_name (str): Name of the industry.
        collection: Chroma collection to retrieve embeddings.
        min_cluster_size (int): Minimum size of clusters.
    """

    print("\n\nClustering papers for industry:", industry_name)
    # Retrieve embeddings for this industry
    chroma_papers = collection.get(
        where={"industry": industry_name},
        include=["embeddings","metadatas"]
    ) 


    df = pd.DataFrame(chroma_papers['metadatas']) 
    print(df.head(10))
    if df.empty:
        print("No papers found for industry:", industry_name)
        return pd.DataFrame()
        
    if len(df) < 10:
        df['cluster'] = -1 
        return df.merge(industry_papers, on='id', how='left')
    
    embeddings = np.array(chroma_papers['embeddings'])  # Make sure they are numpy array
    
    # Normalize embeddings - as in algebra, all vectors to length 1, so following operations will not be affected by length,  only direction 
    embeddings = normalize(embeddings)

    # Compute cosine distances
    cosine_distance_matrix = cosine_distances(embeddings)

    clusterer =hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric='precomputed',
        cluster_selection_method='eom'
    ) 
    df['cluster'] = clusterer.fit_predict(cosine_distance_matrix)
    return df.merge(pd.DataFrame(industry_papers), on='id', how='left')


### 2.2 Summarize a cluster (batching if large)
def get_summaries_batch(papers, industry_name, batch_size=10):
    if len(papers) <= batch_size:
        print("Small cluster, summarizing directly")
        # Small cluster, summarize directly
        texts = papers['abstract'].tolist()
        return get_cluster_summary(texts, industry_name)

    # Large cluster, split into batches
    batch_summaries = []
    for i in range(0, len(papers), batch_size):
        
        batch = papers.iloc[i:i+batch_size]
        batch_texts = batch['abstract'].tolist()
        batch_summary = get_cluster_summary(batch_texts, industry_name)
        print("Batch summarizing papers", i, "to", i+batch_size, " summary len ", len(batch_summary['summary']))
        batch_summaries.append(batch_summary)

    # Now summarize the batch summaries into one
    return summarize_summaries(batch_summaries, industry_name)


### 2.3 Summarize a set of papers

def get_cluster_summary(texts, industry_name):
    print("Summarizing batch of papers", len(texts), " papers")

    prompt = build_cluster_summary_prompt(texts, industry_name)
    print("get_cluster_summary prompt:", len(prompt))
    resp = lib.query_model(prompt,model_name='gemini-2.5-pro-preview-03-25')

    parsed = lib.extract_json_from_model_response(resp)
    return parsed  # Expected {"title": ..., "summary": ...}


### 2.4 Summarize summaries

def summarize_summaries(batch_summaries, industry_name):
    print("Summarizing batch summaries into final summary", len(batch_summaries), " summaries")
    texts = [bs['summary'] for bs in batch_summaries]

    prompt = build_summarize_summaries_prompt(texts, industry_name)
    print("get_summarize_summaries prompt:", prompt,"\n\n\n\n")
    resp = lib.query_model(prompt,model_name='gemini-2.5-pro-preview-03-25')

    parsed = lib.extract_json_from_model_response(resp)
    print("get_summarize_summaries response:", resp)
    return parsed  # {"title": ..., "summary": ...}

 
def build_cluster_summary_prompt(abstracts, industry):
    """
    return a prompt to summarize a cluster of papers

    Args:
        papers - dataframe of papers 
        industry - the industry name 
"""     
 
    abstracts = "\nAbstract:\n".join(abstracts)
    prompt=( 
            f"You are a research assistant compiling a thematic summary of AI research applied to the {industry} industry.\n\n"
            "Carefully read the following abstracts.\n"
            f"Summarize in no more than 200 words the **common contributions** these papers make to {industry}.\n"
            "Focus on shared themes, methods, or impacts — not on listing papers individually.\n\n"
            "Assign a title that reflects the main theme of the summary"
            f"{abstracts}"
            "return output as JSON ex format: { \"title\": \"the title of the summary\", \"summary\": \"the summary\"}"
    )
    return prompt 


def build_summarize_summaries_prompt(summaries, industry):
    """
    Summarizes a list of batch summaries into a final cluster summary and proposes a title (returns JSON).

    Args:
        summaries (List[str]): List of batch summaries.
        industry (str): Industry name for context.

    Returns:
        dict: { "title": str, "summary": str }
    """
    merged_summaries = "\n\n".join(f"Summary:\n{s}" for s in summaries)

    prompt = (
        f"You are compiling a final report of AI research contributions to the {industry} industry.\n\n"
        "Read the following summaries carefully.\n"
        "Write a final clean synthesis of the major shared themes, methods, and impacts.\n"
        "After the synthesis, propose a short, academic-sounding title (max 12 words).\n\n"
        "Return your output as a JSON object with two fields:\n"
        "  - \"title\": the final title\n"
        "  - \"summary\": the final clean synthesis text\n\n"
        "Example format:\n"
        "```json\n{\"title\": \"AI-Driven Advances in Healthcare Diagnostics\", \"summary\": \"This body of research explores...\"}\n```\n\n"
        f"{merged_summaries}"
    )
    return prompt

def extract_industry_info(row, target_industry):
    if isinstance(row['industry_list'], list):
        for item in row['industry_list']:
            if item.get('industry') == target_industry:
                return item.get('relevanceScore', None), item.get('summary', None)
    return None, None

# Use for resume on failure 
# Run just the classification task from a previously saved enriched json file -----------------------------------------------
if __name__ == "__main__": 
    import pandas as pd
    import google.generativeai as genai

    config = lib.load_config()
    genai.configure(api_key=config['GOOGLE_API_KEY'])
    papers = pd.DataFrame(lib.load_enriched_papers())
    print("Loaded papers:", len(papers))

    # Run each papers through the model so we can assign an industry label 
    get_summaries(papers,config['out_dir'])

#  / Run the classification task from a previously saved enriched json file -----------------------------------------------

             
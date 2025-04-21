import yaml
import re
import json
import time
import random
from chromadb import Client
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai  


def load_enriched_papers(path='out/papers_enriched.json'):
    """
    Loads enriched papers from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        List[Dict]: List of enriched papers.
    """
    with open(path, 'r') as f:
        papers = [json.loads(line) for line in f]
    return papers



def load_config(path='config.yaml'):
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise Exception(f"Config file {path} not found. Please copy config.sample.yaml and edit it.")

import os
import chromadb

def get_chroma_collection(name="paper_abstracts", base_path=None):
    """
    Creates or loads a ChromaDB collection safely for both local and Kaggle environments.
    
    Args:
        name (str): The name of the collection.
        base_path (str): Optional base directory for ChromaDB storage. If None, auto-detects.
    
    Returns:
        collection (chromadb.Collection)
    """

    if base_path is None:
        # Auto-detect: Kaggle or local
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
            base_path = "/kaggle/working/chromadb_industry"
        else:
            base_path = "./chromadb_industry"

    # Make sure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=base_path)

    # Get or create the collection
    collection = chroma_client.get_or_create_collection(name=name)

    return collection
def get_chroma_record(collection,id):
    """
    Retrieves a record from the ChromaDB collection by ID.
    
    Args:
        id (str): The ID of the record to retrieve.
    
    Returns:
        record (chromadb.Record): The retrieved record or None if not found.
    """ 
    record = collection.get(ids=[id])
    if record and len(record) > 0:
        return record['ids'][0] if record['ids'] else None
    return None

def query_model(prompt, model_name='gemini-2.0-flash',temperature=0.2,retries=5):
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 1.,
                    # "top_p": top_p,
                    # "top_k": top_k,
                    # "max_output_tokens": max_output_tokens,
                }
            )
            
            resp = model.generate_content(prompt)
            if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                return "".join(part.text for part in resp.candidates[0].content.parts)
            else:
                # Log or handle cases with no candidates or content (e.g., safety blocks)
                print(f"Warning: No valid content returned. Check response object: {resp}")
                print(f"Prompt Feedback: {getattr(resp, 'prompt_feedback', 'N/A')}")
                print(f"Candidates: {getattr(resp, 'candidates', [])}")
                return None
        except ResourceExhausted as e:
            wait_time = random.uniform(2, 4) * (attempt + 1)  # progressive backoff
            print(f"Rate limit hit. Waiting {wait_time:.1f} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            # You can decide what to do with other errors
            print(f"Unexpected error: {e}")
            raise
    raise Exception("Failed after several retries due to rate limits.")

# lib.py

def extract_json_block(text):
    """
    Extracts a JSON block from the given text. Returns the raw JSON string or None.
    """
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

#
#   smaller Models will sometimes output a trailing comma in the json block 
#   which is not valid json.
#
def clean_json_string(json_str):
    """
    Cleans common model mistakes in JSON:
    - Removes trailing commas.
    """
    return re.sub(r',\s*([\]}])', r'\1', json_str)

def extract_json_from_model_response(resp):
    """
    Extracts and parses JSON from a model response.
    """
    raw_json = extract_json_block(resp)
    if raw_json is None:
        print("\t⚠️ No JSON block found in response.")
        return []

    cleaned_json = clean_json_string(raw_json)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        print(f"\t❌ Failed to parse JSON: {cleaned_json}")
        return []
    

def get_industries(papers):
    """
    Get the list of industries from the papers dataframe.
    """
    industries = {
        industry['industry']
        for industry_list in papers['industry_list']
        if industry_list is not None   
        for industry in industry_list
    }
    return list(industries)

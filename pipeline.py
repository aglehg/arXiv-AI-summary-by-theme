import utils.lib as lib
from utils.industry import classify_industry
import json 
import os
import pandas as pd 
import google.generativeai as genai  
import google.genai as genai_embed
from utils.embedings import embed_papers
from utils.summary import get_summaries
from utils.report import generate_html_report

# initialize a google client so we can talk to the api 
config = lib.load_config()  
genai.configure(api_key=config['GOOGLE_API_KEY'])
papers = [] 

# 
# load the papers by reading the data/papers_2025-03-23_2025-03-29.json file
# if there's an intermediate file  load that one 
filepath = os.path.join(config['data_dir'],config['papers_file'])

if  os.path.exists(os.path.join(config['out_dir'],'papers_enriched.json')):
    filepath =  os.path.join(config['out_dir'],'papers_enriched.json')

print("Loading papers from:", filepath)
with open(filepath, 'r') as f:
    for line in f:
        papers.append(json.loads(line))

papers = pd.DataFrame(papers)
print("Loaded papers:", len(papers))

#
#
#     Industries 
# 
#

# Load the industry meta data from the data/industry-list.json file
with open(os.path.join(config['data_dir'], 'industry-list.json'), 'r') as f:
    industry_meta = json.load(f)

# Run each papers through the model so we can assign an industry label 
classify_industry(papers,industry_meta,config['out_dir'])
print(papers.head(10))


# What % of papers are industry related , select those with industry N/A
industry_related = papers[
    papers['industry_list'].apply(lambda x: len(x) > 0 and x[0]['industry'] != 'N/A')
]
print("Papers related to industry development:", len(industry_related))
print("Percentage of papers related to industry development:", len(industry_related)/len(papers)*100)

# Embed the papers so we can later cluster them 
collection = lib.get_chroma_collection(name="paper_abstracts", base_path=config['out_dir'])
sample_embeding = lib.get_chroma_record(collection,'2504.01981__Manufacturing')
if sample_embeding is not None:
    print("\n\n#" * 20, "ChromaDB collection already exists. Skipping embedding.", sample_embeding )
else:
    print("\n\n","#" * 20, "Embedding papers.", sample_embeding)
    client = genai_embed.Client(api_key=config['GOOGLE_API_KEY']) 
    embed_papers(client,papers)

#
# create the summaries 
# Run each papers through the model so we can assign an industry label 
print("\n\n","#" * 20, "Summarizing papers.")
get_summaries(papers,config['out_dir'])
    
# create the actual report
print("\n\n","#" * 20, "Generating report.")
generate_html_report(
    os.path.join(config['out_dir'], 'industry_sections.json'),
    os.path.join(config['out_dir'], 'final_report.html')
)
print("Report generated: ", os.path.join(config['out_dir'], 'final_report.html'))
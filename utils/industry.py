import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import time
import utils.lib  as lib



#
# Small models struggle imensely with diverging instructions 
# ex first determine if a paper mentions industry then choose 1 industry 
# To avoid using a pro model for all classifications 
# we need to first run the determination is this even an industry related paper
def is_industry(paper): 
    '''
        Check if a paper is related to industry development.
        paper - a dictionary with the paper title and abstract
        Returns True if the paper is related to industry development, False otherwise.
    '''
    # check if the paper mentions any industry
    prompt = (
        "You are a research assistant tasked with identifying papers relevant for industry development.\n"
        "Do this step by step.\n" 
        "First determine if the paper mentions any industry.\n"
        "return your determination as JSON ex ```json{is_industry: true/false}```\n"
        "You will be given a title and abstract of a paper.\n"
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract']}\n\n"  
        )
    
    resp = lib.query_model(prompt)
    
    # Extract json from the response
    determination = lib.extract_json_from_model_response(resp)

    if isinstance(determination, dict) and 'is_industry' in determination:
        return determination['is_industry']
    else:
        print(f"⚠️ Malformed determination: {determination}")
        return False  # Assume not industry related if model fails
   
#
#   When we call this function we already know the model thinks there is an industry reference 
#   therefore our task now is to determine which industries are mentioned
#
def what_industry(paper,industry_meta_prompt, allowed_industries, model_name='gemini-2.0-flash'):
    '''
        Classify the industry of a paper using the Google Gemini API.
        paper - a dictionary with the paper title and abstract
        industry_meta_prompt - a string for grounding the selected industry to a pre-selected list 
        Returns a list of industries and their relevance scores.
    '''
    industry_list = [{"industry":"N/A", "relevanceScore": 0, "summary":"Error! Invalid!"}]
    title = paper['title']
    abstract = paper['abstract']
    prompt = (
    "You are a research assistant tasked with identifying the relevant industries for a given academic paper.\n"
    "Below are predefined industries you must choose from:\n"
    f"{industry_meta_prompt}"
    "Do this step by step.\n" 
    "You will be given a title and abstract of a paper.\n"
    "First carefully read the abstract to find any clear mentions of industries or application domains.\n"
    "Only then skim the title if needed to supplement missing context.\n"
    "Map the findings to one or more industries from the list above.\n"
    "If no category is a good fit, assign the industry:Other\n"
    "Summarize this paper based on its contributions / impact into the identified industries "
    "Finally, estimate \"Relevance Score\", On a scale from 0 (not related) to 100 (highly related), how strongly is this paper connected to the [Industry] industry based on its topic and content? Provide only the numeric score.\n"
    "A week match should have a Relevance Score of 50 or less.\n"
    "A strong match should have a Relevance Score of 80 or more.\n"
    "Do not infer other sectors not mentioned.\n"
    "Do not guess possible applications.\n"

    "Ignore general technologies like \"Machine Learning\" or \"Artificial Intelligence\" unless the industry itself is the tech industry (e.g., software engineering, chip design).\n\n"
    f"Title: {title}\n"
    f"Abstract: {abstract}\n\n"  

    "Return your answer as JSON\n"
    "Expose your reasoning Example response format paper matches category x because y\ncategory alfa because beta\n"
    "```json[{ \"industry\": \"Healthcare\",\"relevanceScore\":80, \"summary\":\"relevant because it improves or impacts gama\"}{\"industry\":\"Legal & Compliance\",\"summary\":\"relevant because it improves or impacts teta\",\"relevanceScore\":20]```\n"
    )
        
    # print("Prompt:\n", prompt)
    
    start = time.time()
    resp = lib.query_model(prompt, model_name=model_name)
    end = time.time()

    print("Time taken to query model:", end - start, "seconds")
    # Extract json from the response
    try:
        parsed_industries = lib.extract_json_from_model_response(resp)
        print("Industry list:\n", parsed_industries)    
        print ("\n\n") 
    
      # 🧹 FILTER to allowed industries
        industry_list = [
            item for item in parsed_industries 
            if item.get('industry') in allowed_industries
        ]

        # 🛡 If model hallucinated everything, return the default 
        if not industry_list:
            print("⚠️ Model hallucinated everything, returning default")
            summary = ("Model hallucinated everything, returning default", " ".join(parsed_industries))
            return [{"industry":"N/A", "relevanceScore": 0, "summary": summary }]

    except Exception as e:
        print("Error extracting json from response:", e)
        print("Model returned:\n", resp) 
    
    return industry_list

def classify_industry(papers, industry_meta,OUT_DIR='out'):
    '''
    Classify the industry of a list of papers using the Google Gemini API.
    papers - a pandas data frame with the papers to classify
    industry_meta  -  a list of industries and their keywords
    data_dir - where to output the enriched json file and suspicious classifications
    '''
    # compose the industry grounding prompt from the industry meta 
    allowed_industries = []
    industry_meta_prompt = ""
    for industry in industry_meta:
        allowed_industries.append(industry['industry'])
        industry_meta_prompt += f"Industry: {industry['industry']}\n"
        industry_meta_prompt += f"Keyworkds: { ', ' .join(industry['keywords'])}\n"
        industry_meta_prompt += f"Examples: {', '.join(industry['examples'])}\n\n"

    # Fallback Category 
    allowed_industries.append("Other")

    # Collect metrics on the classification
    n_suspicious_classifications = 0
    n_fixed_classifications = 0
    # keep the smaller model outputs on suspicious cases for meta analysis
    suspicious_classifications = []

    # Iterate the papers and run them through the model 
    for i,row in papers.iterrows():
        # Keep intermediate results in case of failure
        # persist the enriched json 
        if i % 100 == 0:
            print("#"*50,"\nℹ️ Saving intermediate results...")
            papers.to_json(os.path.join(OUT_DIR,'papers_enriched.json'), orient='records', lines=True)

        title = row['title'] 
        arxiv_id = row['id']
        print(f"\n\n","#" * 50, f"\nPaper {i+1}/{len(papers)}: {arxiv_id} {title} \n\n")
        
        # if industry is already defined it's a re-run and we want to skip it
        industry_list = row.get('industry_list')

        
        if isinstance(industry_list, list) and len(industry_list) > 0: 
            print(f"Paper already classified, skipping... {type(industry_list)}", industry_list)
            continue
      
        if not is_industry(row):
            papers.at[i, 'industry_list'] = [{"industry":"N/A", "relevanceScore": 0, "summary":"No industry or application domain mentioned"}]
            print("Paper is not related to industry development, skipping...")
            continue

        industry_list = what_industry(row, industry_meta_prompt,allowed_industries)
        # was the category other assigned even though other categories are present ? 
        is_other = any(industry['industry'] == 'Other' for industry in industry_list)
        
        # If there are more than 3 categories assigned it's a suspecious classification and we want to run this paper over a smarter model
        if len(industry_list) > 2 or (len(industry_list) > 1 and is_other):
            print(f"‼️ Suspicious classification is_other:{is_other}, ncategories:{len(industry_list)} Status: {n_suspicious_classifications}/{len(papers)} fixed so far:{n_fixed_classifications}\nrunning through a smarter model")
        
            n_suspicious_classifications += 1
            suspicious_classifications.append((arxiv_id, industry_list))
            # run the paper through a smarter model 
            industry_list =  what_industry(row,industry_meta_prompt,allowed_industries, model_name='gemini-2.5-pro-preview-03-25')
            print("🤓 Larger model returned:\n", industry_list)
            if (len(industry_list) < 3):
                print("🏆Fixed successfully")
                n_fixed_classifications += 1
    
        # Recheck for halucinations 
        is_other = any(industry['industry'] == 'Other' for industry in industry_list)
        
        # if the other category was incorrectly assigned, remove it!
        if (len(industry_list) > 1 and is_other):     
            print("⚠️ Halucination! Other category assigned along other categories...Removing!")
            for industry in industry_list:
                if industry['industry'] == 'Other':
                    industry_list.remove(industry)
                    break
        
        
        # enrich the paper with the returned information 
        papers.at[i, 'industry_list'] = industry_list  
    
    # persist the enriched json 
    papers.to_json(os.path.join(OUT_DIR,'papers_enriched.json'), orient='records', lines=True)

    print("\n\n","#" * 50)
    print("Number of suspicious classifications:", n_suspicious_classifications)
    print("Number of fixed classifications:", n_fixed_classifications)
    print("Number of papers processed:", len(papers))

    # Recheck if there are still papers that where classified as suspicious for human review 
    # save the suspicious  classifications to a json file so we can review them later
    with open(os.path.join(OUT_DIR, 'suspicious_classifications.json'), 'w') as f:
        json.dump(suspicious_classifications, f, indent=4)




# Use for resume on failure 
# Run just the classification task from a previously saved enriched json file -----------------------------------------------
if __name__ == "__main__": 
    import pandas as pd
    import google.generativeai as genai

    config = lib.load_config()
    genai.configure(api_key=config['GOOGLE_API_KEY'])
    papers = pd.DataFrame(lib.load_enriched_papers())
    print("Loaded papers:", len(papers))

    # Load the industry meta data from the data/industry-list.json file
    with open(os.path.join(config['data_dir'], 'industry-list.json'), 'r') as f:
        industry_meta = json.load(f)

    # Run each papers through the model so we can assign an industry label 
    classify_industry(papers,industry_meta,config['out_dir'])
    
    print(papers.head(10))

#  / Run the classification task from a previously saved enriched json file -----------------------------------------------

    
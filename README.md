# arXiv-AI-summary-by-theme-
Capstone  - Gen AI Intensive Course 2025Q1    Extracting a Summary of Industry Applications from One Week of arXiv AI Papers  


Build a practical method to get a clear overview of what AI researchers are working on, segmented by real-world industry applications. 
Take one week of AI papers from arXiv, classify them by industry, then summarize how those papers contribute to the industry. 

This is a prototype, it contains many weaknesses that can be improved.  
 
## Key Components

1. **Seed Semantic Structure**
   - An initial industry list with keywords/examples to help ground the model when classifying papers into industries.

2. **Structured Summaries**
   - A system that produces an organized document (sections and summaries) to be reviewed by a human or easily adapted into a newsletter or executive report.

3. **Relevance Scoring**
   - A simple relevance score to prioritize papers within industries.
   - *Note:* Scores are assigned heuristically by a fast LLM model (Gemini Flash) and should be treated cautiously.

4. **Automated Clustering**
   - A mechanism (using semantic embeddings + HDBSCAN) to subdivide large industries into smaller, semantically cohesive clusters.


## More context and reflections

👉 [Medium Article: Extracting a Summary of Industry Applications from One Week of arXiv AI Papers](https://cogimagi.medium.com/extracting-a-summary-of-industry-applications-from-one-week-of-arxiv-ai-papers-a7fa9724bacd)

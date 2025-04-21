import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
from utils import lib

def generate_html_report(results_path, output_html_path='summary.html'):
    # Load the results
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    # Start HTML
    style ="<style>body{font-family:\"Courier New\",Courier,monospace;font-size:14px;}#container{width:900px;margin:auto;}.industry-section{border-left:solid thin black;padding-left:20px;margin-bottom:60px;}h1{font-size:32px;text-align:center;margin-bottom:40px;}h2{font-size:26px;margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:5px;cursor:pointer;}h3{font-size:22px;margin-top:30px;color:#333;}p{margin:10px 0 20px;}h4{margin-top:20px;font-size:18px;color:#666;}a{display:inline-block;margin-right:10px;color:#0077cc;text-decoration:none;}a:hover{text-decoration:underline;}</style>";
    html = [f"<html><head><meta charset='utf-8'><title>AI Applications by Industry</title>{style}</head><body>"]
    html.append("<div id=\"container\"><h1>AI Applications by Industry</h1>")

    for industry_summary in all_results:
        industry_name = industry_summary['industry']
        sections = industry_summary['sections']
        html.append(f"<h2>{industry_name}</h2>")

        html.append("<div class=\"industry-section\">")
        for section in sections:
            title = section.get('title', 'No Title')
            summary = section.get('summary', 'No Summary')
            papers = section.get('papers', [])

            html.append(f"<h3>{title}</h3>")
            html.append(f"<p>{summary}</p>")

            if papers:
                html.append("<h4>References</h4>")
                for paper in papers:
                    arxiv_id = paper.get('id')
                    title = paper.get('title', 'No Title')
                    if arxiv_id:
                        html.append(f"<a href='https://arxiv.org/pdf/{arxiv_id}.pdf' title='{title}' target='_blank'>{arxiv_id}</a>  ")

        html.append("</div>")  # Close section div
    # End HTML
    html.append("</div></body></html>")

    # Write to file
    with open(output_html_path, 'w') as f:
        f.write("\n".join(html))

    print(f"✅ Report generated: {output_html_path}")

# Example usage:
# generate_html_report('your_out_dir/industry_sections.json', 'your_out_dir/final_report.html')

# Use for resume on failure 
# Run just the classification task from a previously saved enriched json file -----------------------------------------------
if __name__ == "__main__": 
    config = lib.load_config()

    generate_html_report(
        os.path.join(config['out_dir'], 'industry_sections.json'),
        os.path.join(config['out_dir'])
    )


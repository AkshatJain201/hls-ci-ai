from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Tuple, Optional
import os, re, json, ast
import pandas as pd
from langchain.chains import LLMChain
from langchain_groq import ChatGroq 
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uvicorn
from dotenv import load_dotenv
from warnings import filterwarnings

app = FastAPI()
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
filterwarnings('ignore')

API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3")
]

current_key_index = 0

def get_next_api_key():
    global current_key_index
    api_key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)  # Rotate through the list
    return api_key

common_template = """
    You are a pharmaceutical analyst tasked with reviewing a text and classifying the complete article as either "Notes" worthy, "Newsletter" worthy, or "Irrelevant." Your classification should be based on the following guidelines. Provide a brief (50 words) justification for your classification.

    Rules:

    **For Notes** (High priority daily alerts):
        1) Newswires or Company PRs, including:
            • FDA, EMA, or other drug regulatory authority approvals
            • IND (Investigational New Drug) approvals
            • Future plans, timelines, and regulatory communications (e.g., CRLs)
            • Priority Reviews from regulatory bodies like the FDA
            • Phase 3 clinical trial results or above
            • Initiation or completion of Phase 3 trials or above
            • Collaboration news, mergers, acquisitions, licensing agreements
            • Expansion of manufacturing facilities
            • Preclinical data supporting new mechanisms of action (MOA)
            • Patient Support Programs (PSPs) or patient engagement initiatives
            • Deals on partnerships, collaborations, spin-offs relevant to assets
            {additional_notes_rules}

    **For Newsletter** (Weekly less time-sensitive updates):
        1) Publications or research, including:
            • Asset-related clinical trial results (Industry or non-industry)
            • Post-marketing or long-term clinical results, real-world evidence
            • Secondary analyses or cohort studies
            • Updates on ongoing clinical trials (non-phase 3 or earlier phase data)
            {additional_newsletter_rules}

    **For Irrelevant**:
        - If the content does not pertain to the above categories or is not related to Solid Tumor or Multiple Sclerosis, mark it as irrelevant.
        - Any unrelated therapeutic areas or general news without clinical/regulatory significance should also be considered irrelevant.

    Response format: 
    A tuple: ("Classification", "Reasoning in 50 words")

    Example answers:
    1. (Notes, FDA granted approval for a Phase 3 drug, high priority for daily alert)
    2. (Newsletter, Phase 2 trial results published for a new solid tumor therapy)
    3. (Irrelevant, Discusses general company financials, unrelated to drug approval or clinical results)

    Text: '{text}'
"""

specific_templates = {
    "BMS" : {
        "notes_rules" : """
                            • Conference Participation Updates (ph3 and above)
                        """, 
        "newsletter_rules" : """
                            • News about Marketing Campaigns and Initiatives for drugs in phae 3 of clinical trials or above
                        """, 
    }, 
    "EMD" : {
        "notes_rules" : """
                            • Interventional Studies, Observational studies (related to pregnancy, breast milk, Infants, etc.) (if source is newswire/company PRs)
                            • Preclinical data demonstrating assets new MOA, supporting neuroprotective nature
                            • Investor updates, Launch updates and future plans of Roche, Novartis, TG Therapeutics, Janssen, BMS, Sanofi, Biogen, Innocare
                        """, 
        "newsletter_rules" : """
                            • Interventional Studies, Observational studies (related to pregnancy, breast milk, Infants, etc.) (if source is publications)
                        """, 
    }
}

def clean_string(input_string, allowed_chars):
    # Create a regex pattern that allows letters, numbers, spaces, and the specified characters
    if not isinstance(input_string, str):
        input_string = str(input_string)
    pattern = f'[^a-zA-Z0-9 {"".join(re.escape(char) for char in allowed_chars)}]'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

classification_prompt_template = PromptTemplate(
    input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
    template = common_template,
)

chunks_prompts = """
Please summarize the below text : 
Text : '{text}'
Summary : 
"""

map_prompt_template = PromptTemplate(
    input_variables=['text'], 
    template=chunks_prompts
)

final_combine_prompt = """
Provide a final summary of the entire document with these important points. 
1. The summary should be atleast 200 words long. 
2. Don't make up any details. 
3. Use bullet points when necessary. 
4. Avoid any introductory phrases.
document : '{text}'
"""

final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)

def summarize_document(document, api_key):
    """Generate a summary for the given document."""
    chunks = text_splitter.create_documents([document])
    llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = api_key, seed = 42)
    chain = load_summarize_chain(
        llm, 
        chain_type = 'map_reduce', 
        map_prompt = map_prompt_template, 
        combine_prompt = final_combine_prompt_template,
        verbose=False
    )
    summary = chain.run(chunks)
    return summary


def news_classification(text: str, project_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Classifies the news article based on the provided project name.
    
    Parameters:
    - text (str): The news article text.
    - project_name (str): The name of the project (e.g., "BMS", "EMD").
    
    Returns:
    - Tuple (classification, reasoning): A tuple containing classification and reasoning.
    - If the text is invalid, it returns an error message instead.
    """
    
    # Check for invalid text
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return None, "Invalid input: The provided text is empty or not a valid string."
    
    # Clean and process the text
    allowed_chars = "&%$()?:"
    text = clean_string(text, allowed_chars)
    text = text.replace('$', 'USD')
    
    # Set up classification
    solution = None
    key_to_be_used = get_next_api_key()

    summary = summarize_document(text, key_to_be_used)

    llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = key_to_be_used, seed = 42)
    llm_chain = LLMChain(llm=llm, prompt=classification_prompt_template)  # Assuming `llm` and `classification_prompt_template` are defined elsewhere

    
    # Handle classification based on the project_name
    try:
        if project_name == "BMS":
            solution = llm_chain.invoke({
                "text": summary,
                "additional_notes_rules": specific_templates['BMS']['notes_rules'],
                "additional_newsletter_rules": specific_templates['BMS']['newsletter_rules']
            })
        elif project_name == "EMD":
            solution = llm_chain.invoke({
                "text": summary,
                "additional_notes_rules": specific_templates['EMD']['notes_rules'],
                "additional_newsletter_rules": specific_templates['EMD']['newsletter_rules']
            })
        else:
            solution = llm_chain.invoke({
                "text": summary,
                "additional_notes_rules": '',
                "additional_newsletter_rules": ''
            })

        # Extract the classification and reasoning
        output_tuple = tuple(solution['text'].strip("()").split(", ", 1))
        output_tuple[0].strip('').strip("")
        output_tuple[1].strip('').strip("")
        return output_tuple

    except Exception as e:
        # In case something goes wrong with classification
        return None, f"Error during classification: {str(e)}"

# Define the input data structure using Pydantic
class NewsRequest(BaseModel):
    news_article: str  # Detailed news article (can be multiple pages of text)
    project_type: str  # Project type (e.g., "Technology", "Healthcare")

# Define the API endpoint
@app.post("/api/classify-news/")
async def classify_news_article(request: NewsRequest) -> dict:
    """
    API endpoint that classifies the news article and returns classification and reasoning.
    
    Parameters:
    - request: NewsRequest model (contains news_article and project_type)
    
    Returns:
    - JSON object with classification and reasoning or an error message.
    """
    news_article = request.news_article
    project_type = request.project_type
    
    # Call the classification function
    classification, reasoning = news_classification(news_article, project_type)
    
    if classification is None:
        # If classification failed or input was invalid, raise an HTTP exception
        raise HTTPException(status_code=400, detail=reasoning)
    
    return {
        "classification": classification,
        "reasoning": reasoning
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8018, reload=True)
 
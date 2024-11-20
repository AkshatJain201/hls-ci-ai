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
from dotenv import load_dotenv
from warnings import filterwarnings

app = FastAPI()
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
filterwarnings('ignore')

# Construct the connection string
# host=os.environ.get("host")
# port=os.environ.get("port")
# admin_username=os.environ.get("admin_username") or 'akshat'
# admin_password=os.environ.get("admin_password") or 'akshat'
# database_name=os.environ.get("database_name")
# connection_string = f"mongodb://{urllib.parse.quote_plus(admin_username)}:{urllib.parse.quote_plus(admin_password)}@{host}:{port}/?authSource=admin"

# Connect to MongoDB
# client = MongoClient(connection_string)

# Access the specific database with the user credentials
# db = client[database_name]

common_template = """
    You are a pharmacuetical analyst. Go through the following text from various sources and classify the text into either newsletter or notes. 

    Rules - 

    For notes : 
        1) Newswires / Company PRs
            • Approval from FDA (Food and Drug Administration), EMA ( European Medicines Agency) or any other drug regulatory authority
            • Investigational New Drug approval (IND Approval)
            • Future Plans or timelines
            • Complete Response Letter (CRLs)
            • Priority Reviews, other communication from FDA (food and drug Administration)
            • Clinical Trial results for phase 3 and above
            • Phase or Dosing Initiation or Completion​ for a drug in phase 3 of clinical trial or above
            • Collaboration, Mergers and Acquisitions, Licensing agreements
            • Expansion of manufacturing facilities
            • Preclinical data demonstrating assets new MOA, supporting neuroprotective nature
            • Deals on MBA, partnerships, collaborations, spin offs specific to assets
            • PSPs (Patient Support Programs (PSPs)/patient engagement activities
            {additional_notes_rules}

    For newsletter : 

        1) Publications : 
            • All asset-related clinical trial results irrespective of industry or non-industry
            • Post Marketing Results, Long-term Results, Real World Evidence​
            {additional_newsletter_rules}

    Return response strictly as a tuple (either keyword notes or newsletter, its reasoning in 50 words)
    
    Example answer : 
        (Notes, Its a Ph2 new trial drug but since its a BTKI, its an alert), 
    
    Example answer : 
        (Newsletter, A preclinic product entry)

    Text : '{text}'
    
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

llm = ChatGroq(model = 'llama-3.1-70b-versatile', api_key = groq_api_key, seed = 42)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)

def summarize_document(document):
    """Generate a summary for the given document."""
    chunks = text_splitter.create_documents([document])

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
    
    # Summarize the document
    summary = summarize_document(text)
    
    # Set up classification
    solution = None
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
        return output_tuple

    except Exception as e:
        # In case something goes wrong with classification
        return None, f"Error during classification: {str(e)}"

# Define the input data structure using Pydantic
class NewsRequest(BaseModel):
    news_article: str  # Detailed news article (can be multiple pages of text)
    project_type: str  # Project type (e.g., "Technology", "Healthcare")

# Define the API endpoint
@app.post("/classify-news/")
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
import os, re, json, ast
import pandas as pd
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from dotenv import load_dotenv
from warnings import filterwarnings

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
filterwarnings('ignore')

def clean_string(input_string, allowed_chars):
    # Create a regex pattern that allows letters, numbers, spaces, and the specified characters
    if not isinstance(input_string, str):
        input_string = str(input_string)
    pattern = f'[^a-zA-Z0-9 {"".join(re.escape(char) for char in allowed_chars)}]'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

prompt_template = """
    According to the following rules classify the text into either newsletter or notes. 

    Rules - 

    For notes : 
        1) High Impactful news – Ph3 and above assets, this can include :
            • Approvals (FDA, EMA or any other key regions)
            • IND approval
            • Future Plans or timelines
            • Regular updates (priority review, CRLs etc)
            • Clinical Trial Results 
            • Phase or Dosing Initiation or Completion
            • Collaboration, Mergers and Acquisitions, Licensing agreements
            • Marketing Campaigns and Initiatives
            • Conference Participation Updates
            • Expansion (wrt manufacturing facility)
                
        2) Clinical Trials (new trials in the indication)
            • Industry Sponsored
            • Interventional Studies, Observational studies (related to pregnancy, breast
            milk, Infants, etc.)
            • New Trial Initiation (if it is either aCD20, S1Ps, BTKI or other HETs)
            • Change in Status (Active, Completed, Recruiting, Terminated)
            • Change in Start Date, PCD (if diff is more than 3 months), and SCD (if diff is
            more than 6 months)
            • Enrollment Number (if diff is significant, more than 30 patients for large
            trials (>100 patients)
            • Cover impactful trial design updates such as changes in primary and
            secondary endpoints (for instance a Phase 3 key asset)

        3) If product is either aCD20, S1Ps, BTKI or other HETs
            • New product entry (>=Ph2)
            • FDA, EMA approval/IND Submissions ( for any Ph)
            • Launch Commerical Updates/Future Plans
            
        4) Company Specific (for Roche, Novartis, TG Therapeutics, Janssen, BMS, Sanofi, Biogen or Innocare) and Investor Update or Deals on M&A or partnership or collaboration
        5) All news related to Sanofi

    For newsletter : 

    Return just one keyword, either notes or newsletter and its reasoning. 
    Example answer : [
    (Notes, Its a Ph2 new trial drug but since its a BTKI, its an alert), 
    (Newsletter, A preclinic product entry)
    ...
    ]
    Text : '{text}'
"""

prompt = PromptTemplate.from_template(prompt_template)
llm = ChatGroq(model = 'llama-3.1-70b-versatile', api_key = groq_api_key, seed = 42)
llm_chain = LLMChain(llm=llm, prompt=prompt)

def text_classification(file_path) : 
    df = pd.read_excel(file_path)
    if 'Classification' not in df.columns:
        df['Classification'] = ''
    if 'Reasoning' not in df.columns:
        df['Reasoning'] = ''
    
    allowed_chars = "&%$()?:"
    for index, text in enumerate(df['Detailed_news']):
        if pd.isna(text):  # Check for NaN values
            continue
        
        text = clean_string(text, allowed_chars)
        text = text.replace('$', 'USD')  # Assign the result back to text
        solution = llm_chain.run(text)
        # print(solution)
        output_tuple = tuple(solution.strip("()").split(", ", 1))
        # print(output_tuple)
        df.iloc[index, df.columns.get_loc('Classification')] = output_tuple[0]
        df.iloc[index, df.columns.get_loc('Reasoning')] = output_tuple[1]        

    # Return the modified DataFrame if needed
    print('done')
    return df

input_folder = 'data'
output_folder = 'output'
for file in os.listdir(input_folder) : 
    if(os.path.splitext(file)[1] != '.xlsx') : 
        continue
    file_path = os.path.join(input_folder, file)
    df = text_classification(file_path)
    output_file = os.path.join(output_folder, file)
    df.to_excel(output_file, index=True)

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
from hahaha import summarize_document

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


newasset_template = """
    You are a pharmacuetical analyst. Go through the following text and idenify if it is a new asset : 

    Rules - 

    Search for indications and MoAs in scope
    1. Search for  asset names (Alphanumerical or alphabetical) in the articles flagged as included for the project
    "2. If asset name is found
    - Match with asset names in asset list. If match is not found, flag as new asset"
    3. If asset name is not found, search for in scope MoAs in the article. Flag as new asset 

    Text : '{text}'
    
"""

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

# summary_template = """
#     You are an excellent pharmaceutical analyst. Summarize the following document in 150 words without leaving any important details.
#     Text : '{text}'
# """

classification_prompt = PromptTemplate(
    input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
    template = common_template,
)

# summary_prompt = PromptTemplate(
#     input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
#     template = summary_template,
# )

llm = ChatGroq(model = 'llama-3.1-70b-versatile', api_key = groq_api_key, seed = 42)
llm_chain = LLMChain(llm = llm, prompt = classification_prompt)

def text_classification(file_path) : 
    df = pd.read_excel(file_path)
    if 'Classification' not in df.columns:
        df['Classification'] = ''
    if 'Reasoning' not in df.columns:
        df['Reasoning'] = ''
    
    allowed_chars = "&%$()?:"
    for index, row in df.iterrows():
        print(index)
        if row['RelevancyFlag']=="No" :
            print('skipping no') 
            df.loc[index, 'Classification'] = "NA"
            df.loc[index, 'Reasoning'] = "NA"
            continue
        
        text = row['Detailed_news']

        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping row {index} due to empty or invalid Detailed_news.")
            continue

        text = clean_string(text, allowed_chars)
        text = text.replace('$', 'USD')  # Assign the result back to text

        # llm_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summarize_document(text)
        
        
        solution = None
        if row["Project"] == "BMS" : 
            solution = llm_chain.invoke({"text" : summary, "additional_notes_rules" : specific_templates['BMS']['notes_rules'], "additional_newsletter_rules" : specific_templates['BMS']['newsletter_rules']})
        elif row["Project"] == "EMD" : 
            solution = llm_chain.invoke({"text" : summary, "additional_notes_rules" : specific_templates['EMD']['notes_rules'], "additional_newsletter_rules" : specific_templates['EMD']['newsletter_rules']})
        else : 
            solution = llm_chain.invoke({"text" : summary, "additional_notes_rules" : '', "additional_newsletter_rules" : ''})
        # solution = llm_chain.run(text)
        # print(solution['text'])
        output_tuple = tuple(solution['text'].strip("()").split(", ", 1))
        print(output_tuple)
        df.loc[index, 'Classification'] = output_tuple[0]
        df.loc[index, 'Reasoning'] = output_tuple[1]

    # Return the modified DataFrame if needed
    print('done')
    return df

# def classification_text()

input_folder = 'data'
output_folder = 'output'
for file in os.listdir(input_folder) : 
    if(os.path.splitext(file)[1] != '.xlsx') or (os.path.exists(os.path.join(output_folder, file))): 
        continue
    file_path = os.path.join(input_folder, file)
    df = text_classification(file_path)
    output_file = os.path.join(output_folder, file)
    df.to_excel(output_file, index=True)
    # break

# input_file = os.path.join(os.getcwd(), 'data', 'Bio_Space_relevancy.xlsx')
# df = text_classification(input_file)

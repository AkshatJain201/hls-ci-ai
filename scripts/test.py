import time
import os, re, json, ast
import pandas as pd
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from warnings import filterwarnings
from hahaha import summarize_document
filterwarnings('ignore')
load_dotenv()

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

def clean_string(input_string, allowed_chars):
    # Create a regex pattern that allows letters, numbers, spaces, and the specified characters
    if not isinstance(input_string, str):
        input_string = str(input_string)
    pattern = f'[^a-zA-Z0-9 {"".join(re.escape(char) for char in allowed_chars)}]'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string


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

classification_prompt_template = PromptTemplate(
    input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
    template = common_template)


def text_classification(file_path) : 
    print(os.path.basename(file_path))
    df = pd.read_excel(file_path)
    if 'Classification' not in df.columns:
        df['Classification'] = ''
    if 'Reasoning' not in df.columns:
        df['Reasoning'] = ''
    
    allowed_chars = "&%$()?:"

    for index, rows in df.iterrows():
        if rows['RelevancyFlag'].lower() == "no" : 
            continue
        text = rows['Detailed_news']
        if pd.isna(text):  # Check for NaN values
            continue
        key_to_be_used = get_next_api_key()
        llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = key_to_be_used, seed = 42)
        print(key_to_be_used)
        project_name = rows['Project']
        text = clean_string(text, allowed_chars)
        text = text.replace('$', 'USD')  # Assign the result back to text
        summary = summarize_document(text)
        # in_scope(text=text, project_name=rows['Project'])
        # mark_newasset(source = rows['Category'], text = text, project_name=rows['Project'])
        # llm_chain = LLMChain(llm=llm, prompt = prompt_template)  # Assuming `llm` and `classification_prompt_template` are defined elsewhere
    
        # Handle classification based on the project_name
        try:
            llm_chain = LLMChain(llm=llm, prompt=classification_prompt_template)
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
            print(output_tuple[0])
            print(output_tuple[0].strip('').strip(""))
            print(type(output_tuple[0]))
    
        except Exception as e:
            return None, f"Error during classification: {str(e)}"

        # df.iloc[index, df.columns.get_loc('Classification')] = output_tuple[0]
        # df.iloc[index, df.columns.get_loc('Reasoning')] = output_tuple[1]        

    # Return the modified DataFrame if needed
    print('done')
    # return df

# input_folder = 'data'
# output_folder = 'output'
# for file in os.listdir(input_folder) : 
#     if(os.path.splitext(file)[1] != '.xlsx') : 
#         continue
#     file_path = os.path.join(input_folder, file)
#     # df = pd.read_excel(file_path)
#     # if df['RelevancyFlag'].lower()=='no' : 
#     #     continue
#     # mark_newasset(source, text, asset_list, newasset_rules, project_name):
#         # mark_newasset(df['Category']
#     df = text_classification(file_path)
    # output_file = os.path.join(output_folder, file)
    # df.to_excel(output_file, index=True)




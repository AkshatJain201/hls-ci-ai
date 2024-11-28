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

import json

# JSON to map source types to their respective rules
newasset_rules = {
    "news": {
        "steps": [
            "Look for asset names (either alphanumeric or alphabetical).",
            "If an asset name is found, compare it with the asset list.",
            "If no match is found, return 'yes' (flag for SME review).",
            "If no asset name is found, look for in-scope MoAs related to the project.",
            "If in-scope MoAs are found, return 'yes' (flag for SME review)."
        ]
    },
    "general_terms": {
        "steps": [
            "Check for terms like 'discovery', 'new', 'preclinical', 'biosimilars', or 'repurposing'.",
            "If any of these terms are found, return 'yes' (flag for SME review)."
        ]
    }
}
   
asset_dict = {
    "EMD" : ["Ocrevus", "aCD20s", "Kesimpta", "aCD20s", "Tysabri", "Integrin α4β1 antagonist mAb", "Lemtrada", "Anti-CD52 mAb", "Ponvory", "S1PR modulator", "Zeposia", "S1PR modulator", "Mayzent", "S1PR modulator", "Gilenya", "S1PR modulator", "Tascenso ODT", "S1PR modulator", "Mavenclad", "Lymphocyte targeting agent", "Tecfidera", "NRF2 activator", "Vumerity", "NRF2 activator", "Bafiertam", "NRF2 activator", "Rebif", "IFN β-1a", "Avonex", "IFN β-1a", "Plegridy", "PegIFN β-1a", "Aubagio", "DHODH inhibitor", "Copaxone", "Peptide copolymer", "Betaferon/Betaseron", "IFN β-1b", "Extavia", "IFN β-1b", "Briumvi", "aCD20s", "Tolebrutinib", "BTKi", "Fenebrutinib", "BTKi", "Remibrutinib (LOU064)", "BTKi", "Orelabrutinib", "BTKi", "Plegridy, ped (Biogen/PegIFN β-1a)", "PegIFN β-1a", "IMU-838", "DHODH inhibitor", "GA Depot", "Peptide copolymer", "Masitinib", "c-kit/PDGFR TKI", "CT-P53 (Ocrevus Biosimilar)", "aCD20s", "Xacrel (Ocrevus Biosimilar)", "aCD20s", "Frexalimab (SAR441344)", "Anti-CD40L Mab", "BIIB091", "BTKi", "Zeposia", "S1PR modulator", "EHP-101", "PPARγ/CB2 dual agonist", "Ibudilast (MN-166)", "PDE4 and cytokines inhibitor", "23", "LSD1 MAO-B inhibitor", "Telitacicept", "TACIFc fusion protein", "NVG-291", "PTPσ Inhibitor", "SN1011", "BTKi", "Temelimab", "HERVW Env antagonist", "Foralumab", "aCD3 mAb", "SAR443820 (DNL788)", "RIPK1 inhibitor", "Jaypirca (Pirtobrutinib/LY3527727/LOXO-305)", "BTKi", "LY3541860", "aCD19 mAb", "MP101", "Mitochondrial modulator", "IMCY-0141", "Antigen-specific cytolytic CD4 T cells stimulant", "HuL001", "aENO1 mAb", "Ninlaro", "Proteasome inhibitor", "GSK3888130B", "anti-IL7", "SIR2446M (SIR-2446)", "RIP1 inhibitor", "ACT-1004-1239", "ACKR3/CXCR7 antagonist", "BOS172767/ARN-6039", "ROR-γt inverse agonist", "ANK-700", "Unknown", "T20K", "IL2 inhibitor", "PIPE-307", "M1R Antagonist", "BIIB107", "aVLA4/alpha-4 integrins targeting mAb", "RG6035 (BS)-CD20 (RO7121932)", "aCD20 mAb", "IMG-004", "BTKi", "LP-168", "BTKi", "PIPE-791", "LPA1R Antagonist", "LPX-TI641", "Immunotherapy", "Bryostatin-1", "Protein kinase C activator", "Undisclosed asset", "BTKi", "BMS-986353 (CD19 NEX T)", "Immunologic cytotoxicity; T lymphocyte replacements", "BMS-986465/TYK2i-CNS", "TYK2i", "Lucid-21-302 (LUCID-MS)", "Protein-arginine deiminase inhibitor", "MTR-601", "Highly selective fast twitch myosin 2 ATPase inhibitor", "IC 100-02", "Inflammasome ASC Inhibitor", "Lu AG22515", "Anti-CD40L", "KYV-101", "Anti-CD19 CAR-T", "MRT-6160", "VAV1-directed MGD", "ACT-101", "recombinant human alpha fetoprotein", "BMS-986196", "Unknown", "MP1032", "Anti-inflammatory compounds targeting activated macrophages", "Aldesleukin (ILT-101)", "Interleukin 2 agonist", "EmtinB", "LRP-1 agonist", "CNM-AU8", "BCMA", "BRL-303", "CD19", "Equecabtagene Autoleucel", "BCMA CAR-T", "IMP761", "LAG-3 agonist antibody", "Azer-Cel", "CD19 CAR T", "Obexelimab", "aCD19 mAb", "ABA-101", "autologous Treg Cell therapy", "IMPT-514", "CD19/CD20 bispecific CAR-T cell therapy"],    
    "BMS" : ["Unnamed CCR8", "CHS-3318", "CTM033", "FG-3175", "FG-3163", "FPA157", "GB2101", "GNUV-202", "IO-1", "ABBV-514", "PSB114", "REMD-355", "SD-356253", "αCCR8 mAb", "AMG 355", "BCG005", "IPG0521", "SSFF-02", "HL2401", "DT-7012", "Unnamed CCR8", "Unnamed CCR8", "Unnamed CCR8", "PM1092", "GBD201", "Unnamed CCR8", "TBR323", "BAY3375968", "BGB-A3055", "BMS-986340", "CM369 (ICP-B05)", "GS-1811 (JTX-1811)", "HBM1022", "IPG-7236", "LM-108", "RO7502175", "S-531011", "SRF-114", "ZL-1218", "REMD-578", "2MW4691", "Unnamed CCR8"]
    }

def mark_newasset(source, text, project_name):
    """
    Generate a prompt and call LLM to determine whether the text is about a new or existing asset.
    
    Args:
    source (str): The source to select (e.g., 'ct.gov', 'pubmed', 'newswires_prs', 'general_terms')
    text (str): The text content to analyze.
    asset_list (list): The list of existing assets to compare against.
    newasset_rules (dict): The dictionary holding rules for each source (e.g., 'ct.gov', 'pubmed').
    groq_api_key (str): The API key for the ChatGroq model.
    
    Returns:
    str: 'yes' or 'no' based on whether the asset is new or existing.
    """
    # Check if the source exists in the rules
    if source not in newasset_rules:
        return "Invalid source selected."
    

    asset_list = ''
    if project_name == "BMS" : 
        asset_list = asset_dict[project_name]
    elif project_name == "EMD" : 
        asset_list = asset_dict[project_name]
    else : 
        asset_list = []
        asset_list.extend(asset_dict['BMS'])
        asset_list.extend(asset_dict['EMD'])

    steps = ''
    if source in ['news', 'ct.gov', 'pubmed'] : 
        steps = newasset_rules['news']['steps']
    else : 
        # steps = newasset_rules['general_instructions']['steps']
        steps = newasset_rules['news']['steps']
    
    # Prepare the prompt template
    prompt_template = """
    You are a medical research expert. Follow these instructions to determine whether the text is about a new or existing asset compared to the asset list. 
    
    Instructions: '{steps}'

    Asset List: '{asset_list}'

    Text: '{text}'

    Your response should be either 'yes' or 'no'.
    """
    
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=['text', 'asset_list', 'steps'],
        template=prompt_template,
    )
    
    # Initialize the LLM chain with the selected model and API key
    llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=groq_api_key, seed=42)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain with correct variable names
    result = llm_chain.invoke({'text': text, 'asset_list': asset_list, 'steps': steps})
    
    print(result)

def in_scope(text, project_name) : 
    prompt_dict = {
        "EMD" : """
        You are a pharmaceutical expert. Review the given below text and check if the content is relevant.
        The content should be talking about Multiple Sclerosis, a variation of multiple sclerosis and not about any other therapeutic area. However, if 
        the content does not talk about any therapeutic area at all, and rather talks about a strategic alliance, financial performance or 
        corporate strategy, do not flag it as irrelevant.

        Answer should be like a tuple with yes if it is relevant and no if its not along with its reasoning.

        Text : '{text}'
        
        Answer : 
        ("yes", "the moa is in clinical trials btki")
        ("no", "the article is about pregnancy")
        """, 
        "BMS" : """
            For the content to be flagged as relevant, the content should be about clinical trials, drugs, therapies or other aspects of Solid Tumor. 
            If the content is talking about any other therapeutic area, mark the content as not relevant. However, if the content does not talk about 
            any therapeutic area at all, and rather talks about strategic alliance, corporate strategy or financial results, do not mark the content 
            as irrelevant.
        
            Answer should be like a tuple with yes if it is relevant and no if its not along with its reasoning.

            Text : '{text}'
            
            Answer : 
            ("yes", "the moa is in clinical trials btki")
            ("no", "the article is about pregnancy")
        """,
    }
    if (project_name not in ['BMS', 'EMD']) : 
        print('our current technology limits us') 
        
    else :     
    # Define the prompt template
        prompt = PromptTemplate(
            input_variables=['text'],
            template= prompt_dict[project_name],
        )
        
        # Initialize the LLM chain with the selected model and API key
        llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=groq_api_key, seed=42)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Invoke the LLM chain with correct variable names
        result = llm_chain.invoke({'text': text})
        print(result)

prompt = PromptTemplate.from_template(prompt_template)
llm = ChatGroq(model = 'llama-3.1-70b-versatile', api_key = groq_api_key, seed = 42)
llm_chain = LLMChain(llm=llm, prompt=prompt)

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
        
        text = clean_string(text, allowed_chars)
        text = text.replace('$', 'USD')  # Assign the result back to text
        # in_scope(text=text, project_name=rows['Project'])
        # mark_newasset(source = rows['Category'], text = text, project_name=rows['Project'])
        solution = llm_chain.run(text)
        # print(solution)
        output_tuple = tuple(solution.strip("()").split(", ", 1))
        print(output_tuple)
        df.iloc[index, df.columns.get_loc('Classification')] = output_tuple[0]
        df.iloc[index, df.columns.get_loc('Reasoning')] = output_tuple[1]        

    # Return the modified DataFrame if needed
    print('done')
    # return df

input_folder = 'data'
output_folder = 'output'
for file in os.listdir(input_folder) : 
    if(os.path.splitext(file)[1] != '.xlsx') : 
        continue
    file_path = os.path.join(input_folder, file)
    # df = pd.read_excel(file_path)
    # if df['RelevancyFlag'].lower()=='no' : 
    #     continue
        # mark_newasset(source, text, asset_list, newasset_rules, project_name):
        # mark_newasset(df['Category']
    df = text_classification(file_path)
    # output_file = os.path.join(output_folder, file)
    # df.to_excel(output_file, index=True)

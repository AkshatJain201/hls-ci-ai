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
# from hahaha import summarize_document

load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY_2')
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

def clean_string(input_string, allowed_chars):
    # Create a regex pattern that allows letters, numbers, spaces, and the specified characters
    if not isinstance(input_string, str):
        input_string = str(input_string)
    pattern = f'[^a-zA-Z0-9 {"".join(re.escape(char) for char in allowed_chars)}]'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

def in_scope(text, project_name, api_key) : 
    prompt_dict = {
        "EMD" : """
        You are a pharmaceutical expert. Review the given below text and check if the content is relevant.
        The content should be talking about Multiple Sclerosis, a variation of multiple sclerosis and not about any other therapeutic area or if the content is talking about a strategic alliance, financial performance or 
        corporate strategy, flag it as EMD.

        Answer should strictly be a tuple with two elements
         1. category :  EMD if it is relevant and no if its not
         2. reasoning : in one line in about 50 words, no internal inverted commas allowed. 

        Text : '{text}'
        
        Answer : 
        (EMD, the moa is in clinical trials btki)
        (no, the article is about pregnancy)
        """, 
        "BMS" : """
            For the content to be flagged as relevant, the content should be about clinical trials, drugs, therapies or other aspects of Solid Tumor or if the content talks about 
            strategic alliance, corporate strategy or financial results, still mark it as relevant.

            Answer should strictly be a tuple with two elements
         1. category :  BMS if it is relevant and no if its not
         2. reasoning : in one line in about 50 words, no internal inverted commas allowed. 

            Text : '{text}'
            
            Answer : 
            (BMS, the moa is in clinical trials btki)
            (no, the article is not talking about solid tumor)
        """,
        "Both" : """
            You are a pharmaceutical expert. Review the given below text and check if the content is relevant.
            1. If the content is talking about Multiple Sclerosis, a variation of multiple sclerosis mark it as EMD.
            2. If the content is talking about about clinical trials, drugs, therapies or other aspects of Solid Tumor mark it as BMS. 
            3. If the content talks about a strategic alliance, financial performance or corporate strategy, flag it as relevant.

            Answer should strictly be a tuple with two elements
         1. category :  EMD or BMS or relevant or no.
         2. reasoning : in one line in about 50 words, no internal inverted commas allowed. 

            Text : '{text}'
            
            Answer : 
            (EMD, the moa is in clinical trials btki)
            (BMS, the content is about a newly discovered moa)
            (relevant, talks about Roche's and Novartis' new joint venture)
            (no, the article is about pregnancy)
        """
    }
    choose_template = ''
    if project_name.lower() in ['bms', 'emd'] : 
        choose_template = prompt_dict[project_name]
    else : 
        choose_template = prompt_dict['Both']

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=['text'],
        template = choose_template,
    )
        
    # Initialize the LLM chain with the selected model and API key
    llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=api_key, seed=42)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
        
    # Invoke the LLM chain with correct variable names
    result = llm_chain.invoke({'text': text})
    output_list = result['text'].strip('()').split(', ')
    output_tuple = (output_list[0], ' '.join(output_list[1:]))
    return output_tuple

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

# def mark_newasset(source, text, project_name):
def mark_newasset(text, project_name, api_key):
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
    # if source not in newasset_rules:
    #     return "Invalid source selected."
    

    asset_list = ''
    if project_name.lower() == "bms" or project_name.lower() == "emd": 
        land = asset_dict[project_name]
        asset_list = ', '.join(land)
    else : 
        land = []
        land.extend(asset_dict['BMS'])
        land.extend(asset_dict['EMD'])
        asset_list = ', '.join(land)

    steps = ''
    # if source in ['news', 'ct.gov', 'pubmed'] : 
    #     steps = newasset_rules['news']['steps']
    # else : 
        # steps = newasset_rules['general_instructions']['steps']
    steps = newasset_rules['news']['steps']
    
    # Prepare the prompt template
    prompt_template = """
    You are a medical research expert. Follow these instructions to determine whether the text is about a new or existing asset compared to the asset list. 
    
    Use this list as a set of instructions to evaluate the content.
    Instructions: '{steps}'

    This is the current list of assets to check against.
    Asset List: '{asset_list}'

    Text: '{text}'

    Your response should stricly be a tuple with : 
    1. yes/no
    2. reasoning in 50 words

    Example Answer : 
    (no, the report talks about financial quarter this year from Roche and does not include a drug name)
    """
    
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=['text'],
        template=prompt_template,
        partial_variables={'asset_list' : asset_list, 'steps' : steps}
    )
    
    # Initialize the LLM chain with the selected model and API key
    llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=api_key, seed=42)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain with correct variable names
    result = llm_chain.invoke({'text': text})
    output_list = result['text'].strip('()').split(', ')
    output_tuple = (output_list[0], ' '.join(output_list[1:]))
    # print(output_tuple)
    return output_tuple


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


classification_prompt_template = PromptTemplate(
    input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
    template = common_template)

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
def text_classification(file_path):
    df = pd.read_excel(file_path)
    if 'Classification' not in df.columns:
        df['Classification'] = ''
    if 'Classification_Reasoning' not in df.columns:
        df['Classification_Reasoning'] = ''
    if 'Inscope' not in df.columns:
        df['Inscope'] = ''
    if 'Inscope_Reasoning' not in df.columns:
        df['Inscope_Reasoning'] = ''
    if 'NewAsset' not in df.columns:
        df['NewAsset'] = ''
    if 'NewAsset_Reasoning' not in df.columns:
        df['NewAsset_Reasoning'] = ''
    
    allowed_chars = "&%$()?:"
    
    for index, row in df.iterrows():
        if index%5 == 0 : 
            print(index)
        if pd.isna(row['Relevancy Flag']) or row['Relevancy Flag'].lower() == "no":
            continue
            
        text = row['Detailed News']
        if pd.isna(text):
            continue
            
        project_name = row['Project Name']
        text = clean_string(text, allowed_chars)
        text = text.replace('$', 'USD')
        key_to_be_used = get_next_api_key()
        summary = summarize_document(text)

        project_name_detected, project_reasoning = in_scope(text=summary, project_name=project_name, api_key=key_to_be_used)
        
        # Update inscope columns directly in dataframe
        df.at[index, "Inscope"] = project_name_detected if project_name_detected.lower() in ['bms', 'emd', 'no'] else "BMS, EMD"
        df.at[index, "Inscope_Reasoning"] = project_reasoning
        
        if project_name_detected.lower() == "no":
            continue

        if project_name_detected.lower() in ['bms', 'emd']:
            new_asset_indentified, asset_reasoning = mark_newasset(text=summary, project_name=project_name_detected, api_key=key_to_be_used)
            # Update new asset columns
            df.at[index, "NewAsset"] = new_asset_indentified
            df.at[index, "NewAsset_Reasoning"] = asset_reasoning
            
            if new_asset_indentified.lower() == "yes":
                df.at[index, "Classification"] = "Notes"
                df.at[index, "Classification_Reasoning"] = "New Asset"
                continue

        llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=key_to_be_used, seed=42)
        try:
            llm_chain = LLMChain(llm=llm, prompt=classification_prompt_template)
            
            template_key = project_name if project_name in ["BMS", "EMD"] else None
            
            solution = llm_chain.invoke({
                "text": summary,
                "additional_notes_rules": specific_templates[template_key]['notes_rules'] if template_key else '',
                "additional_newsletter_rules": specific_templates[template_key]['newsletter_rules'] if template_key else ''
            })
            
            # Extract the classification and reasoning
            output_tuple = tuple(solution['text'].strip("()").split(", ", 1))
            
            # Update classification columns
            df.at[index, 'Classification'] = output_tuple[0]
            df.at[index, 'Classification_Reasoning'] = output_tuple[1]

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue

    return df

# input_folder = 'data'
# output_folder = 'output'
# values = 0
# for file in os.listdir(input_folder) : 
#     # if file not in ['output_3.xlsx', 'output_10.xlsx'] : 
#     #     continue
#     print(file)
#     if values == 2 : 
#         break
#     if(os.path.splitext(file)[1] != '.xlsx') or (os.path.exists(os.path.join(output_folder, file))): 
#         continue
#     file_path = os.path.join(input_folder, file)
#     df = pd.read_excel(file_path)
#     df = text_classification(file_path)
#     output_file = os.path.join(output_folder, file)
#     df.to_excel(output_file, index=True)
#     values += 1

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

# prompt_template = """
#     You are a pharmacuetical analyst. Go through the following text from various sources and classify the complete article as either newsletter worthy or notes worthy or irrelavant. 

#     Rules - 

#     For notes : 
#         1) Newswires / Company PRs
#             • Approval from FDA (Food and Drug Administration), EMA ( European Medicines Agency) or any other drug regulatory authority
#             • Investigational New Drug approval (IND Approval)
#             • Future Plans or timelines
#             • Complete Response Letter (CRLs)
#             • Priority Reviews, other communication from FDA (food and drug Administration)
#             • Clinical Trial results for phase 3 and above
#             • Phase or Dosing Initiation or Completion​ for a drug in phase 3 of clinical trial or above
#             • Collaboration, Mergers and Acquisitions, Licensing agreements
#             • Expansion of manufacturing facilities
#             • Preclinical data demonstrating assets new MOA, supporting neuroprotective nature
#             • Deals on MBA, partnerships, collaborations, spin offs specific to assets
#             • PSPs (Patient Support Programs (PSPs)/patient engagement activities
#             {additional_notes_rules}

#     For newsletter : 

#         1) Publications : 
#             • All asset-related clinical trial results irrespective of industry or non-industry
#             • Post Marketing Results, Long-term Results, Real World Evidence​
#             {additional_newsletter_rules}

#     For irrelevant : 
        
#         If the content does not talk about anything above and is neither related to Solid Tumor nor Multiple Sclerosis, mark it as irrelevant.

#     Response should strictly be a tuple (either keyword 'Notes' or 'Newsletter' or 'Irrelevant', its reasoning in 50 words) else you will be penalized.
    
#     Example answer : 
#         (Notes, Its a Ph2 new trial drug but since its a BTKI, its an alert), 
    
#     Example answer : 
#         (Newsletter, A preclinic product entry)

#     Example answer : 
#         (Irrelevant, it talks about new drug trials for neuroscience)
#     Text : '{text}'
# """

prompt_template = """
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
    
    Use this list as a set of instructions to evaluate the content.
    Instructions: '{steps}'

    This is the current list of assets to check against.
    Asset List: '{asset_list}'

    Text: '{text}'

    Do not include instructions or asset list in the output.
    Your response should only be 'yes' or 'no' and reasoning.
    """
    
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=['text'],
        template=prompt_template,
        partial_variables={'asset_list' : asset_list, 'steps' : steps}
    )
    
    # Initialize the LLM chain with the selected model and API key
    llm = ChatGroq(model='llama-3.1-70b-versatile', api_key=groq_api_key, seed=42)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Invoke the LLM chain with correct variable names
    result = llm_chain.invoke({'text': text})
    
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

classification_prompt_template = PromptTemplate(
    input_variables= ["text", "additional_notes_rules", "additional_newsletter_rules"],
    template = prompt_template)

llm = ChatGroq(model = 'llama-3.1-8b-instant', api_key = groq_api_key, seed = 42)

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
            print(output_tuple)
    
        except Exception as e:
            return None, f"Error during classification: {str(e)}"

        # df.iloc[index, df.columns.get_loc('Classification')] = output_tuple[0]
        # df.iloc[index, df.columns.get_loc('Reasoning')] = output_tuple[1]        

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

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
from summarizer import summarize_document

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
def mark_newasset(text, project_name, sourceid, api_key):
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

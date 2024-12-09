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

def text_classification(text, project_name, sourceid, key_to_be_used):

    if not isinstance(text, str) or text == '':
        print("text is not proper")
        return '', ''

            
    llm = ChatGroq(model='llama-3.1-8b-instant', api_key=key_to_be_used, seed=42)
    try:
        llm_chain = LLMChain(llm=llm, prompt=classification_prompt_template)
            
        template_key = project_name if project_name in ["BMS", "EMD"] else None
            
        solution = llm_chain.invoke({
        "text": text,
        "additional_notes_rules": specific_templates[template_key]['notes_rules'] if template_key else '',
        "additional_newsletter_rules": specific_templates[template_key]['newsletter_rules'] if template_key else ''
        })
            
        # Extract the classification and reasoning
        output_tuple = tuple(solution['text'].strip("()").split(", ", 1))

    except Exception as e:
        print(f"Classification encountered some server error")
        




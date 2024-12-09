import os, re, json, ast
import pandas as pd
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def in_scope(text, project_name, sourceid, api_key) : 
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

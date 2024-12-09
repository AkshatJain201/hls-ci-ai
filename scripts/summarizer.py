import os
import json
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from warnings import filterwarnings

filterwarnings('ignore')

#load env data
load_dotenv()

#output parser
parser = StrOutputParser()
groq_api_key = os.getenv('GROQ_API_KEY_1')
# Initialize the ChatGroq model
model = ChatGroq(model = 'llama-3.1-8b-instant', api_key = groq_api_key, seed = 42)


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
1. The summary should be atmost 400 words long. 
2. Don't make up any details. 
3. Use bullet points when necessary. 
4. Avoid any introductory phrases.
document : '{text}'
"""

final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)

#using Map Reduce
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)

def summarize_document(document):
    """Generate a summary for the given document."""
    chunks = text_splitter.create_documents([document])

    chain = load_summarize_chain(
        model, 
        chain_type = 'map_reduce', 
        map_prompt = map_prompt_template, 
        combine_prompt = final_combine_prompt_template,
        verbose=False
    )
    summary = chain.run(chunks)
    # print('summary done')
    return summary
    

# input_folder = os.path.join(os.getcwd(), 'output')
# text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
# summaries = []
# for text_file in text_files : 
#     text_path = os.path.join(input_folder, text_file)
#     document = load_text_file(text_path)
#     summary = summarize_document(document)
#     summaries.append({"document": text_file, "summary": summary})
#     with open(os.path.join(metadata_folder, "summaries.json"), "w") as f:
#         json.dump(summaries, f, indent=4)
#     print("Summaries saved to summaries.json")

# document = ''
# for doc in summaries : 
#     document += doc['summary']
# doc_level_summary = summarize_document(document)
# print(doc_level_summary)
# summaries.append({'document' : 'doc_level_summary', 'summary' : doc_level_summary})
# with open(os.path.join(metadata_folder, "summaries.json"), "w") as f:
#     json.dump(summaries, f, indent=4)

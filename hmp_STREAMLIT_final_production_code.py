# Importing Necessary packages
import streamlit as st 
# import streamlit.components.v1 as components
# import json
from snowflake.snowpark.context import get_active_session
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import * 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import ast
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sys
import base64
import io
import gzip


session = get_active_session() 


################################## Global Constants START #############################################
MASTER_TABLE = "CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW"

EMBEDDING_MODEL = 'snowflake-arctic-embed-m'
LLM_MODEL = "llama3.1-70b"
TOP_K = 30
DELIMITER = ";"

################################## Global Constants END #############################################


# Function to save CSV in chunks
def convert_into_csv(df):
    # Limiting to 300,000 rows if needed
    df_new = df if len(df) <= 200000 else df[:200000]
    
    # Create a buffer to store compressed data
    buffer = io.BytesIO()
    
    # Write the CSV data to Gzip file in memory
    with gzip.GzipFile(fileobj=buffer, mode="w") as gz:
        csv_data = df_new.to_csv(index=False)
        gz.write(csv_data.encode('utf-8'))
    
    # Convert buffer content to base64 for download
    buffer.seek(0)  # Reset buffer position
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return b64

def disable():
    st.session_state.disabled = True

st.set_page_config(layout="wide")
# Defining App Styling

pd.set_option("styler.render.max_elements", 611832)

# ----------------------------------- HMP LOGO -------------------------------------------
def display_hmp_logo():
    session.file.get('@"CUSTOMERS_DEV"."DATA_SCIENCE_DEV"."JMQ1F7QVN2D336D7 (Stage)"/HMP_logo_Logo.jpg','/tmp')
    left_co,center_co ,right_co = st.columns([1,1,1])
    with center_co:
        st.image('/tmp/HMP_logo_Logo.jpg',width=300)

display_hmp_logo()
def highlight(val):
    if val == 'High':
        color = '#28a745'
    elif val == 'Medium':
        color = '#fd7e14'
    else:
        color = '#dc3545'
    return f'background-color: {color}'
    
st.write("***:beginner: You can use this app to find the relevent Healthcare professional's information based on thier interest in a particular specialization, a drug or a disease.***")

url = "https://www.dropbox.com/scl/fi/r993jmq852kymaksyz6tv/HMP_Global_DS_User_Guide_v2_-All_Specs.pdf?rlkey=7dtjjt6f94t84iykevveksl8d&st=n4xx60gj&dl=0"
# st.markdown("[App Usage Guide](%s)" % url)
# Create a styled hyperlink
styled_link = f'''
<a href="{url}" style="
    display: inline-block;
    font-size: 16px;
    color: white;
    background-color: #4CAF50;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    border-radius: 5px;
    margin-top: 10px;
">APP USAGE GUIDE</a>
'''
st.markdown(styled_link, unsafe_allow_html=True)

# ---------------------------- CSS for Download Button ---------------------------
button_style = """
        <style>
            .stDownloadButton button {
                background-color: #4CAF50; /* Green background */
                color: white; /* White text */
                padding: 10px 20px; /* Padding */
                text-align: center; /* Centered text */
                text-decoration: none; /* No underline */
                display: inline-block; /* Inline-block */
                font-size: 16px; /* Font size */
                margin: 4px 2px; /* Margin */
                cursor: pointer; /* Pointer cursor on hover */
                border-radius: 12px; /* Rounded corners */
                border: none; /* No border */
                transition: transform 0.2s, box-shadow 0.2s; /* Smooth transition */
            }
            .stDownloadButton button:hover {
                background-color: #45a049; /* Darker green on hover */
                color: white; /* Keep text color white on hover */
                transform: scale(1.05); /* Slightly increase size */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Add shadow */
            }
    </style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(button_style, unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------------------

st.divider()
option_msg = (
    "**Select the first option to find Healthcare Professionals in the HMP Via database based on selected parameters of interest.**",
    "**Select the second option to enter your own open-text query about the Healthcare Professionals in the HMP Via database.**"
)
# To solve following streamlit warning, placeholder label is added & hidden with label visibility 
# Warning Msg:  `label` got an empty value. This is discouraged for accessibility reasons and may be disallowed in the future by raising an exception. Please provide a non-empty label and hide it with label_visibility if needed.
option = st.radio("placeholder",option_msg, label_visibility='hidden') 

################################## LLM Spell Correction START #############################################
# ---------------------- Extracting & Correcting names from the input query -------------------------
def get_completion(prompt: str, llm_model=LLM_MODEL):
    sql_stmt = "select snowflake.cortex.complete(:1, :2) as LLM_REPONSE"
    # log.info(f"get_completion SQL: {sql_stmt}")
    sql_response = session.sql(sql_stmt, (llm_model, prompt)).collect()
    llm_response = sql_response[0][0]
    return llm_response


def get_context_v1():
    def get_context_helper(vocab: str):        
        get_context_sql_stmt = f"SELECT DISTINCT value FROM {MASTER_TABLE}  WHERE vocab = :1 order by value;"
        # log.info(f"running sql_stmt for {vocab}: {get_context_sql_stmt}")
        sql_params = (vocab)
        sql_response = session.sql(get_context_sql_stmt, sql_params).collect()
        context = ';'.join(str(row["VALUE"]).replace("'","").replace('"','') for row in sql_response)
        return context

    drug_context = get_context_helper(vocab = 'drug') # Get list of drugs
    disease_context = get_context_helper(vocab = 'disease') # Get list of diseases
    return drug_context, disease_context


def get_context_v2(question: str):
    get_context_sql_stmt: str = f"""
    select distinct VALUE from ( 
      select distinct 
        VALUE,
        vector_cosine_similarity(
          EMBEDDING_VALUE, 
          snowflake.cortex.embed_text_768(:1, :2)
        ) as SIMILARITY
      from {MASTER_TABLE} 
      where VOCAB = :3 
      order by SIMILARITY desc
      limit :4
    )
    order by VALUE;
    """
    def get_context_helper(vocab: str):
        # log.info(f"running sql_stmt for {vocab}: {get_context_sql_stmt}")
        sql_params = (EMBEDDING_MODEL, question, vocab, TOP_K)
        sql_response = session.sql(get_context_sql_stmt, sql_params).collect()
        context = ';'.join(str(row["VALUE"]).replace("'","").replace('"','') for row in sql_response)
        return context

    drug_context = get_context_helper(vocab = 'drug') # Get list of drugs
    disease_context = get_context_helper(vocab = 'disease') # Get list of diseases
    return drug_context, disease_context


GET_CONTEXT_V2_PROMPT_TEMPLATE = """
You are given a user question given within <QUESTION></QUESTION> tag.
Information to understand the user question, process it & give output is available within <CONTEXT></CONTEXT> tag.

The Context will contain 2 lists namely a "List of Drugs" and a "List of Diseases". Items within each list will be separated by delimiter {DELIMITER} .
The User question will be in natural language that will contain name of a single drug or disease.
There is a possibility of spelling mistakes for drug/disease name in user question. In such a case, correct the spelling based on closest match found within Context.
From the statement below, extract only the names of drugs and diseases that are explicitly mentioned, matching them to the closest valid name from the list. "
Correct any spelling mistakes by replacing the incorrect name with the closest valid name. 
Output should be a single line of comma-separated names, with the original names corrected to match the valid names. 
Do not include any additional text, explanations, notes, or unmentioned items. 
Strictly return only the corrected and matched names from the original statement, with corrected spellings as needed. 
#Your objective is to identify & if necessary correct spelling of the drug/disease name entered by user based on List of Drugs & List of Diseases in context.

<CONTEXT>
List of Drugs: {drug_context}

List of Diseases: {disease_context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>

Output only the name of drug/disease with the corrected spelling, if needed. Do not output anything else.
In case, you find that user question does not contain any drug/disease name, output 1 word namely "error" without the double quotes 
"""

def perform_query_sanitization(question: str):
    """
    Given user query which may/may not contain names of drugs and/or diseases, perform sanity check to do following:
      1. If query contains a valid drug/disease name do nothing. Passed Sanity Check
      2. If query contains a wrong drug/disease name, remove it from user query.
      3. If query contains a misspelt drug/disease name, correct the spelling based on available info in DB.
      4. If query contains no valid drugs/diseases, then mark it as error & dont proceed to error step.
    """
    # drug_context, disease_context = get_context_v1()
    drug_context, disease_context = get_context_v2(question) # uses RAG to make context small
    prompt: str = GET_CONTEXT_V2_PROMPT_TEMPLATE.format(DELIMITER=DELIMITER, drug_context=drug_context, disease_context=disease_context, question=question)
    llm_response = get_completion(prompt)
    return llm_response


MULTI_ITEM_PROMPT_TEMPLATE = """
You are given a user question given within <QUESTION></QUESTION> tag.
Information to understand the user question, process it & give output is available within <CONTEXT></CONTEXT> tag.

You are an advanced AI model specializing in extracting named entities like drugs and diseases from text.
Below is a user question that may mention up to 3 drugs or diseases, potentially with spelling mistakes. 
Additionally, you are provided with two {DELIMITER} delimited lists: one containing possible drug names and the other containing possible disease names. 
These lists were derived by performing a vector cosine similarity match between the embedding of the user question and embeddings of drugs and diseases in a database.

Your task is to:
1. Use the provided lists to aid in spelling correction and identification of drugs or diseases mentioned in the user question.
2. Correct any spelling errors in drug or disease names (if necessary).
3. Identify all drugs and diseases mentioned in the question.
4. Return the identified names in a Python list.
5. If no drugs or diseases are mentioned, return an empty Python list.

<CONTEXT>
List of Drugs: {drug_context}

List of Diseases: {disease_context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>

Your Output should return a Python List only that can be parsed correctly by the ast.literal_eval function. Account for quotes in the drug/disease name so that they wont fail the parsing.
"""


def extract_items_from_user_query(question : str) -> List[str]:
    drug_context, disease_context = get_context_v2(question) # uses RAG to make context small
    prompt: str = MULTI_ITEM_PROMPT_TEMPLATE.format(DELIMITER=DELIMITER, drug_context=drug_context, disease_context=disease_context, question=question)
    llm_response = get_completion(prompt)
    # log.info(f"LLM Response: [{llm_response}]")
    st.write(f"LLM Response: [{llm_response}]")
    drug_disease_list: List[str] = ast.literal_eval(llm_response)
    # log.info(f"Parsed LLM Response: [{drug_disease_list}]")
    st.write(f"Parsed LLM Response: [{drug_disease_list}]")
    return drug_disease_list


INIT_EXTRACT_PROMPT_TEMPLATE = """
You are a medical assistant trained to identify names of drugs and diseases mentioned in a text.
You are given a user question given within <QUESTION></QUESTION> tag. 
Analyze the user question and extract all potential drug and disease names. 
If the sentence does not contain any drugs or diseases, return an empty Python list.

<QUESTION>
{question}
</QUESTION>

Output: A Python list of identified drug and disease names.

Your Output should return a Python List only that can be parsed correctly by the ast.literal_eval function. Account for quotes in the drug/disease name so that they wont fail the parsing.
"""

CONTEXTUAL_MATCHING_PROMPT_TEMPLATE = """
You are a medical assistant trained to validate and match names of drugs and diseases. 
You are given a medical term given within <TERM></TERM> tag.
Information to understand the medical term, process it & give output is available within <CONTEXT></CONTEXT> tag.

The medical term might be a drug or disease name but could contain spelling errors. 
Additionally, you are provided with two {DELIMITER} delimited lists: one containing possible drug names and the other containing possible disease names. 
These two lists were derived by performing a vector cosine similarity match between the embedding of the medical term and embeddings of drugs and diseases in database.
Select the most likely matching drug name or disease name or if there is no relevant match, return "".

<CONTEXT>
List of Drugs: {drug_context}

List of Diseases: {disease_context}
</CONTEXT>

<TERM>
{term}
</TERM>

Output: The best match or "". Do not output anything else

"""

# 12-11-2024
def extract_items_from_user_query_v2(question : str) -> List[str]:
    # STEP 1
    prompt: str = INIT_EXTRACT_PROMPT_TEMPLATE.format(question=question)
    llm_response = get_completion(prompt)
    # log.info(f"INIT_EXTRACT_PROMPT_TEMPLATE LLM Response: [{llm_response}]")
    # st.write(f"INIT_EXTRACT_PROMPT_TEMPLATE LLM Response: [{llm_response}]")   
    init_drug_disease_list: List[str] = ast.literal_eval(llm_response)
    drug_disease_list = sorted(list(set(init_drug_disease_list)))
    # st.markdown(f"**Initial List of Drugs and Diseases identified in User Query:** :orange[{drug_disease_list}]")

    # STEP 2
    drug_disease_list = []
    for term in init_drug_disease_list:
        drug_context, disease_context = get_context_v2(term)
        prompt: str = CONTEXTUAL_MATCHING_PROMPT_TEMPLATE.format(DELIMITER=DELIMITER, drug_context=drug_context, disease_context=disease_context, term=term)
        llm_response = get_completion(prompt)
        # log.info(f"CONTEXTUAL_MATCHING_PROMPT_TEMPLATE:\n\t\t Input:[{term}],\n\t\t Drug Context:[{drug_context}],\n\t\t Disease Context:[{disease_context}],\n\t\t Output:[{llm_response}]")
        # st.markdown(f"CONTEXTUAL_MATCHING_PROMPT_TEMPLATE:\n\t\t Input:[{term}],\n\t\t Drug Context:[{drug_context}],\n\t\t Disease Context:[{disease_context}],\n\t\t Output:[{llm_response}]")
        drug_disease_list.append(llm_response)
    
    # Ensure duplicate values are removed
    drug_disease_list = sorted(list(set(drug_disease_list)))
    
    # log.info(f"Final Drug-Disease List: [{drug_disease_list}]")
    # st.write(f"Final Drug-Disease List: [{drug_disease_list}]")
    # st.markdown(f"**Final List of Drugs and Diseases identified in User Query:** :green[{drug_disease_list}]")
    return drug_disease_list

def allValidNames(drug_disease_list):
    validity = [is_valid_name(drug) for drug in drug_disease_list]
    if False in validity:
        return False
    return True
################################## LLM Spell Correction END #############################################

# 27-11-2024: Summary Table Construction TODO
# TODO
def display_summary_table(item_label_data_dict: dict, is_option1: bool = True):
    # Input is a dict where key = drug/disease name, value: df for that drug/disease
    def create_summary_row(item_label, item_data_df):
        # Create a new DataFrame with the desired 
        if is_option1 == True:
            summary_dict = {            
                "VALUE": [item_label.upper()],  # Unique values in VALUE column concatenated
                "NPPES_NPI_COUNT": [item_data_df["NPPES_NPI"].nunique()],  # Count of all rows in the input DataFrame
                "EMAIL_COUNT": [(item_data_df["EMAIL_COUNT"] > 0).sum()],
                "NLP_COUNT": [(item_data_df["NLP_COUNT"] > 0).sum()],
                "RIDDLE_COUNT": [(item_data_df["RIDDLE_COUNT"] > 0).sum()],
                "JS_COUNT": [(item_data_df["JS_COUNT"] > 0).sum()],
                "OTHERS_COUNT": [(item_data_df["OTHERS_COUNT"] > 0).sum()],
                "NEI_APP_COUNT": [(item_data_df["NEI_APP_COUNT"] > 0).sum()]
            }
        else:
            summary_dict = {            
                "VALUE": [item_label.upper()],  # Unique values in VALUE column concatenated
                "NPPES_NPI_COUNT": [item_data_df["NPPES_NPI"].nunique()],  # Count of all rows in the input DataFrame
                "EMAIL_COUNT": [(item_data_df["TOTAL_EMAIL_COUNT"] > 0).sum()],
                "NLP_COUNT": [(item_data_df["TOTAL_NLP_COUNT"] > 0).sum()],
                "RIDDLE_COUNT": [(item_data_df["TOTAL_RIDDLE_COUNT"] > 0).sum()],
                "JS_COUNT": [(item_data_df["TOTAL_JS_COUNT"] > 0).sum()],
                "OTHERS_COUNT": [(item_data_df["TOTAL_OTHERS_COUNT"] > 0).sum()],
                "NEI_APP_COUNT": [(item_data_df["TOTAL_NEI_APP_COUNT"] > 0).sum()]
            }            
        summary_df = pd.DataFrame(summary_dict) # Convert the dictionary into a DataFrame       
        return summary_df

    
    individual_summary_df_list: list = []
    for item_label, item_data_df in item_label_data_dict.items(): # key = drug/disease label, value = dataframe of data for key
        individual_summary_df = create_summary_row(item_label, item_data_df)
        individual_summary_df_list.append(individual_summary_df)

    # Concatenate DataFrames into a single DataFrame
    combined_summary_df = pd.concat(individual_summary_df_list, ignore_index=True)
    
    st.markdown(f"***:pushpin: :blue[Summary Table for Selected Drug(s)/Disease(s)/Condition(s):]***")
    st.dataframe(combined_summary_df)
    st.markdown("**Disclaimer:** *:blue-background[The total count of providers from various sources may exceed the total Nppes_npi count, as a single provider could engage with content from multiple sources.]*")


def get_embedding(query):
    sql_stmt = f"select snowflake.cortex.embed_text_768(:1, :2) as VECTOR_EMBEDDING"
    sql_response = session.sql(sql_stmt, (EMBEDDING_MODEL, query)).collect()
    embedding = sql_response[0]["VECTOR_EMBEDDING"]
    return embedding


def is_valid_name(query):
    # Define the regex pattern to check for single words or two-word phrases
    pattern = r"^[A-Za-z0-9\-\+_&]+( [A-Za-z0-9\-\+_&]+){0,4}$"
    
    # Perform the regex match
    return bool(re.match(pattern, query))

@st.cache_data
def fetch_data(query):
    return session.sql(query).to_pandas()

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# --------------------------------- Pie Chart --------------------------------

color_pie_map = {
    'High': '#28a745',   # Green
    'Medium': '#fd7e14', # Orange
    'Low': '#dc3545'     # Red
}

def display_engagement_pie_chart(df, column):
    # Group by the specified column and count occurrences
    classes = df[column].value_counts().reset_index()
    classes.columns = ['Engagement', 'Count']
    
    # Create the pie chart with Plotly
    fig = px.pie(classes, values='Count', names='Engagement', 
                 color='Engagement', color_discrete_map=color_pie_map,
                 hole=0.3,
                 width=700,  
                 height=500)
    
    fig.update_traces(textposition='outside', textinfo='value+percent', textfont=dict(size=14), selector=dict(type='pie'))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    
    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)



# --------------------------------------- OPTION 1 ---------------------------------------------------
st.divider()
if option == "**Select the first option to find Healthcare Professionals in the HMP Via database based on selected parameters of interest.**":
    specializations = session.sql(f"""SELECT DISTINCT FIX_SPEC FROM {MASTER_TABLE} order by 1 asc""").to_pandas() 
    specializations_count = session.sql(f"""SELECT count(DISTINCT FIX_SPEC) C1 FROM {MASTER_TABLE} """).to_pandas()["C1"][0]
    col1, col2, col3, = st.columns([1,1,1])
    with st.container():
        with col1:    
            st.markdown("***Select a Specialization***")
        with col2:
            st.markdown("***Select a Drug name***")
        with col3:
            st.markdown("***Select a Disease or Condition name***")
    with st.container():
        with col1:
            selected_specialization = st.selectbox("", specializations,
                             label_visibility="collapsed",index=None,placeholder=f"{specializations_count} SPECIALIZATIONS")
        with col2:       
            drugs = session.sql(f"""SELECT DISTINCT value FROM {MASTER_TABLE} WHERE FIX_SPEC = '{selected_specialization}' and vocab = 'drug' 
                                    order by CASE
                                        WHEN (LEFT(lower(value), 1) BETWEEN 'a' AND 'z') or (LEFT(lower(value), 1) BETWEEN 'A' AND 'Z') THEN 1
                                        WHEN LEFT(value, 1) BETWEEN '0' AND '9' THEN 2
                                        ELSE 3
                                      END, lower(value) asc""").to_pandas() 
            drugs_count = session.sql(f"""SELECT count(DISTINCT lower(value)) C1 FROM {MASTER_TABLE} WHERE FIX_SPEC = '{selected_specialization}' and vocab = 'drug' """).to_pandas()["C1"][0]
            
            selected_drug = st.selectbox("",drugs,label_visibility="collapsed",index=None,placeholder=f"{drugs_count} DRUGS")
            if selected_drug:
                selected_drug_escaped = selected_drug.replace("'", "''")
        with col3:
            diseases = session.sql(f"""SELECT DISTINCT value FROM {MASTER_TABLE} WHERE FIX_SPEC = '{selected_specialization}' and vocab = 'disease'
                                        order by CASE
                                            WHEN (LEFT(lower(value), 1) BETWEEN 'a' AND 'z') or (LEFT(lower(value), 1) BETWEEN 'A' AND 'Z') THEN 1
                                            WHEN LEFT(value, 1) BETWEEN '0' AND '9' THEN 2
                                            ELSE 3
                                          END, lower(value) asc""").to_pandas()
            diseases_count = session.sql(f"""SELECT count(DISTINCT lower(value)) C2 FROM {MASTER_TABLE} WHERE FIX_SPEC = '{selected_specialization}' and vocab = 'disease'""").to_pandas()["C2"][0]
            
            selected_disease = st.selectbox("", diseases, label_visibility="collapsed",index=None,placeholder=f"{diseases_count} DISEASES")
            if selected_disease:
                selected_disease_escaped = selected_disease.replace("'", "''")
        st.divider()
                
        usertext_embed =''
        if selected_specialization:   
            button_clicked = st.button("Find Match",type="primary", use_container_width=False)
            if button_clicked:
            # ------------------------------- IF BOTH DRUG & DISEASE SELECTED --------------------------------     
                if selected_drug and selected_disease:
                    with st.spinner('Finding match ...'):
                        time.sleep(3)
                    
                    def display_provider_degree_distribution():
                        
                        st.markdown("***:pushpin: :blue[Distribution - NPI count based on distinct provider degree for the drug and disease selected]***")     
                   
                        drug_disease_npi_count= session.sql(f""" SELECT 
                            A.PROVIDER_DEGREE AS PROVIDER_DEGREE,
                            A.VALUE AS disease_name, 
                            B.Value AS drug_name,
                            COUNT(DISTINCT A.NPPES_NPI)  AS NPI_count,

                        FROM
                            (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(upper(VALUE)) AS VALUE, 
                                    TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                    PROVIDER_DEGREE 
                             FROM {MASTER_TABLE}
                             WHERE vocab = 'disease' AND fix_spec = '{selected_specialization}'
                            ) A
                        JOIN
                            (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(upper(VALUE)) AS VALUE, 
                                    TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                    PROVIDER_DEGREE 
                             FROM {MASTER_TABLE}
                             WHERE vocab = 'drug' AND fix_spec = '{selected_specialization}'
                            ) B
                        ON 
                            A.FN = B.FN
                            AND A.LN = B.LN
                            AND A.PROVIDER_DEGREE = B.PROVIDER_DEGREE  -- Ensure matching degrees if needed
                        WHERE
                            upper(A.VALUE) = upper('{selected_disease_escaped}')
                            AND upper(B.VALUE) = upper('{selected_drug_escaped}')
                        GROUP BY 
                            A.Value, 
                            B.Value, 
                            A.PROVIDER_DEGREE order by NPI_count desc ;""").to_pandas()
                        drug_disease_npi_count_top20 =drug_disease_npi_count.head(20)
                        
                        # if len(drug_disease_npi_count) ==0:
                        #     st.write(f"***:red[No Matching NPI for :blue-background[{selected_drug}] drug and :blue-background[{selected_disease}] disease and hence no summary table is displayed !!***")
                        # else:
                        col1,col2 = st.columns([1,1])
                        
                        with col1:  
                            st.write(f"*Top 20 Degrees by Count of Distinct NPIs for :blue-background[{selected_drug}] Drug and :blue-background[{selected_disease}] Disease selected*")
                            fig = px.bar(drug_disease_npi_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['DISEASE_NAME','DRUG_NAME'])
                            st.plotly_chart(fig)
                        with col2:
                            st.write(f"*Degrees by Count of Distinct NPIs for :blue-background[{selected_drug}] Drug and :blue-background[{selected_disease}] Disease selected*")
                            st.write(drug_disease_npi_count)
                        
                            # --------------- Download button ---------------------------
                            if len(drug_disease_npi_count) > 30000:
                                b64 = convert_into_csv(drug_disease_npi_count)
                                file = f"NPIs_degrees_data_for_drug_{selected_drug_escaped}_&_disease_{selected_disease_escaped}.csv"
                            else:
                                csv = convert_df(drug_disease_npi_count)
                                b64 = base64.b64encode(csv).decode('utf-8')
                                file = f"NPIs_degrees_data_for_drug_{selected_drug_escaped}_&_disease_{selected_disease_escaped}.csv"
                                                
                            href = f'''
                                    <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                        display: inline-block;
                                        font-size: 16px;
                                        color: white;
                                        background-color: #4CAF50;
                                        padding: 10px 20px;
                                        text-align: center;
                                        text-decoration: none;
                                        border-radius: 5px;
                                        margin-top: 10px;
                                    ">Download data as CSV</a>
                                    '''
                            st.markdown(href, unsafe_allow_html=True)
                    
                        
                # --------------------- FOR COMMON MATCH --------------------------------
                    selected_drug_engagement = session.sql(f""" with score as (
                                                        Select 
                                                        distinct A.NPPES_NPI,A.Value as disease_name, B.value as drug_name,A.fix_spec,A.PROVIDER_FIRST_NAME,A.PROVIDER_LAST_NAME,A.PROVIDER_ORGANIZATION_NAME,
                                                        A.PROVIDER_DEGREE, A.BUSINESS_MAILING_ADDRESS, A.PROVIDER_TELEPHONE_NUMBER, A.ENGAGED_EMAIL,A.CLASSIFICATION,A.ALL_EMAILS_BLUECONIC,
                                                        (A.EMAIL_COUNT+B.EMAIL_COUNT) as TOTAL_EMAIL_COUNT,
                                                        (A.NLP_COUNT+B.NLP_COUNT) as TOTAL_NLP_COUNT,
                                                        (A.RIDDLE_COUNT+B.RIDDLE_COUNT) AS TOTAL_RIDDLE_COUNT,
                                                        (A.JS_COUNT+B.JS_COUNT) AS TOTAL_JS_COUNT,
                                                        (A.OTHERS_COUNT+B.OTHERS_COUNT) AS TOTAL_OTHERS_COUNT,
                                                        (A.NEI_APP_COUNT+B.NEI_APP_COUNT) AS TOTAL_NEI_APP_COUNT,
                                                       (A.TOTAL_ENGAGEMENT +B.TOTAL_ENGAGEMENT) AS NPI_TOTAL_ENGAGEMENT
                                                        from
                                                        (Select NPPES_NPI, VALUE, VOCAB,fix_spec, Provider_FIRST_NAME FN, PROVIDER_LAST_NAME LN, PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                       total_count as TOTAL_ENGAGEMENT  from {MASTER_TABLE} 
                                                    where vocab = 'disease' and fix_spec ='{selected_specialization}' ) A, 
                                                    (Select NPPES_NPI,VALUE, VOCAB,fix_spec, Provider_FIRST_NAME FN, PROVIDER_LAST_NAME LN,PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                       total_count as TOTAL_ENGAGEMENT  from {MASTER_TABLE} 
                                                    where vocab = 'drug' and fix_spec ='{selected_specialization}' ) B
                                                    
                                                    Where
                                                    
                                                    A.FN=B.FN
                                                    and A.LN=B.LN
                                                    
                                                    and upper(A.value) = upper('{selected_disease_escaped}')
                                                    and upper(B.Value) = upper('{selected_drug_escaped}'))
                                                    , CTE_AGGREGATE AS
                                                        (
                                                        SELECT NPPES_NPI
                                                                , upper(disease_name) as disease_name
                                                                , upper(drug_name) as drug_name
                                                                , fix_spec
                                                                , PROVIDER_FIRST_NAME
                                                                , PROVIDER_LAST_NAME
                                                                , PROVIDER_ORGANIZATION_NAME
                                                                , PROVIDER_DEGREE
                                                                , BUSINESS_MAILING_ADDRESS
                                                                , PROVIDER_TELEPHONE_NUMBER 
                                                                , ENGAGED_EMAIL
                                                                , CLASSIFICATION
                                                                , ALL_EMAILS_BLUECONIC
                                                                , sum(TOTAL_EMAIL_COUNT) as EMAIL_COUNT
                                                                , sum(TOTAL_NLP_COUNT) as NLP_COUNT
                                                                , sum(TOTAL_RIDDLE_COUNT) as RIDDLE_COUNT
                                                                , sum(TOTAL_JS_COUNT) as JS_COUNT
                                                                , sum(TOTAL_OTHERS_COUNT) as OTHERS_COUNT
                                                                , sum(TOTAL_NEI_APP_COUNT) as NEI_APP_COUNT
                                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                                        FROM score
                                                        GROUP BY all
                                                        ),
                                                     CTE_PERCENTILE as
                                                    (select *
                                                            , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT)), 4) AS ENGAGEMENT_PERCENTILE
                                                            , case 
                                                                when NPI_TOTAL_ENGAGEMENT >= (select avg(NPI_TOTAL_ENGAGEMENT) from score) then 'High'
                                                                else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                            , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT desc) as RANK
                                                        from CTE_AGGREGATE
                                                        order by NPI_TOTAL_ENGAGEMENT desc)
                                                    
                                                    (select  cte.NPPES_NPI, disease_name,drug_name,fix_spec,
                                                                EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT,
                                                                cte.Rank as ENGAGEMENT_RANK,
                                                                cte.ENGAGEMENT_PERCENTILE
                                                                , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                                when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                                else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                                cte.H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET,
                                                                PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC       
                                                        from CTE_PERCENTILE cte  
                                                        order by NPI_TOTAL_ENGAGEMENT desc)
                                                                """).to_pandas()
                    #df_drugAndDisease = pd.DataFrame(selected_drug_engagement)

                    if len(selected_drug_engagement) == 0:
                        st.markdown(f"***:red[No Matching NPI for :blue-background[{selected_drug}] drug and :blue-background[{selected_disease}] disease and hence no result tables and plots are displayed !!]***")
                    else:
                        with st.container(border=True):
                            with st.spinner('Displaying the summary table...'):
                                combine_drug_disease_str = selected_drug_escaped + " , " + selected_disease_escaped
                                display_summary_table({combine_drug_disease_str : selected_drug_engagement})

                        st.markdown('##')
                        with st.container(border=True):
                            with st.spinner('Displaying degree distributions ...'):
                                display_provider_degree_distribution()
                        st.markdown('###')
                        
                        #-------------------------------- PLOTS - COMMON MATCH -----------------------------------
                        with st.container(border=True):
                            with st.spinner('Displaying the combined result...'):
                                st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :grey-background[{selected_drug}] drug and :grey-background[{selected_disease}] disease:]***")
                                cola, colb, colc, cold, cole = st.columns([1,4,1,4,1])
                                with colb:
                                    with st.container():
                                        st.markdown("*1. Distribution - Percentile based engagements for all NPIs*")
                                        display_engagement_pie_chart(selected_drug_engagement, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                with cold:    
                                    with st.container(): 
                                        st.markdown("*2. Distribution - Average engagements based for all NPIs*")
                                        display_engagement_pie_chart(selected_drug_engagement, 'H_L_ENGAGEMENT_BUCKET')
                                
                                st.markdown(f":memo: ***List of Top 1000 NPIs having :green[HIGH], :orange[MEDIUM] and :red[LOW] engagements for the :blue-background[{selected_drug}] drug and :blue-background[{selected_disease}] disease:***")
                                df_drugAndDisease = pd.DataFrame(selected_drug_engagement)
                                
                                data_drugAndDisease = df_drugAndDisease[:1000]
                                st.dataframe(data_drugAndDisease.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                
                                if len(df_drugAndDisease) > 30000:
                                    b64 = convert_into_csv(df_drugAndDisease)
                                    file = f"NPIdataFor_drug_{selected_drug_escaped}_&_disease_{selected_disease_escaped}.csv"
                                else:
                                    csv = convert_df(df_drugAndDisease)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIdataFor_drug_{selected_drug_escaped}_&_disease_{selected_disease_escaped}.csv"
                                                    
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                            
                    #----------------------------------- SIMILARITY MATCH FOR BOTH DRUG AND DISEASE ------------------------        
                    st.markdown('#')
                    with st.container(border=True):
                        with st.spinner('Generating similar matches...'):
                            usertext_embed = get_embedding(selected_drug_escaped)
                            usertext_disease = get_embedding(selected_disease_escaped)
        
                            
                            
                            low_engagement = session.sql(f"""with CTE_MAIN as (Select 
                                                        distinct A.NPPES_NPI,A.fix_spec, B.value as drug_name,B.SIMILARITY_drugscore as Drug_score,A.Value as disease_name,A.SIMILARITY_diseasescore as Disease_score, A.PROVIDER_FIRST_NAME,A.PROVIDER_LAST_NAME,A.PROVIDER_ORGANIZATION_NAME,
                                                        A.PROVIDER_DEGREE, A.BUSINESS_MAILING_ADDRESS, A.PROVIDER_TELEPHONE_NUMBER, A.ENGAGED_EMAIL,A.CLASSIFICATION,A.ALL_EMAILS_BLUECONIC,
                                                        (A.EMAIL_COUNT+B.EMAIL_COUNT) as EMAIL_COUNT,
                                                        (A.NLP_COUNT+B.NLP_COUNT) as NLP_COUNT,
                                                        (A.RIDDLE_COUNT+B.RIDDLE_COUNT) AS RIDDLE_COUNT,
                                                        (A.JS_COUNT+B.JS_COUNT) AS JS_COUNT,
                                                        (A.OTHERS_COUNT+B.OTHERS_COUNT) AS OTHERS_COUNT,
                                                        (A.NEI_APP_COUNT+B.NEI_APP_COUNT) AS NEI_APP_COUNT,
                                                       (A.TOTAL_ENGAGEMENT +B.TOTAL_ENGAGEMENT) AS NPI_TOTAL_ENGAGEMENT
                                                        from
                                                        (Select NPPES_NPI, VALUE, VOCAB, fix_spec,Provider_FIRST_NAME FN, PROVIDER_LAST_NAME LN, PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                       total_count as TOTAL_ENGAGEMENT,VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_disease}::VECTOR(FLOAT, 768)) as SIMILARITY_diseasescore  from {MASTER_TABLE} 
                                                    where vocab = 'disease' and fix_spec ='{selected_specialization}' ) A, 
                                                    (Select NPPES_NPI,VALUE, VOCAB,fix_spec, Provider_FIRST_NAME FN, PROVIDER_LAST_NAME LN,PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                       total_count as TOTAL_ENGAGEMENT ,VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_drugscore from {MASTER_TABLE} 
                                                    where vocab = 'drug' and fix_spec ='{selected_specialization}' ) B
                                                                    
                                        where  A.FN=B.FN
                                             and A.LN=B.LN and  (SIMILARITY_drugscore<0.96 and SIMILARITY_drugscore>=.90) and  (SIMILARITY_diseasescore<0.96 and SIMILARITY_diseasescore>=.90) order by (SIMILARITY_drugscore,SIMILARITY_diseasescore) desc)
                                        (SELECT NPPES_NPI
                                            , fix_spec
                                            , upper(drug_name) as drug_name
                                            , avg(Drug_score) as Drug_SIMILARITY_score
                                            , upper(disease_name) as disease_name
                                            , avg(Disease_score) as Disease_SIMILARITY_score
                                            , sum(EMAIL_COUNT) as EMAIL_COUNT
                                            , sum(NLP_COUNT) as NLP_COUNT
                                            , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                            , sum(JS_COUNT) as JS_COUNT
                                            , sum(OTHERS_COUNT) OTHERS_COUNT
                                            , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER
                                            , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) AS ALL_EMAILS_BLUECONIC
                                            
                                    FROM CTE_MAIN
                                    group by all order by Drug_SIMILARITY_score desc, Disease_SIMILARITY_score desc
                                    )
                                        """).to_pandas() 
                                
                                
        
                        ################################## SIMILARITY TABLES FOR ONLY DRUG & DISEASE WHEN BOTH DRUG AND DISEASE SELECTED ####################
                            low_engagement_drug = session.sql(f"""with CTE_main as (SELECT NPPES_NPI, upper(value) as value, fix_spec,
                                                                    EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,total_count as NPI_TOTAL_ENGAGEMENT,
                                                                    VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                    PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                                                    FROM {MASTER_TABLE} where SIMILARITY_score<0.96 and SIMILARITY_score>=.90 and fix_spec ='{selected_specialization}'   order by SIMILARITY_score desc)
                                        (SELECT NPPES_NPI
                                                , value
                                                , fix_spec
                                                , avg(SIMILARITY_score) as SIMILARITY_score
                                                , sum(EMAIL_COUNT) as EMAIL_COUNT
                                                , sum(NLP_COUNT) as NLP_COUNT
                                                , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                                , sum(JS_COUNT) as JS_COUNT
                                                , sum(OTHERS_COUNT) as OTHERS_COUNT
                                                , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                                , PROVIDER_FIRST_NAME
                                                , PROVIDER_LAST_NAME
                                                , PROVIDER_ORGANIZATION_NAME
                                                , PROVIDER_DEGREE
                                                , BUSINESS_MAILING_ADDRESS
                                                , PROVIDER_TELEPHONE_NUMBER
                                                , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                                , CLASSIFICATION
                                                , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                        FROM CTE_main
                                        GROUP BY ALL order by SIMILARITY_score desc)""").to_pandas()
                             # 3. Disease
                            low_engagement_disease = session.sql(f"""with CTE_main as (SELECT NPPES_NPI, upper(value) as value, fix_spec,
                                                                    EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,total_count as NPI_TOTAL_ENGAGEMENT,
                                                                    VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_disease}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                    PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                                                    FROM {MASTER_TABLE} where SIMILARITY_score<0.96 and SIMILARITY_score>=.90 and fix_spec ='{selected_specialization}'   order by SIMILARITY_score desc
                                                                    )
                                        (SELECT NPPES_NPI
                                            , value
                                            , fix_spec
                                            , avg(SIMILARITY_score) as SIMILARITY_score
                                            , sum(EMAIL_COUNT) as EMAIL_COUNT
                                            , sum(NLP_COUNT) as NLP_COUNT
                                            , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                            , sum(JS_COUNT) as JS_COUNT
                                            , sum(OTHERS_COUNT) as OTHERS_COUNT
                                            , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT                                           
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER
                                            , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                    FROM CTE_main
                                    GROUP BY ALL order by SIMILARITY_score desc)""").to_pandas()
                                            
                            
                            st.markdown(f":memo: ***List of Top 1000 NPIs having interest in drugs/diseases :green[similar] to :blue-background[{selected_drug}] drug :blue-background[{selected_disease}] disease:***")
                            st.markdown("*:fast_forward: :green-background[The semantic matches for drug and disease are based on advanced contextual analysis of medical data, considering factors like similar drugs, related diseases, therapeutic areas, and associated symptoms. Only the most relevant matches will be shown for your review.]*")
                            
                            with st.container():
                                if low_engagement.empty:
                                    st.write(f"***:red[No Matching NPI results for {selected_drug} drug and {selected_disease} disease!]***")
                                else:
                                    st.dataframe(low_engagement[:1000], hide_index=False)
        
                                    if len(low_engagement) > 30000:
                                        b64 = convert_into_csv(low_engagement)
                                        file = f"NPIdata_similarTo_drug_{selected_drug}_&_disease_{selected_disease}.csv"
                                    else:
                                        csv = convert_df(low_engagement)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIdata_similarTo_drug_{selected_drug}_&_disease_{selected_disease}.csv"
                                                                
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
        
        
                        ################################## SIMILARITY TABLE FOR ONLY DRUG WHEN BOTH DRUG AND DISEASE SELECTED ####################
                            
                            st.markdown('#')
                            st.markdown(f":memo: ***List of Top 1000 NPIs having interest in drugs :green[similar] to :blue-background[{selected_drug}] drug:***")
                            st.markdown("*:fast_forward: :green-background[The semantic matches for drug and disease are based on advanced contextual analysis of medical data, considering factors like similar drugs, related diseases, therapeutic areas, and associated symptoms. Only the most relevant matches will be shown for your review.]*")
                                                                    
                            with st.container():
                                if low_engagement_drug.empty:
                                    st.write(f"***:red[No Matching NPI results for {selected_drug} drug!]***")
                                else:
                                    # st.write(f"***:green[Matching NPI results for {selected_drug} drug!]***")
                                    data_similarDrug = low_engagement_drug[:1000]
                                    st.dataframe(data_similarDrug, hide_index=False)
                                    if len(low_engagement_drug) > 30000:
                                        b64 = convert_into_csv(low_engagement_drug)
                                        if len(b64) >= 300000:
                                            b64 = b64[:300000]
                                        file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv.gz"
                                    else:
                                        csv = convert_df(low_engagement_drug)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv"
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
                                    
                            ################################## SIMILARITY TABLE FOR ONLY DISEASE WHEN BOTH DRUG AND DISEASE SELECTED ####################
                            st.markdown('#')
                            st.markdown(f":memo: ***List of Top 1000 NPIs having interest in disease :green[similar] to :blue-background[{selected_disease}] disease:***")
                            st.markdown("*:fast_forward: :green-background[The semantic matches for drug and disease are based on advanced contextual analysis of medical data, considering factors like similar drugs, related diseases, therapeutic areas, and associated symptoms. Only the most relevant matches will be shown for your review.]*")
                            
        
                            with st.container():
                                if low_engagement_disease.empty:
                                    st.write(f"***:red[No Matching NPI results for {selected_disease} disease!]***")
                                else:
                                    # st.write(f"***:green[Matching NPI results for {selected_disease} disease !]***")
                                    data_similarDisease = low_engagement_disease[:1000]
                                    st.dataframe(data_similarDisease, hide_index=False)
                                    
                                    if len(low_engagement_disease) > 30000:
                                        b64 = convert_into_csv(low_engagement_disease)
                                        if len(b64) >= 300000:
                                            b64 = b64[:300000]
                                        file = f"NPIdataFor_disease_similarTo_{selected_disease}.csv.gz"
                                    else:
                                        csv = convert_df(low_engagement_disease)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIdataFor_disease_similarTo_{selected_disease}.csv"
                                    
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
                    st.markdown("*:red[Note: Tables with records greater than 2,00,000 will be downloaded as zip files]*")    
                    
                # ------------------------------- ONLY DRUG SELECTED --------------------------------
                
                elif selected_drug:
                    # st.session_state.button_disabled = True
                    
                    with st.spinner('Finding matches...'):
                        time.sleep(3)
                   
                    def display_provider_degree_distribution():
                            st.markdown("***:pushpin: :blue[Distribution - NPI count based on distinct provider degree for the drug selected]***")
                              
                            degree_count = session.sql(f"""select PROVIDER_DEGREE,
                                        UPPER(value) as DRUG_NAME, 
                                        count(*) as NPI_COUNT 
                                        from {MASTER_TABLE}  where UPPER(value) = UPPER('{selected_drug_escaped}')  and fix_spec = '{selected_specialization}' and vocab ='drug' group by PROVIDER_DEGREE,UPPER(value) order by NPI_COUNT desc""").to_pandas()   
                            degree_count_top20 = degree_count.head(20)
                            col1, col2 = st.columns([1,1])
                            with col1:
                                st.write(f"*Top 20 Degrees by Count of Distinct NPIs for {selected_drug} Drug*")
                                fig = px.bar(degree_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['DRUG_NAME'])
                                st.plotly_chart(fig)
                            with col2:
                                st.write(f"*Degrees by Count of Distinct NPIs for {selected_drug} Drug*")
                                st.write(degree_count)
                                # --------------- Download button ---------------------------
                                if len(degree_count) > 30000:
                                    b64 = convert_into_csv(degree_count)
                                    file = f"NPIs_degrees_data_for_drug_{selected_drug}.csv.gz"
                                else:
                                    csv = convert_df(degree_count)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIs_degrees_data_for_drug_{selected_drug}.csv"
                                                
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                    
                   
                    # # Achieving Entity Extraction & converting it to embeddings
                    usertext_embed = get_embedding(selected_drug_escaped)
                    engagement = session.sql(f"""with score as (
                                        SELECT NPPES_NPI, value,fix_spec, PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                       total_count as NPI_TOTAL_ENGAGEMENT, VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as score
                                FROM {MASTER_TABLE} where score>=0.96  and fix_spec ='{selected_specialization}' order by score desc
                                )
                                , CTE_AGGREGATE AS
                                    (
                                    SELECT NPPES_NPI
                                            , upper(VALUE) as value
                                            , fix_spec
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER 
                                            , ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , ALL_EMAILS_BLUECONIC
                                            , sum(EMAIL_COUNT) as EMAIL_COUNT
                                            , sum(NLP_COUNT) as NLP_COUNT
                                            , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                            , sum(JS_COUNT) as JS_COUNT
                                            , sum(OTHERS_COUNT) as OTHERS_COUNT
                                            , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                    FROM score
                                    GROUP BY all
                                    ),
                                                CTE_PERCENTILE as
                                                    (select *
                                                            , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT)), 4) AS ENGAGEMENT_PERCENTILE
                                                            , case 
                                                                when NPI_TOTAL_ENGAGEMENT >= (select avg(NPI_TOTAL_ENGAGEMENT) from score) then 'High'
                                                                else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                            , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT desc) as RANK
                                                        from CTE_AGGREGATE
                                                        order by NPI_TOTAL_ENGAGEMENT desc)
                                                    
                                                    (select  cte.NPPES_NPI, cte.value,cte.fix_spec,
                                                                            EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT,
                                                                             cte.Rank as ENGAGEMENT_RANK,
                                                                            cte.ENGAGEMENT_PERCENTILE
                                                                        , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                                        when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                                        else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                                        cte.H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET,
    
                                                                        PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC    
                                                        from CTE_PERCENTILE cte 
                                                        order by NPI_TOTAL_ENGAGEMENT desc)
                                   """).to_pandas()
                    
                    if len(engagement) == 0:
                        st.markdown(f"***:red[No matching NPI results for] :blue-background[{selected_drug}] :red[drug and hence no result tables and plots are displayed !!]***")
                    else:
                        df_onlyDrug = pd.DataFrame(engagement).reset_index(drop=True)          
                        with st.container(border=True):
                            with st.spinner('Displaying summary table ...'):
                                display_summary_table({selected_drug : engagement}) 
                        st.markdown('#') 
                        with st.container(border=True):
                            with st.spinner('Displaying degree distributions ...'):
                                display_provider_degree_distribution()

                        st.markdown('#')
                        #----------------------- PLOTS - DRUGS MATCH -----------------------------------
                        with st.container(border=True):
                            with st.spinner('Displaying matches ...'):
                                st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :blue-background[{selected_drug}] drug:]***")
                                cola, colb, colc, cold, cole = st.columns([1,4,1,4,1])
                                with colb:
                                    with st.container():
                                        st.markdown("*1. Distribution - Percentile based engagements for all NPIs*")
                                        display_engagement_pie_chart(engagement, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
        
                                with cold:    
                                    with st.container():      
                                        st.markdown("*2. Distribution - Average engagements based for all NPIs*")
                                        display_engagement_pie_chart(engagement, 'H_L_ENGAGEMENT_BUCKET')
        
                                st.write(f":memo: ***List of Top 1000 NPIs having :green[HIGH], :orange[MEDIUM] and :red[LOW] engagements for the :blue-background[{selected_drug}] drug:***")                        
        
                                data_onlyDrug = df_onlyDrug[:1000]
                                st.dataframe(data_onlyDrug.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
        
                                if len(df_onlyDrug) > 30000:
                                    b64 = convert_into_csv(df_onlyDrug)
                                    file = f"NPIdataFor_drug_{selected_drug}.csv.gz"
                                else:
                                    csv = convert_df(df_onlyDrug)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIdataFor_drug_{selected_drug}.csv"
                                
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                                
                    #-------------------------- SIMILARITY MATCH - ONLY DRUG ---------------------------- 
                    st.markdown("#")
                    with st.container(border=True):
                        with st.spinner('Displaying similar results ...'):
                            st.markdown(f":memo: ***List of Top 1000 NPIs having interest in drugs/diseases :green[similar] to :blue-background[{selected_drug}] drug:***")
                            st.markdown("*:fast_forward: :green-background[The semantic matches for drug and disease are based on advanced contextual analysis of medical data, considering factors like similar drugs, related diseases, therapeutic areas, and associated symptoms. Only the most relevant matches will be shown for your review.]*")
                            
                            with st.container():
                                low_engagement = session.sql(f"""with CTE_main as (SELECT NPPES_NPI, upper(value) as value, fix_spec,
                                                                    EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,total_count as NPI_TOTAL_ENGAGEMENT,
                                                                    VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                    PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                        FROM {MASTER_TABLE} where SIMILARITY_score<0.96 and SIMILARITY_score>=.90 and fix_spec ='{selected_specialization}' order by SIMILARITY_score desc)
                                        (SELECT NPPES_NPI
                                                , value
                                                , fix_spec
                                                , avg(SIMILARITY_score) as SIMILARITY_score
                                                , sum(EMAIL_COUNT) as EMAIL_COUNT
                                                , sum(NLP_COUNT) as NLP_COUNT
                                                , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                                , sum(JS_COUNT) as JS_COUNT
                                                , sum(OTHERS_COUNT) as OTHERS_COUNT
                                                , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                                , PROVIDER_FIRST_NAME
                                                , PROVIDER_LAST_NAME
                                                , PROVIDER_ORGANIZATION_NAME
                                                , PROVIDER_DEGREE
                                                , BUSINESS_MAILING_ADDRESS
                                                , PROVIDER_TELEPHONE_NUMBER
                                                , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                                , CLASSIFICATION
                                                , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                        FROM CTE_main
                                        GROUP BY ALL order by SIMILARITY_score desc)
                                        """).to_pandas()
                                #df_similarDrug = pd.DataFrame(low_engagement)     
                            
                                if low_engagement.empty:
                                    #st.write(f"***:red[No Matching NPI results for {selected_drug} drug!]***")
                                    low_engagement_new = session.sql(f"""with CTE_main as (SELECT NPPES_NPI, value, fix_spec,
                                                                    EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,total_count as NPI_TOTAL_ENGAGEMENT,
                                                                    VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                    PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                        FROM {MASTER_TABLE} where SIMILARITY_score<0.90 and SIMILARITY_score>=.80 and fix_spec ='{selected_specialization}'  order by SIMILARITY_score desc)
                                        (SELECT NPPES_NPI
                                                , upper(value) as value
                                                , fix_spec
                                                , avg(SIMILARITY_score) as SIMILARITY_score
                                                , sum(EMAIL_COUNT) as EMAIL_COUNT
                                                , sum(NLP_COUNT) as NLP_COUNT
                                                , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                                , sum(JS_COUNT) as JS_COUNT
                                                , sum(OTHERS_COUNT) as OTHERS_COUNT
                                                , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT                                               
                                                , PROVIDER_FIRST_NAME
                                                , PROVIDER_LAST_NAME
                                                , PROVIDER_ORGANIZATION_NAME
                                                , PROVIDER_DEGREE
                                                , BUSINESS_MAILING_ADDRESS
                                                , PROVIDER_TELEPHONE_NUMBER
                                                , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                                , CLASSIFICATION
                                                , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                        FROM CTE_main
                                        GROUP BY ALL order by SIMILARITY_score desc)""").to_pandas()
                                    #df_similarDrug = pd.DataFrame(low_engagement) 
                                    if low_engagement_new.empty:
                                        st.write(f"***:red[No Matching NPI results for {selected_drug} drug!]***")
                                    else:
                                        st.write(f"***:green[Matching NPI results for {selected_drug} drug having similarity score in range of .80 to .90!]***")
                                        data_similarDrug = low_engagement_new[:1000]
                                        st.dataframe(low_engagement_new, hide_index=False)
                                        
                                        if len(low_engagement_new) > 30000:
                                            b64 = convert_into_csv(low_engagement_new)
                                            file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv.gz"
                                        else:
                                            csv = convert_df(low_engagement_new)
                                            b64 = base64.b64encode(csv).decode('utf-8')
                                            file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv"
                                        
                                        href = f'''
                                                <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                    display: inline-block;
                                                    font-size: 16px;
                                                    color: white;
                                                    background-color: #4CAF50;
                                                    padding: 10px 20px;
                                                    text-align: center;
                                                    text-decoration: none;
                                                    border-radius: 5px;
                                                    margin-top: 10px;
                                                ">Download data as CSV</a>
                                                '''
                                        st.markdown(href, unsafe_allow_html=True)
             
                                else:
                                    data_similarDrug = low_engagement[:1000]
                                    st.dataframe(data_similarDrug, hide_index=False)
        
                                    if len(low_engagement) > 30000:
                                        b64 = convert_into_csv(low_engagement)
                                        file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv.gz"
                                    else:
                                        csv = convert_df(low_engagement)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIdataFor_drug_similarTo_{selected_drug}.csv"
                                    
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
                                    
                    st.markdown("*:red[Note: Tables with records greater than 2,00,000 will be downloaded as zip files]*")                           
                
                #-------------------------- ONLY DISEASE ----------------------------
                elif selected_disease:
                    with st.spinner('Finding matches...'):
                        time.sleep(3)
                    
                    def display_provider_degree_distribution():
                            st.markdown(f"***:pushpin: :blue[Distribution - NPI count based on distinct provider degree for :grey-background[{selected_disease}] disease:]***")
                              
                            degree_count = session.sql(f"""select PROVIDER_DEGREE,
                                    UPPER(value) as DISEASE_NAME, 
                                    count(*) as NPI_COUNT
                                    from {MASTER_TABLE}  where UPPER(value) = UPPER('{selected_disease_escaped}')  and fix_spec = '{selected_specialization}' and vocab ='disease' group by PROVIDER_DEGREE,UPPER(value) order by NPI_COUNT desc""").to_pandas()   
                            degree_count_top20 =degree_count.head(20)
                            col1, col2 = st.columns([1,1])
                            with col1:
                                st.write(f"Top 20 Degrees by Count of Distinct NPIs for {selected_disease} Disease")
                                fig = px.bar(degree_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['DISEASE_NAME'])
                                st.plotly_chart(fig)
                            with col2:
                                st.write(f"Degrees by Count of Distinct NPIs for {selected_disease} Disease")
                                st.write(degree_count)
                                # --------------- Download button ---------------------------
                                if len(degree_count) > 30000:
                                    b64 = convert_into_csv(degree_count)
                                    file = f"NPIs_degrees_for_disease_{selected_disease}.csv.gz"
                                else:
                                    csv = convert_df(degree_count)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIs_degrees_for_disease_{selected_disease}.csv"
                                                
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                        
                
                    
                    
                    usertext_embed = get_embedding(selected_disease)
                    
                    #------------------ ENGAGEMENT LOGIC ------------------------------
                    engagement = session.sql(f"""with score as (
                                        SELECT NPPES_NPI, value,fix_spec,PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                        EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                       total_count as NPI_TOTAL_ENGAGEMENT, VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as score
                                FROM {MASTER_TABLE} where score>=0.96 and fix_spec ='{selected_specialization}'  order by score desc
                                ), CTE_AGGREGATE AS
                                    (
                                    SELECT NPPES_NPI
                                            , upper(VALUE) as value
                                            , fix_spec
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER 
                                            , ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , ALL_EMAILS_BLUECONIC
                                            , sum(EMAIL_COUNT) as EMAIL_COUNT
                                            , sum(NLP_COUNT) as NLP_COUNT
                                            , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                            , sum(JS_COUNT) as JS_COUNT
                                            , sum(OTHERS_COUNT) as OTHERS_COUNT
                                            , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                    FROM score
                                    GROUP BY all
                                    ),
      
                                     CTE_PERCENTILE as
                                    (select *
                                            , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT)), 4) AS ENGAGEMENT_PERCENTILE
                                            , case 
                                                when NPI_TOTAL_ENGAGEMENT >= (select avg(NPI_TOTAL_ENGAGEMENT) from score) then 'High'
                                                else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                            , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT desc) as RANK
                                        from CTE_AGGREGATE
                                        order by NPI_TOTAL_ENGAGEMENT desc)
                                    
                                    (select  cte.NPPES_NPI, cte.value,cte.fix_spec,
                                                            EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT,
                                                            cte.Rank as ENGAGEMENT_RANK,
                                                            cte.ENGAGEMENT_PERCENTILE
                                                        , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                        when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                        else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                        cte.H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                      PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                        from CTE_PERCENTILE cte 
                                        order by NPI_TOTAL_ENGAGEMENT desc)""").to_pandas()
     

                    if len(engagement) == 0:
                        st.markdown(f"***:red[No matching NPI results for :blue-background[{selected_disease}] disease and hence no result tables and plots are displayed !!]***")
                    else:
                        #----------------------- PLOTS - DISEASE MATCH -----------------------------------
                        
                        with st.container(border=True):
                            with st.spinner('Generating summary results ...'):
                                display_summary_table({selected_disease : engagement})
                        st.markdown("#")
                        with st.container(border=True):
                            with st.spinner('Displaying degree distributions ...'):
                                display_provider_degree_distribution()
                        st.markdown("#")
                        with st.container(border=True):
                            with st.spinner('Displaying matches ...'):
                                st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :blue-background[{selected_disease}] disease:]***")
                                cola, colb, colc, cold, cole = st.columns([1,4,1,4,1])
                                with colb:
                                    with st.container():
                                        st.markdown("*1. Distribution - Percentile based engagements*")
                                        display_engagement_pie_chart(engagement, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
        
                                with cold:    
                                    with st.container():      
                                        # with cold:
                                        st.markdown("*2. Distribution - Average engagements based*")
                                        display_engagement_pie_chart(engagement, 'H_L_ENGAGEMENT_BUCKET')
                                        
                                st.markdown(f"***:pushpin: List of NPIs having :green[HIGH] :orange[MEDIUM] and :red[LOW] engagements for the :grey-background[{selected_disease}] disease***")
                                data_onlyDisease = engagement[:1000]
                                st.dataframe(data_onlyDisease.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
        
                                if len(engagement) > 30000:
                                    b64 = convert_into_csv(engagement)
                                    if len(b64) > 300000:
                                        b64 = b64[:300000]
                                    file = f"NPIdataFor_disease_{selected_disease}.csv.gz"
                                else:
                                    csv = convert_df(engagement)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIdataFor_disease_{selected_disease}.csv"
                                
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                    #-------------------------- SIMILARITY MATCH - ONLY DISEASE ---------------------------- 
                    st.markdown("#")
                    with st.container(border=True):
                        with st.spinner('Displaying similar results ...'):
                            st.markdown(f":memo: ***List of Top 1000 NPIs having interest in the drugs/diseases :green[similar] to :blue-background[{selected_disease}] disease:***")
                            st.markdown("*:fast_forward: :green-background[The semantic matches for drug and disease are based on advanced contextual analysis of medical data, considering factors like similar drugs, related diseases, therapeutic areas, and associated symptoms. Only the most relevant matches will be shown for your review.]*")
                            
                            low_engagement = session.sql(f""" with CTE_main as
                                                        (SELECT NPPES_NPI, upper(value) as value,fix_spec,EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                                total_count as NPI_TOTAL_ENGAGEMENT,VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                                                 
                                    FROM {MASTER_TABLE} where SIMILARITY_score<0.96 and SIMILARITY_score>=.90 and fix_spec ='{selected_specialization}'   order by SIMILARITY_score desc)
                                    (SELECT NPPES_NPI
                                                , value
                                                , fix_spec
                                                , avg(SIMILARITY_score) as SIMILARITY_score
                                                , sum(EMAIL_COUNT) as EMAIL_COUNT
                                                , sum(NLP_COUNT) as NLP_COUNT
                                                , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                                , sum(JS_COUNT) as JS_COUNT
                                                , sum(OTHERS_COUNT) as OTHERS_COUNT
                                                , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT
                                                , PROVIDER_FIRST_NAME
                                                , PROVIDER_LAST_NAME
                                                , PROVIDER_ORGANIZATION_NAME
                                                , PROVIDER_DEGREE
                                                , BUSINESS_MAILING_ADDRESS
                                                , PROVIDER_TELEPHONE_NUMBER
                                                , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                                , CLASSIFICATION
                                                , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                        FROM CTE_main
                                        GROUP BY ALL order by SIMILARITY_score desc)
                                    """).to_pandas()
        
                            if low_engagement.empty:
                                low_engagement_new = session.sql(f""" with CTE_main as
                                                        (SELECT NPPES_NPI, value,fix_spec,EMAIL_COUNT,NLP_COUNT,RIDDLE_COUNT,JS_COUNT,OTHERS_COUNT,NEI_APP_COUNT,
                                                                total_count as NPI_TOTAL_ENGAGEMENT,VECTOR_COSINE_SIMILARITY(embedding_value,{usertext_embed}::VECTOR(FLOAT, 768)) as SIMILARITY_score,
                                                                PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC,
                                    FROM {MASTER_TABLE} where SIMILARITY_score<0.90 and SIMILARITY_score>=.80 and fix_spec ='{selected_specialization}'  order by SIMILARITY_score desc)
                                    (SELECT NPPES_NPI
                                                , value
                                                , fix_spec
                                                , avg(SIMILARITY_score) as SIMILARITY_score
                                                , sum(EMAIL_COUNT) as EMAIL_COUNT
                                                , sum(NLP_COUNT) as NLP_COUNT
                                                , sum(RIDDLE_COUNT) as RIDDLE_COUNT
                                                , sum(JS_COUNT) as JS_COUNT
                                                , sum(OTHERS_COUNT) as OTHERS_COUNT
                                                , sum(NEI_APP_COUNT) as NEI_APP_COUNT
                                                , sum(NPI_TOTAL_ENGAGEMENT) as NPI_TOTAL_ENGAGEMENT 
                                                , PROVIDER_FIRST_NAME
                                                , PROVIDER_LAST_NAME
                                                , PROVIDER_ORGANIZATION_NAME
                                                , PROVIDER_DEGREE
                                                , BUSINESS_MAILING_ADDRESS
                                                , PROVIDER_TELEPHONE_NUMBER
                                                , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                                , CLASSIFICATION
                                                , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                        FROM CTE_main
                                        GROUP BY ALL  order by SIMILARITY_score desc)
                                    """).to_pandas()

                                if low_engagement_new.empty:
                                    st.markdown(f"***:red[No matching NPI results similar to :blue-background[{selected_disease}] disease!]***")
                                else:
                                    st.write(f"***:green[Matching NPI results for {selected_disease} drug having similarity score in range of .80 to .90!]***")
                                    data_similarDisease = low_engagement_new[:1000]
                                    st.dataframe(data_similarDisease, hide_index=False)
                                    
                                    if len(low_engagement_new) > 30000:
                                        b64 = convert_into_csv(low_engagement_new)
                                        if len(b64) > 300000:
                                            b64 = b64[:300000]
                                        file = f"NPIdataFor_disease_similarTo_{selected_disease_escaped}.csv.gz"
                                    else:
                                        csv = convert_df(low_engagement_new)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIdataFor_disease_similarTo_{selected_disease_escaped}.csv"
                                
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
                            else:
                                data_similarDisease = low_engagement[:1000]
                                st.dataframe(data_similarDisease, hide_index=False)
                                
                                if len(low_engagement) > 30000:
                                    b64 = convert_into_csv(low_engagement)
                                    if len(b64) > 300000:
                                        b64 = b64[:300000]
                                    file = f"NPIdataFor_disease_similarTo_{selected_disease_escaped}.csv.gz"
                                else:
                                    csv = convert_df(low_engagement)
                                    b64 = base64.b64encode(csv).decode('utf-8')
                                    file = f"NPIdataFor_disease_similarTo_{selected_disease_escaped}.csv"
                                    
                                href = f'''
                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                            display: inline-block;
                                            font-size: 16px;
                                            color: white;
                                            background-color: #4CAF50;
                                            padding: 10px 20px;
                                            text-align: center;
                                            text-decoration: none;
                                            border-radius: 5px;
                                            margin-top: 10px;
                                        ">Download data as CSV</a>
                                        '''
                                st.markdown(href, unsafe_allow_html=True)
                                
                    st.markdown("*:red[Note: Tables with records greater than 2,00,000 will be downloaded as zip files]*")    
                else:
                    st.session_state.button_disabled = False
                    st.markdown("***:red[Please select a DRUG or a DISEASE!]***")
                    

# --------------------------------------- OPTION 2 ---------------------------------------------------

elif option == "**Select the second option to enter your own open-text query about the Healthcare Professionals in the HMP Via database.**":

    st.write("""***Enter your query here with drug or disease name (upto 3):***""")

    max_chars = 250

    user_query = st.text_input("Enter prompt", 
                            placeholder="Please enter your query using drug or disease name (upto 3) to identify medical practitioners",
                            label_visibility="collapsed", 
                              max_chars=max_chars)

    submit = st.button("Find Match", type="primary",use_container_width=False)
    st.divider()
        
    # Main code starts
    if submit:
        with st.spinner("Finding Match ..."):
            try:       
                # sanitized_user_query = perform_query_sanitization(user_query)
                # sanitized_user_query= (sanitized_user_query.strip().lower())
                drug_disease_list = extract_items_from_user_query_v2(user_query)
                if len(drug_disease_list) == 0:
                    st.markdown("***:red[No drug/disease identified in the query.]***")    
                elif len(drug_disease_list) > 3:
                    st.markdown("***:red[Please enter atmost 3 values!]***")
                
                # if is_valid_name(user_query):
                elif allValidNames(drug_disease_list): # FIXME this needs to be changed to reflect multi item
                    if user_query == 'error': # FIXME this needs to be changed to reflect multi item
                        st.write("***:red[User Query is either out of scope or does not have sufficient information to retrieve records]***")
                        sys.exit(0)
                    else:
                        usertext_embed = user_query # FIXME Check if this is required or not
                        
                        st.write(f"Drug/Disease Names extracted from user query: {drug_disease_list}")
                    
                        query = ''    
                        if len(drug_disease_list) == 1:
                            def display_provider_degree_distribution():
                                
                                st.markdown("***:pushpin: :blue[Distribution -  NPI count based on distinct provider degree for the drugs/diseases selected]***")
                                #value1_degree_count_top20 = session.sql(f"""select PROVIDER_DEGREE,UPPER(value) as VALUE, count(*) as NPI_COUNT from {MASTER_TABLE}  where UPPER(value) like (UPPER('%{drug_disease_list[0]}%'))  group by PROVIDER_DEGREE,UPPER(value) order by NPI_COUNT desc limit 20""").to_pandas()    
                                value1_degree_count = session.sql(f"""select PROVIDER_DEGREE,
                                                UPPER(value) as VALUE, 
                                                count(*) as NPI_COUNT 
                                                from {MASTER_TABLE}  where UPPER(value) like (UPPER('%{drug_disease_list[0]}%'))  group by PROVIDER_DEGREE,UPPER(value) order by NPI_COUNT desc""").to_pandas()   
                                value1_degree_count_top20 = value1_degree_count.head(20)
                                
                                
                                col1,col2 = st.columns([1,1])
                                with col1:  
                                    st.write(f"*Top 20 Degrees by Count of Distinct NPIs for :blue-background[{drug_disease_list[0]}]*")
                                    fig = px.bar(value1_degree_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['VALUE'])
                                    st.plotly_chart(fig)
                                with col2:
                                    st.write(f"*Degrees by Count of Distinct NPIs :blue-background[{drug_disease_list[0]}]*")
                                    st.write(value1_degree_count)
                                    # --------------- Download button ---------------------------
                                    if len(value1_degree_count) > 30000:
                                        b64 = convert_into_csv(value1_degree_count)
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}.csv.gz"
                                    else:
                                        csv = convert_df(value1_degree_count)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}.csv"
                                                    
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)

                            
                            query1 = f""" with score as(SELECT NPPES_NPI,
                                                LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                                LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                                LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                                sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                                sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                                sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                                PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                    FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[0]}%' 
                                    group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                    order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                    CTE_AGGREGATED as 
                                    (
                                    SELECT nppes_npi
                                            , ALL_VALUES
                                            , ALL_DISTINCT_TERMS
                                            , ALL_DISTINCT_SPECIALITY
                                            , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                            , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                            , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                            , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                            , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                            , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER
                                            , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                    FROM score
                                    GROUP BY ALL
                                ),
                                    
                                             CTE_PERCENTILE as
                                            (select *
                                                    , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                    , case 
                                                        when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                        else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                    , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                                from CTE_AGGREGATED
                                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                            
                                            (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                    TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                    RANK as ENGAGEMENT_RANK,
                                                                    ENGAGEMENT_PERCENTILE
                                                                , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                                when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                                else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                                H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                              PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                        PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                                from CTE_PERCENTILE  
                                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            
                            df1 = fetch_data(query1)
                     
                            if len(df1) == 0:
                                st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[0]}] and hence no result tables and plots are displayed !!]***")
                            else:
                                
                                with st.container(border=True):
                                    with st.spinner('Generating summary results ...'):
                                        display_summary_table({drug_disease_list[0] : df1}, is_option1=False)
                                st.markdown("#")
                                with st.container(border=True):
                                    with st.spinner('Displaying degree distributions ...'):
                                        display_provider_degree_distribution()

                                st.markdown('#')
                                
                                #----------------------- PLOTS - DRUGS MATCH -----------------------------------
                                #top_match1_sf = session.create_dataframe(top_match1)
                                
                                with st.container(border=True):
                                    with st.spinner('Displaying match results ...'):
                                        st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :blue-background[{drug_disease_list[0]}]:]***")
                                        
                                        col1, col2, col3, col4, col5 = st.columns([1,4,1,4,1])
                                        with col2:
                                            with st.container():
                                                st.markdown("*1. Distribution - Percentile based engagements for all NPIs*")
                                                display_engagement_pie_chart(df1, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                        
                                        with col4:
                                            with st.container():
                                                st.markdown("*2. Distribution - Average engagements based for all NPIs*")
                                                display_engagement_pie_chart(df1, 'H_L_ENGAGEMENT_BUCKET')
                                                
                                        st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[0]}]:***")
                                        data1 = df1[:1000]
                                        st.dataframe(data1.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                            
                                        # -----------------------------------
                                        if len(df1) > 30000:
                                            b64 = convert_into_csv(df1)
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv.gz"
                                        else:
                                            csv_df1 = convert_df(df1)
                                            b64 = base64.b64encode(csv_df1).decode('utf-8')
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv"
                                            
                                        href = f'''
                                                <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                    display: inline-block;
                                                    font-size: 16px;
                                                    color: white;
                                                    background-color: #4CAF50;
                                                    padding: 10px 20px;
                                                    text-align: center;
                                                    text-decoration: none;
                                                    border-radius: 5px;
                                                    margin-top: 10px;
                                                ">Download data as CSV</a>
                                                '''
                                        st.markdown(href, unsafe_allow_html=True)
                                
                        elif len(drug_disease_list) == 2:
                            def display_provider_degree_distribution():
                                st.markdown(f"***:pushpin: :blue[Distribution - NPI count based on distinct provider degree for :grey-background[{drug_disease_list[0]}] and :grey-background[{drug_disease_list[1]}]]***")
                                
                                drug_disease_npi_count= session.sql(f""" SELECT 
                                                A.PROVIDER_DEGREE AS PROVIDER_DEGREE,
                                                UPPER(A.value) AS Value_1, 
                                                UPPER(B.value) AS Value_2,
                                                COUNT(DISTINCT A.NPPES_NPI)  AS NPI_count        
                                                FROM
                                                (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(VALUE) AS VALUE, 
                                                        TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                                        PROVIDER_DEGREE 
                                                 FROM {MASTER_TABLE}      
                                                ) A
                                            JOIN
                                                (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(VALUE) AS VALUE, 
                                                        TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                                        PROVIDER_DEGREE 
                                                 FROM {MASTER_TABLE}
                                                ) B
                                            ON 
                                                A.FN = B.FN
                                                AND A.LN = B.LN
                                                AND A.PROVIDER_DEGREE = B.PROVIDER_DEGREE  
                                            WHERE
                                                upper(A.value) = upper('{drug_disease_list[0]}') and upper(B.value) = upper('{drug_disease_list[1]}')
                                            GROUP BY 
                                               UPPER(A.value), 
                                                UPPER(B.value), 
                                                A.PROVIDER_DEGREE order by NPI_COUNT desc;""").to_pandas()
                                
                                drug_disease_npi_count_top20 = drug_disease_npi_count.head(20)
                                
                                col1,col2 = st.columns([1,1])
                            
                                with col1:  
                                    st.write(f"Top 20 Degrees by Count of Distinct NPIs for :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}]")
                                    fig = px.bar(drug_disease_npi_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['VALUE_1', 'VALUE_2'])
                                    st.plotly_chart(fig)
                                with col2:
                                    st.write(f"Degrees by Count of Distinct NPIs for :blue-background[{drug_disease_list[0]}] and  :blue-background[{drug_disease_list[1]}]")
                                    st.write(drug_disease_npi_count)
                                
                                    # --------------- Download button ---------------------------
                                    if len(drug_disease_npi_count) > 30000:
                                        b64 = convert_into_csv(drug_disease_npi_count)
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}_&_{drug_disease_list[1]}.csv.gz"
                                    else:
                                        csv = convert_df(drug_disease_npi_count)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}_&_{drug_disease_list[1]}.csv"
                                                    
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)

                            ####################################################################################
                            query2 = f""" with CTE as(
                                 SELECT * FROM {MASTER_TABLE}
                                where lower(value) = lower('{drug_disease_list[0]}') order by TOTAL_COUNT desc),
                                
                                CTE1 as ( SELECT *                                 
                                FROM {MASTER_TABLE}
                                where lower(value) = lower('{drug_disease_list[1]}')
                                order by TOTAL_COUNT desc) 
                                ,cte2 as
                                (select CTE.nppes_npi,
                                    UPPER(CTE.value) as Value_1 ,
                                    UPPER(cte1.value) as  Value_2,
                                    COALESCE(cte.VOCAB,cte1.vocab) AS ALL_DISTINCT_TERMS,
                                    COALESCE(cte.FIX_SPEC,cte1.FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                    (cte.EMAIL_COUNT+cte1.EMAIL_COUNT) as TOTAL_EMAIL_COUNT,
                                    (cte.NLP_COUNT+cte1.NLP_COUNT) as TOTAL_NLP_COUNT,
                                    (cte.RIDDLE_COUNT+cte1.RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,
                                    (cte.JS_COUNT+cte1.JS_COUNT) as TOTAL_JS_COUNT,
                                    (cte.OTHERS_COUNT +cte1.OTHERS_COUNT) as TOTAL_OTHERS_COUNT,      
                                    (cte.NEI_APP_COUNT +cte1.NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                    (cte.total_count +cte1.total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                    cte.PROVIDER_FIRST_NAME,
                                    cte.PROVIDER_LAST_NAME,
                                    cte.PROVIDER_ORGANIZATION_NAME,
                                    cte.PROVIDER_DEGREE, 
                                    cte.BUSINESS_MAILING_ADDRESS, 
                                    cte.PROVIDER_TELEPHONE_NUMBER, 
                                    cte.ENGAGED_EMAIL as ENGAGED_EMAIL_Value_1,
                                    cte1.ENGAGED_EMAIL as ENGAGED_EMAIL_Value_2,
                                    cte.CLASSIFICATION,
                                    cte.ALL_EMAILS_BLUECONIC
                                from cte 
                                INNER JOIN cte1 ON CTE.nppes_npi = CTE1.nppes_npi
                                ), 
                                CTE_AGGREGATED as 
                                (
                                SELECT nppes_npi
                                        , Value_1
                                        , Value_2
                                        , ALL_DISTINCT_TERMS
                                        , ALL_DISTINCT_SPECIALITY
                                        , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                        , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                        , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                        , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                        , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                        , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                        , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                        , PROVIDER_FIRST_NAME
                                        , PROVIDER_LAST_NAME
                                        , PROVIDER_ORGANIZATION_NAME
                                        , PROVIDER_DEGREE
                                        , BUSINESS_MAILING_ADDRESS
                                        , PROVIDER_TELEPHONE_NUMBER
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL_Value_1), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL_Value_1)) AS ENGAGED_EMAIL_Value_1
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL_Value_2), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL_Value_2)) AS ENGAGED_EMAIL_Value_2
                                        , CLASSIFICATION
                                        , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                FROM cte2
                                GROUP BY ALL
                                ),
                                
                                CTE_PERCENTILE as
                                (select *
                                        , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                        , case 
                                            when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from cte2) then 'High'
                                            else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                        , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                    from CTE_AGGREGATED
                                    order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                
                                (select NPPES_NPI, Value_1, Value_2,ALL_DISTINCT_SPECIALITY,
                                                        TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                        RANK as ENGAGEMENT_RANK,
                                                        ENGAGEMENT_PERCENTILE
                                                    , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                    when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                    else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                    H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL_Value_1,ENGAGED_EMAIL_Value_2,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                    from CTE_PERCENTILE  
                                    order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                """
                            ###########################################################################################
                            sub_query2_1 = f""" with score as(SELECT NPPES_NPI,
                                           LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                            LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                            LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                            sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                            sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                            sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                            PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[0]}%'
                                group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                CTE_AGGREGATED as 
                                    (
                                    SELECT nppes_npi
                                            , ALL_VALUES
                                            , ALL_DISTINCT_TERMS
                                            , ALL_DISTINCT_SPECIALITY
                                            , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                            , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                            , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                            , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                            , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                            , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER
                                            , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                    FROM score
                                    GROUP BY ALL
                                ),
                                         CTE_PERCENTILE as
                                        (select *
                                                , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                , case 
                                                    when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                    else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                            from CTE_AGGREGATED
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                        
                                        (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                RANK as ENGAGEMENT_RANK,
                                                                ENGAGEMENT_PERCENTILE
                                                            , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                            when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                            else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                            H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                          PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                            from CTE_PERCENTILE  
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            
                            sub_query2_2 = f"""with score as(SELECT NPPES_NPI,
                                            LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                            LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                            LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                            sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                            sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                            sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                            PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[1]}%'
                                group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                CTE_AGGREGATED as 
                                    (
                                    SELECT nppes_npi
                                            , ALL_VALUES
                                            , ALL_DISTINCT_TERMS
                                            , ALL_DISTINCT_SPECIALITY
                                            , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                            , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                            , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                            , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                            , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                            , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                            , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                            , PROVIDER_FIRST_NAME
                                            , PROVIDER_LAST_NAME
                                            , PROVIDER_ORGANIZATION_NAME
                                            , PROVIDER_DEGREE
                                            , BUSINESS_MAILING_ADDRESS
                                            , PROVIDER_TELEPHONE_NUMBER
                                            , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                            , CLASSIFICATION
                                            , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                    FROM score
                                    GROUP BY ALL
                                ), 
                                         CTE_PERCENTILE as
                                        (select *
                                                , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                , case 
                                                    when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                    else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                            from CTE_AGGREGATED
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                        
                                        (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                RANK as ENGAGEMENT_RANK,
                                                                ENGAGEMENT_PERCENTILE
                                                            , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                            when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                            else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                            H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                          PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                            from CTE_PERCENTILE  
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            
                
                            df2 = fetch_data(query2)
                            df2_1 = fetch_data(sub_query2_1)
                            df2_2 = fetch_data(sub_query2_2)
                            
                            

                            if len(df2) == 0 and len(df2_1) == 0 and len(df2_2) == 0:
                                st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}], hence no result tables or plots displayed!]***")
                            else: 
                                with st.container(border=True):
                                    with st.spinner('Generating summary results ...'):
                                        drug_disease_combined_str = ",".join(drug_disease_list)
                                        display_summary_table({drug_disease_combined_str : df2, drug_disease_list[0] : df2_1, drug_disease_list[1] : df2_2}, is_option1=False)
                                st.markdown("#")
                                with st.container(border=True):
                                    with st.spinner('Displaying degree distributions ...'):
                                        display_provider_degree_distribution()

                                st.markdown('#')
                                #----------------------- PLOTS - DRUGS/DISEASE MATCH -----------------------------------
                                with st.container(border=True):
                                    with st.spinner('Displaying match results ...'):
                                        st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}]:]***")                           
                                        col1, col2, col3, col4, col5 = st.columns([1,4,1,4,1])
                                        with col2:
                                            with st.container():
                                                st.markdown("*1. Distribution - Percentile based engagements for all NPIs*")
                                                display_engagement_pie_chart(df2, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                        
                                        with col4:
                                            with st.container():
                                                st.markdown("*2. Distribution - Average engagements based for all NPIs*")
                                                display_engagement_pie_chart(df2, 'H_L_ENGAGEMENT_BUCKET')
                                                
                                        st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}]:***")
                                        data2 = df2[:1000]
                                        st.dataframe(data2.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                        if len(df2) > 30000:
                                            b64 = convert_into_csv(df2)
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}_&_{drug_disease_list[1]}.csv.gz"
                                        else:
                                            csv_df2 = convert_df(df2)
                                            b64 = base64.b64encode(csv_df2).decode('utf-8')
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}_&_{drug_disease_list[1]}.csv"
                                            
                                        href = f'''
                                                <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                    display: inline-block;
                                                    font-size: 16px;
                                                    color: white;
                                                    background-color: #4CAF50;
                                                    padding: 10px 20px;
                                                    text-align: center;
                                                    text-decoration: none;
                                                    border-radius: 5px;
                                                    margin-top: 10px;
                                                ">Download data as CSV</a>
                                                '''
                                        st.markdown(href, unsafe_allow_html=True)
                            st.markdown("#")
                                
                            colxx, colyy, colzz = st.columns([1,1,1])
                            with colyy:
                                st.markdown("**:red[Results for Individual Value]**")
                                st.text("")
                
                            # ------------ Sub-query 2_1 ------------------------------------
                            
                            
                            with st.container(border=True):
                                with st.spinner('Displaying individual results ...'):
                                    if len(df2_1) == 0:
                                        st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[0]}]!]***")
                                    else:
                                        with st.container():
                                            cola, colb = st.columns([1,1])
                                            with cola:
                                                #top_match2_1_sf = session.create_dataframe(top_match2_1)
                                                st.markdown(f"*Distribution - Percentile based engagements for :blue-background[{drug_disease_list[0]}]*")
                                                display_engagement_pie_chart(df2_1, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                            with colb:    
                                                st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[0]}]:***")
                                                data2_1 = df2_1[:1000]
                                                st.dataframe(data2_1.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                                if len(df2_1) > 30000:
                                                    b64 = convert_into_csv(df2_1)
                                                    file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv.gz"
                                                else:
                                                    csv_df2_1 = convert_df(df2_1)
                                                    b64 = base64.b64encode(csv_df2_1).decode('utf-8')
                                                    file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv"
                                                    
                                                href = f'''
                                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                            display: inline-block;
                                                            font-size: 16px;
                                                            color: white;
                                                            background-color: #4CAF50;
                                                            padding: 10px 20px;
                                                            text-align: center;
                                                            text-decoration: none;
                                                            border-radius: 5px;
                                                            margin-top: 10px;
                                                        ">Download data as CSV</a>
                                                        '''
                                                st.markdown(href, unsafe_allow_html=True)
                       
                                    # ------------ Sub-query 2_2 ------------------------------------                                 
                                    if len(df2_2) == 0:
                                        st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[1]}]!]***")
                                    else:
                                        with st.container():
                                            colc, cold = st.columns([1,1])
                                            with colc:
                                                st.markdown(f"*Distribution - Percentile based engagements for :blue-background[{drug_disease_list[1]}]*")
                                                #top_match2_2_sf = session.create_dataframe(top_match2_2)
                                                display_engagement_pie_chart(df2_2, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                            with cold:
                                                st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[1]}]:***")
                                                st.dataframe(df2_2[:1000].style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                                if len(df2_2) > 30000:
                                                    b64 = convert_into_csv(df2_2)
                                                    file = f"NPIdataFor_Value_{drug_disease_list[1]}.csv.gz"
                                                else:
                                                    csv_df2_2 = convert_df(df2_2)
                                                    b64 = base64.b64encode(csv_df2_2).decode('utf-8')
                                                    file = f"NPIdataFor_Value_{drug_disease_list[1]}.csv"
                                                    
                                                href = f'''
                                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                            display: inline-block;
                                                            font-size: 16px;
                                                            color: white;
                                                            background-color: #4CAF50;
                                                            padding: 10px 20px;
                                                            text-align: center;
                                                            text-decoration: none;
                                                            border-radius: 5px;
                                                            margin-top: 10px;
                                                        ">Download data as CSV</a>
                                                        '''
                                                st.markdown(href, unsafe_allow_html=True)
                        elif len(drug_disease_list) == 3:
                            def display_provider_degree_distribution():
                                st.markdown(f"***:pushpin: :blue[Distribution -  NPI count based on distinct provider degree for :grey-background[{drug_disease_list[0]}], :grey-background[{drug_disease_list[1]}] and :grey-background[{drug_disease_list[2]}]]***")
                                
                                drug_disease_npi_count= session.sql(f""" SELECT 
                                                A.PROVIDER_DEGREE AS PROVIDER_DEGREE,
                                                UPPER(A.value) AS Value_1, 
                                                UPPER(B.value) AS Value_2,
                                                UPPER(C.value) AS Value_3,
                                                COUNT(DISTINCT A.NPPES_NPI)  AS NPI_count   
                                            FROM
                                                (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(VALUE) AS VALUE, 
                                                        TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                                        PROVIDER_DEGREE 
                                                 FROM {MASTER_TABLE}      
                                                ) A
                                            JOIN
                                                (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(VALUE) AS VALUE, 
                                                        TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                                        PROVIDER_DEGREE 
                                                 FROM {MASTER_TABLE}
                                                ) B
                                            ON 
                                                A.FN = B.FN
                                                AND A.LN = B.LN
                                                AND A.PROVIDER_DEGREE = B.PROVIDER_DEGREE
                                            JOIN
                                                (SELECT TRIM(NPPES_NPI) AS NPPES_NPI, TRIM(VALUE) AS VALUE, 
                                                        TRIM(Provider_FIRST_NAME) AS FN, TRIM(PROVIDER_LAST_NAME) AS LN, 
                                                        PROVIDER_DEGREE 
                                                 FROM {MASTER_TABLE}
                                                ) C
                                            ON 
                                                A.FN = C.FN
                                                AND A.LN = C.LN
                                                AND A.PROVIDER_DEGREE = C.PROVIDER_DEGREE  -- Ensure matching degrees if needed
                                            WHERE
                                                upper(A.value) = upper('{drug_disease_list[0]}') and upper(B.value) = upper('{drug_disease_list[1]}') and upper(c.value) = upper('{drug_disease_list[2]}')
                                            GROUP BY 
                                               UPPER(A.value), 
                                                UPPER(B.value), 
                                                UPPER(C.value),
                                                A.PROVIDER_DEGREE order by NPI_COUNT desc;""").to_pandas()
                                
                                drug_disease_npi_count_top20 =drug_disease_npi_count.head(20)
                                
                               
                                col1,col2 = st.columns([1,1])
                                
                                with col1:  
                                    st.write(f"*Top 20 Degrees by Count of Distinct NPIs for :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}] and :blue-background[{drug_disease_list[2]}]*")
                                    fig = px.bar(drug_disease_npi_count_top20, x='PROVIDER_DEGREE', y='NPI_COUNT',hover_data=['VALUE_1', 'VALUE_2','VALUE_3'])
                                    st.plotly_chart(fig)
                                with col2:
                                    st.write(f"*Degrees by Count of Distinct NPIs for :blue-background[{drug_disease_list[0]}] and :blue-background[{drug_disease_list[1]}] and :blue-background[{drug_disease_list[2]}]*")
                                    st.write(drug_disease_npi_count)
                                
                                    # --------------- Download button ---------------------------
                                    if len(drug_disease_npi_count) > 30000:
                                        b64 = convert_into_csv(drug_disease_npi_count)
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}_&_{drug_disease_list[1]}_&_{drug_disease_list[2]}.csv.gz"
                                    else:
                                        csv = convert_df(drug_disease_npi_count)
                                        b64 = base64.b64encode(csv).decode('utf-8')
                                        file = f"NPIs_degrees_for_{drug_disease_list[0]}_&_{drug_disease_list[1]}_&_{drug_disease_list[2]}.csv"
                                                    
                                    href = f'''
                                            <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                display: inline-block;
                                                font-size: 16px;
                                                color: white;
                                                background-color: #4CAF50;
                                                padding: 10px 20px;
                                                text-align: center;
                                                text-decoration: none;
                                                border-radius: 5px;
                                                margin-top: 10px;
                                            ">Download data as CSV</a>
                                            '''
                                    st.markdown(href, unsafe_allow_html=True)
                            ########################################################################
                            
                            query3 = f""" with CTE as(
                                 SELECT * 
                                FROM {MASTER_TABLE}
                                        where lower(value) = lower('{drug_disease_list[0]}') 
                                        order by TOTAL_COUNT desc),
                                        CTE1 as ( SELECT *                                 
                                FROM {MASTER_TABLE}
                                        where lower(value) = lower('{drug_disease_list[1]}')
                                        order by TOTAL_COUNT desc)
                                        , CTE2 as ( SELECT *                                 
                                FROM {MASTER_TABLE}
                                        where lower(value) = lower('{drug_disease_list[2]}')
                                        order by TOTAL_COUNT desc)
                                        ,
                                cte3 as
                                (select  CTE.nppes_npi,upper(CTE.value) as VALUE_1 ,upper(cte1.value) as VALUE_2, upper(cte2.value) as  VALUE_3,
                                 COALESCE(cte.VOCAB,cte1.vocab,cte2.vocab) AS ALL_DISTINCT_TERMS,
                                    COALESCE(cte.FIX_SPEC, cte1.FIX_SPEC, cte2.FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                    (cte.EMAIL_COUNT+cte1.EMAIL_COUNT +cte2.EMAIL_COUNT) as TOTAL_EMAIL_COUNT,
                                    (cte.NLP_COUNT+cte1.NLP_COUNT+cte2.NLP_COUNT) as TOTAL_NLP_COUNT,
                                    (cte.RIDDLE_COUNT+cte1.RIDDLE_COUNT+cte2.RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,
                                    (cte.JS_COUNT+cte1.JS_COUNT+cte2.JS_COUNT) as TOTAL_JS_COUNT,
                                    (cte.OTHERS_COUNT +cte1.OTHERS_COUNT+cte2.OTHERS_COUNT) as TOTAL_OTHERS_COUNT,
                                    (cte.NEI_APP_COUNT +cte1.NEI_APP_COUNT+cte2.NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                    (cte.total_count +cte1.total_count+cte2.total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                    cte.PROVIDER_FIRST_NAME,cte.PROVIDER_LAST_NAME,cte.PROVIDER_ORGANIZATION_NAME,
                                    cte.PROVIDER_DEGREE, cte.BUSINESS_MAILING_ADDRESS, cte.PROVIDER_TELEPHONE_NUMBER, 
                                    cte.ENGAGED_EMAIL as ENGAGED_EMAIL_VALUE_1,
                                    cte1.ENGAGED_EMAIL as ENGAGED_EMAIL_VALUE_2,
                                    cte2.ENGAGED_EMAIL as ENGAGED_EMAIL_VALUE_3,
                                    cte.CLASSIFICATION,cte.ALL_EMAILS_BLUECONIC
                                from cte 
                                INNER JOIN cte1 ON CTE.nppes_npi = CTE1.nppes_npi INNER JOIN cte2 ON CTE.nppes_npi = CTE2.nppes_npi
                                ),                                   
                                CTE_AGGREGATED as 
                                (
                                SELECT nppes_npi
                                        , VALUE_1
                                        , VALUE_2
                                        , VALUE_3
                                        , ALL_DISTINCT_TERMS
                                        , ALL_DISTINCT_SPECIALITY
                                        , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                        , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                        , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                        , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                        , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                        , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                        , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                        , PROVIDER_FIRST_NAME
                                        , PROVIDER_LAST_NAME
                                        , PROVIDER_ORGANIZATION_NAME
                                        , PROVIDER_DEGREE
                                        , BUSINESS_MAILING_ADDRESS
                                        , PROVIDER_TELEPHONE_NUMBER
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL_VALUE_1), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL_VALUE_1)) AS ENGAGED_EMAIL_VALUE_1
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL_VALUE_2), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL_VALUE_2)) AS ENGAGED_EMAIL_VALUE_2
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL_VALUE_3), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL_VALUE_3)) AS ENGAGED_EMAIL_VALUE_3
                                        , CLASSIFICATION
                                        , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                FROM cte3
                                GROUP BY ALL
                                ),
                                 CTE_PERCENTILE as
                                (select *
                                        , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                        , case 
                                            when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from cte3) then 'High'
                                            else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                        , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                    from CTE_AGGREGATED
                                    order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                
                                (select NPPES_NPI, VALUE_1,VALUE_2,VALUE_3,
                                ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                    TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                        RANK as ENGAGEMENT_RANK,
                                                        ENGAGEMENT_PERCENTILE
                                                    , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                    when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                    else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                    H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL_VALUE_1,ENGAGED_EMAIL_VALUE_2,ENGAGED_EMAIL_VALUE_3,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                    from CTE_PERCENTILE  
                                    order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            # ------------------ Sub-query 3_1 -------------------------------------
                            sub_query3_1 = f"""with score as(SELECT NPPES_NPI,
                                            LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                            LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                            LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                            sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                            sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                            sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                            PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[0]}%'
                                group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                CTE_AGGREGATED as 
                                (
                                SELECT nppes_npi
                                        , ALL_VALUES
                                        , ALL_DISTINCT_TERMS
                                        , ALL_DISTINCT_SPECIALITY
                                        , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                        , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                        , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                        , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                        , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                        , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                        , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                        , PROVIDER_FIRST_NAME
                                        , PROVIDER_LAST_NAME
                                        , PROVIDER_ORGANIZATION_NAME
                                        , PROVIDER_DEGREE
                                        , BUSINESS_MAILING_ADDRESS
                                        , PROVIDER_TELEPHONE_NUMBER
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                        , CLASSIFICATION
                                        , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                FROM score
                                GROUP BY ALL
                                ),
                                         CTE_PERCENTILE as
                                        (select *
                                                , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                , case 
                                                    when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                    else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                            from CTE_AGGREGATED
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                        
                                        (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                RANK as ENGAGEMENT_RANK,
                                                                ENGAGEMENT_PERCENTILE
                                                            , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                            when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                            else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                            H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                          PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                            from CTE_PERCENTILE  
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            # ------------------ Sub-query 3_2 -------------------------------------
                            sub_query3_2 = f"""with score as(SELECT NPPES_NPI,
                                            LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                            LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                            LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                            sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                            sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                            sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                            PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[1]}%'
                                group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                CTE_AGGREGATED as 
                                (
                                SELECT nppes_npi
                                        , ALL_VALUES
                                        , ALL_DISTINCT_TERMS
                                        , ALL_DISTINCT_SPECIALITY
                                        , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                        , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                        , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                        , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                        , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                        , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                        , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                        , PROVIDER_FIRST_NAME
                                        , PROVIDER_LAST_NAME
                                        , PROVIDER_ORGANIZATION_NAME
                                        , PROVIDER_DEGREE
                                        , BUSINESS_MAILING_ADDRESS
                                        , PROVIDER_TELEPHONE_NUMBER
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                        , CLASSIFICATION
                                        , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                FROM score
                                GROUP BY ALL
                                ),
                                         CTE_PERCENTILE as
                                        (select *
                                                , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                , case 
                                                    when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                    else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                            from CTE_AGGREGATED
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                        
                                        (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                RANK as ENGAGEMENT_RANK,
                                                                ENGAGEMENT_PERCENTILE
                                                            , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                            when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                            else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                            H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                          PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                            from CTE_PERCENTILE  
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            # ------------------ Sub-query 3_3 -------------------------------------
                            sub_query3_3 = f"""with score as(SELECT NPPES_NPI,
                                            LISTAGG(DISTINCT upper(VALUE), ';\n') WITHIN GROUP (ORDER BY upper(VALUE)) AS ALL_VALUES,
                                            LISTAGG(DISTINCT VOCAB, ';\n') WITHIN GROUP (ORDER BY VOCAB) AS ALL_DISTINCT_TERMS,
                                            LISTAGG(DISTINCT FIX_SPEC, ';\n') WITHIN GROUP (ORDER BY FIX_SPEC) AS ALL_DISTINCT_SPECIALITY,
                                            sum(EMAIL_COUNT) as TOTAL_EMAIL_COUNT,sum(NLP_COUNT) as TOTAL_NLP_COUNT,
                                            sum(RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT,sum(JS_COUNT) as TOTAL_JS_COUNT,sum(OTHERS_COUNT) as TOTAL_OTHERS_COUNT,sum(NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT,
                                            sum(total_count) as NPI_TOTAL_ENGAGEMENT_COUNT,
                                            PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                            PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                   
                                FROM {MASTER_TABLE}  where lower(value) ILIKE '%{drug_disease_list[2]}%'
                                group by NPPES_NPI,  PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC
                                order by NPI_TOTAL_ENGAGEMENT_COUNT desc), 
                                CTE_AGGREGATED as 
                                (
                                SELECT nppes_npi
                                        , ALL_VALUES
                                        , ALL_DISTINCT_TERMS
                                        , ALL_DISTINCT_SPECIALITY
                                        , sum(TOTAL_EMAIL_COUNT) as TOTAL_EMAIL_COUNT
                                        , sum(TOTAL_NLP_COUNT) as TOTAL_NLP_COUNT
                                        , sum(TOTAL_RIDDLE_COUNT) as TOTAL_RIDDLE_COUNT
                                        , sum(TOTAL_JS_COUNT) as TOTAL_JS_COUNT
                                        , sum(TOTAL_OTHERS_COUNT) as TOTAL_OTHERS_COUNT
                                        , sum(TOTAL_NEI_APP_COUNT) as TOTAL_NEI_APP_COUNT
                                        , sum(NPI_TOTAL_ENGAGEMENT_COUNT) as NPI_TOTAL_ENGAGEMENT_COUNT
                                        , PROVIDER_FIRST_NAME
                                        , PROVIDER_LAST_NAME
                                        , PROVIDER_ORGANIZATION_NAME
                                        , PROVIDER_DEGREE
                                        , BUSINESS_MAILING_ADDRESS
                                        , PROVIDER_TELEPHONE_NUMBER
                                        , LISTAGG(DISTINCT (ENGAGED_EMAIL), ';\n') WITHIN GROUP (ORDER BY (ENGAGED_EMAIL)) AS ENGAGED_EMAIL
                                        , CLASSIFICATION
                                        , LISTAGG(DISTINCT (ALL_EMAILS_BLUECONIC), ';\n') WITHIN GROUP (ORDER BY (ALL_EMAILS_BLUECONIC)) as ALL_EMAILS_BLUECONIC 
                                FROM score
                                GROUP BY ALL
                                ),
                                         CTE_PERCENTILE as
                                        (select *
                                                , round(100*(PERCENT_RANK() OVER (ORDER BY NPI_TOTAL_ENGAGEMENT_COUNT)), 4) AS ENGAGEMENT_PERCENTILE
                                                , case 
                                                    when NPI_TOTAL_ENGAGEMENT_COUNT >= (select avg(NPI_TOTAL_ENGAGEMENT_COUNT) from score) then 'High'
                                                    else 'Low' end as H_L_ENGAGEMENT_BUCKET
                                                , dense_rank() over (order by NPI_TOTAL_ENGAGEMENT_COUNT desc) as RANK
                                            from CTE_AGGREGATED
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)
                                        
                                        (select NPPES_NPI, ALL_VALUES,ALL_DISTINCT_TERMS,ALL_DISTINCT_SPECIALITY,
                                                                TOTAL_EMAIL_COUNT,TOTAL_NLP_COUNT,TOTAL_RIDDLE_COUNT,TOTAL_JS_COUNT,TOTAL_OTHERS_COUNT,TOTAL_NEI_APP_COUNT,NPI_TOTAL_ENGAGEMENT_COUNT,
                                                                RANK as ENGAGEMENT_RANK,
                                                                ENGAGEMENT_PERCENTILE
                                                            , case when ENGAGEMENT_PERCENTILE between 66.6 and 100 then 'High'
                                                            when ENGAGEMENT_PERCENTILE between 33.3 and 66.6 then 'Medium'
                                                            else 'Low' end as H_M_L_ENGAGEMENT_BUCKET_PERCENTILE,
                                                            H_L_ENGAGEMENT_BUCKET as H_L_ENGAGEMENT_BUCKET, 
                                                          PROVIDER_FIRST_NAME,PROVIDER_LAST_NAME,PROVIDER_ORGANIZATION_NAME,
                                                    PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, ENGAGED_EMAIL,CLASSIFICATION,ALL_EMAILS_BLUECONIC                                         
                                            from CTE_PERCENTILE  
                                            order by NPI_TOTAL_ENGAGEMENT_COUNT desc)"""
                            
                            # -------------------- Combined Query Result -------------------- 
                       
                            df3 = fetch_data(query3)
                            df3_1 = fetch_data(sub_query3_1)
                            df3_2 = fetch_data(sub_query3_2)
                            df3_3 = fetch_data(sub_query3_3)

                            
                            if len(df3) == 0 and len(df3_1) == 0 and len(df3_2) == 0 and len(df3_3) == 0 :
                                st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[0]}], :blue-background[{drug_disease_list[1]}], :blue-background[{drug_disease_list[2]}]!] :red[ and hence no result tables and plots are displayed !!]***")
                            else:
                                #top_match3_sf = session.create_dataframe(top_match3)
                                with st.container(border=True):
                                    with st.spinner('Generating summary results ...'):
                                        drug_disease_combined_str = ",".join(drug_disease_list)
                                        display_summary_table({drug_disease_combined_str : df3, drug_disease_list[0] : df3_1, drug_disease_list[1] : df3_2, drug_disease_list[2] : df3_3}, is_option1=False)
                                
                                st.markdown("#")
                                with st.container(border=True):
                                    with st.spinner('Displaying degree distributions ...'):
                                        display_provider_degree_distribution()

                                st.markdown('#')
                                #----------------------- PLOTS - DRUGS MATCH -----------------------------------
                                with st.container(border=True):
                                    with st.spinner('Displaying match results ...'):
                                        st.markdown(f"***:pushpin: :blue[Plots showing the distributions for engagements of healthcare professionals interested in :grey-background[{drug_disease_list[0]}], :grey-background[{drug_disease_list[1]}] and :grey-background[{drug_disease_list[2]}]:]***")
                                        
                                        col1, col2, col3, col4, col5 = st.columns([1,4,1,4,1])
                                        with col2:
                                            with st.container():
                                                st.markdown("*1. Distribution - Percentile based engagements for all NPIs*")
                                                display_engagement_pie_chart(df3, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                        
                                        with col4:
                                            with st.container():
                                                st.markdown("*2. Distribution - Average engagements based for all NPIs*")
                                                display_engagement_pie_chart(df3, 'H_L_ENGAGEMENT_BUCKET')
                                                
                                        st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[0]}], :blue-background[{drug_disease_list[1]}] and :blue-background[{drug_disease_list[2]}]:***")
                                        data3 = df3[:1000]
                                        st.dataframe(data3.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                        if len(df3) > 30000:
                                            b64 = convert_into_csv(df3)
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}_{drug_disease_list[1]}_&_{drug_disease_list[2]}.csv.gz"
                                        else:
                                            csv_df3 = convert_df(df3)
                                            b64 = base64.b64encode(csv_df3).decode('utf-8')
                                            file = f"NPIdataFor_Value_{drug_disease_list[0]}_{drug_disease_list[1]}_&_{drug_disease_list[2]}.csv"
                                            
                                        href = f'''
                                                <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                    display: inline-block;
                                                    font-size: 16px;
                                                    color: white;
                                                    background-color: #4CAF50;
                                                    padding: 10px 20px;
                                                    text-align: center;
                                                    text-decoration: none;
                                                    border-radius: 5px;
                                                    margin-top: 10px;
                                                ">Download data as CSV</a>
                                                '''
                                        st.markdown(href, unsafe_allow_html=True)
                                
                            st.markdown("#")
                            colxx, colyy, colzz = st.columns([1,1,1])
                            with colyy:
                                st.markdown("**:red[Results for Individual Values]**")
                                st.text("")
                            
                            # -------------------- Sub-Query 1 Result -------------------- 
                            
                            with st.container(border=True):
                                with st.spinner('Displaying individual results ...'):
                                    if len(df3_1) == 0:
                                        st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[0]}]!]***")
                                    else:
                                        with st.container():
                                            cola, colb = st.columns([1,1])
                                            with cola:
                                                st.markdown(f"*Distribution - Percentile based engagements for :blue-background[{drug_disease_list[0]}]*")
                                                display_engagement_pie_chart(df3_1, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                            with colb:    
                                                st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[0]}]:***")
                                                data3_1 = df3_1[:1000]
                                                st.dataframe(data3_1.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                                if len(df3_1) > 30000:
                                                    b64 = convert_into_csv(df3_1)
                                                    file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv.gz"
                                                else:
                                                    csv_df3_1 = convert_df(df3_1)
                                                    b64 = base64.b64encode(csv_df3_1).decode('utf-8')
                                                    file = f"NPIdataFor_Value_{drug_disease_list[0]}.csv"
                                                    
                                                href = f'''
                                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                            display: inline-block;
                                                            font-size: 16px;
                                                            color: white;
                                                            background-color: #4CAF50;
                                                            padding: 10px 20px;
                                                            text-align: center;
                                                            text-decoration: none;
                                                            border-radius: 5px;
                                                            margin-top: 10px;
                                                        ">Download data as CSV</a>
                                                        '''
                                                st.markdown(href, unsafe_allow_html=True)
            
                                    # -------------------- Sub-Query 2 Result -------------------- 
                                    
                                    
                                    if len(df3_2) == 0:
                                        st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[1]}]!]***")
                                    else:
                                        with st.container():
                                            colc, cold = st.columns([1,1])
                                            with colc:
                                                #top_match3_2_sf = session.create_dataframe(df3_2)
                
                                                st.markdown(f"*Distribution - Percentile based engagements for :blue-background[{drug_disease_list[1]}]*")
                                                display_engagement_pie_chart(df3_2, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                            with cold:    
                                                st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[1]}]:***")
                                                data3_2 = df3_2[:1000]
                                                st.dataframe(data3_2.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                                if len(df3_2) > 30000:
                                                    b64 = convert_into_csv(df3_2)
                                                    file = f"NPIdataFor_Value_{drug_disease_list[1]}.csv.gz"
                                                else:
                                                    csv_df3_2 = convert_df(df3_2)
                                                    b64 = base64.b64encode(csv_df3_2).decode('utf-8')
                                                    file = f"NPIdataFor_Value_{drug_disease_list[1]}.csv"
                                                    
                                                href = f'''
                                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                            display: inline-block;
                                                            font-size: 16px;
                                                            color: white;
                                                            background-color: #4CAF50;
                                                            padding: 10px 20px;
                                                            text-align: center;
                                                            text-decoration: none;
                                                            border-radius: 5px;
                                                            margin-top: 10px;
                                                        ">Download data as CSV</a>
                                                        '''
                                                st.markdown(href, unsafe_allow_html=True)
                                    
                                    # -------------------- Sub-Query 3 Result -------------------- 
                                  
                                    df3_3 = fetch_data(sub_query3_3)
                                    
                                    if len(df3_3) == 0:
                                        st.write(f"***:red[No Matching NPI for :blue-background[{drug_disease_list[2]}]!]***")
                                    else:
                                        with st.container():
                                            cole, colf = st.columns([1,1])
                                            with cole:
                                                #top_match3_3_sf = session.create_dataframe(top_match3_3)
                                                st.markdown(f"*Distribution Plot - Percentile based engagements for :blue-background[{drug_disease_list[2]}]*")
                                                display_engagement_pie_chart(df3_3, 'H_M_L_ENGAGEMENT_BUCKET_PERCENTILE')
                                            with colf:          
                                                st.write(f":memo: ***List of Top 1000 NPIs having engagement with :blue-background[{drug_disease_list[2]}]:***")
                                                data3_3 = df3_3[:1000]
                                                st.dataframe(data3_3.style.applymap(highlight, subset=['H_M_L_ENGAGEMENT_BUCKET_PERCENTILE', 'H_L_ENGAGEMENT_BUCKET']), hide_index=False)
                                                if len(df3_3) > 30000:
                                                    b64 = convert_into_csv(df3_3)
                                                    file = f"NPIdataFor_Value_{drug_disease_list[2]}.csv.gz"
                                                else:
                                                    csv_df3_3 = convert_df(df3_3)
                                                    b64 = base64.b64encode(csv_df3_3).decode('utf-8')
                                                    file = f"NPIdataFor_Value_{drug_disease_list[2]}.csv"
                                                    
                                                href = f'''
                                                        <a href="data:file/csv;base64,{b64}" download="{file}" style="
                                                            display: inline-block;
                                                            font-size: 16px;
                                                            color: white;
                                                            background-color: #4CAF50;
                                                            padding: 10px 20px;
                                                            text-align: center;
                                                            text-decoration: none;
                                                            border-radius: 5px;
                                                            margin-top: 10px;
                                                        ">Download data as CSV</a>
                                                        '''
                                                st.markdown(href, unsafe_allow_html=True)
                                                
                    st.markdown("*:red[Note: Tables with records greater than 2,00,000 will be downloaded as zip files]*")
            
                else:
                    st.write("***:red[User Query is either out of scope or does not have sufficient information to retrieve records]***")
            except Exception as e:
                st.markdown("***:red[Please check your query again!]***")
                # For local run only
                # st.error(f"**🚨Error occured in processing your query due to Exception :{e}, Please refresh the page**")
                # log.error(traceback.format_exc())
                # st.error(f"**🚨Error occured in processing your query due to Exception :{e}, Please check you query again!**")
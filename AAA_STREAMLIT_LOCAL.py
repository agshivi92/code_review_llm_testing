import json  # To handle JSON data
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import snowflake  # For interacting with Snowflake-specific APIs
import pandas as pd
import streamlit as st  # Streamlit library for building the web app
#from snowflake.snowpark.context import get_active_session  # To interact with Snowflake sessions
from snowflake.snowpark.exceptions import SnowparkSQLException
# from snowflake.snowpark.async_job import AsyncJob
# import snowflake.connector
# from snowflake.connector.cursor import ASYNC_RETRY_PATTERN
#import snowflake.cortex as cortex
from snowflake import cortex
import plotly.graph_objects
# from plotly.graph_objects as go
import os
from PIL import Image
import tempfile
from snowflake.snowpark import Session
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit
import getpass
import requests
import re


API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 100000  # in milliseconds

# Initialize a Snowpark session for executing queries
#session = get_active_session()

st.set_page_config(layout='wide')

connection_parameters = {
    "account": "LDDJJUA-TKA20186",             # e.g., "xy12345.us-east-1"
    "user": "SAGARWAL",   # Your Okta SSO username (not password)
    "authenticator": "externalbrowser",      # This triggers Okta SSO via browser
    "role": "RBP_REPORTS_UAT_RW",
    "warehouse": "RBP_KIPI_UAT_WH",
    "database": "RBP_UAT",
    "schema": "RBP_REPORTS"
}

# Add this near the top of your file with other constants
PREDEFINED_QUESTIONS = [
    "What is the status of case id 41526142?",
    "Show me the comments for case 41526142",
    "Show me case details for case_id 2525550",
    "show me case counts by month for General Motors along with billing status in 2025",
    "show me general motors case count that are not in Billing_Ok for 2025 by status",
    "show me count of all cases for general motors by week by club for 2025"
    #"show me # of cases and status for program id 147 for time between May1st, 2025 till today.",
    # "show me ALL the cases in progress with status rule_ok",
    #"Count of GM vehicles with current status FINANCIAL_OK",
    # "What is the average time it takes to be available on location"
]

if "debug_variable" not in st.session_state:
    st.session_state["debug_variable"] = None
    
def main():
    # Initialize session state
    # session_state = {}
    if "messages" not in st.session_state:
        reset_session_state()
    else:
        ## Clean up the message sequence if needed
        cleanup_message_sequence()

    

     # Create header with logo
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title("RAP Case Query Agent")
          
    with header_col2:
        #for file in ['chat-bot']:
        session.file.get(f'@"RAP_UAT"."RBP_REPORTS"."RBP_KIPI_STAGE"/AAA_logo.png','/tmp')
        #Display the logo in the top right
        #st.image("HT_img", width=150)  
        st.plotly_chart(image_png('/tmp/AAA_logo.png', img_width = 400, img_height = 200))

    st.write("***:red[The data was last updated as of date 4th June,2025]***")
    # Create two main tabs at the top level
    main_tab1, main_tab2 = st.tabs(["Chat Interface", "Recent Queries"])
    
    with main_tab1:
        # Current chatbot interface
        show_header_and_sidebar()
        
        # Replace the API call with static questions
        if len(st.session_state.messages) == 0:
            #  welcome message and show predefined questions
            welcome_message = {
                "role": "analyst",
                "content": [
                    {"type": "text", "text": "Welcome! Here are some questions you can ask:"},
                    {"type": "suggestions", "suggestions": PREDEFINED_QUESTIONS}
                ]
            }
            st.session_state.messages.append(welcome_message)
        
        display_conversation()
        handle_user_inputs()
        handle_error_notifications()
        # st.write(st.session_state.actual_analyst_time)
        # st.write(st.session_state.analyst_time)
    
    with main_tab2:
        # Query history interface
        display_query_history()
        # st.write(st.experimental_user["user_name"])
        # print("username",st.experimental_user["user_name"])
        #pass

def image_png(file, img_width, img_height):

    loc = os.path.abspath(file)

    # Create figure
    fig = plotly.graph_objects.Figure()

    # Constants
    img_width = img_width
    img_height = img_height
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        plotly.graph_objects.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        # range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        # range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )
    pyLogo = Image.open(loc)
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=pyLogo)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig
    
def display_query_history():
    """
    Display the user's query history.
    """
  #  st.header("Your Recent Queries")
    st.info(
        "This tab shows your queries in past 30 days"
    )
    #st.divider()
    
    # Get the current user
    #current_user = st.experimental_user["user_name"]
    if hasattr(st, "experimental_user") and st.experimental_user and "user_name" in st.experimental_user:
        current_user = st.experimental_user["user_name"]
        
    else:
        current_user = session.sql(f'SELECT CURRENT_USER()').collect()[0][0]
    # st.markdown()
    # st.write("User_Name:", current_user)
    # Query to fetch the user's history
    query = f"""
    SELECT  user_id, raw_query, query_time
    FROM  RAP_UAT.RBP_REPORTS.QUERY_HISTORY
    where user_id = '{current_user}' and QUERY_TIME >= DATEADD(DAY, -30, CURRENT_DATE())
    ORDER BY query_time DESC  
    """
    
    try:
        # Execute the query
        df = session.sql(query).to_pandas()
       
        
        if df.empty:
            st.info("You haven't asked any questions in the past 30 days.")
        else:
            
            unique_df = df.drop_duplicates(subset=['RAW_QUERY'])
    
            # Optionally display the DataFrame (use unique_df instead of df1)
            #st.dataframe(unique_df,use_container_width=True, hide_index=True)
    
            # Iterate over unique rows and display each RAW_QUERY
            for index, row in unique_df.iterrows():
                st.code(f""" {row['RAW_QUERY']}""") 
                
    except SnowparkSQLException as e:
        st.error(f"Error retrieving query history: {e}")
    

def reset_session_state():
    """Reset session state to clear chat history."""
    # Clear existing messages
    st.session_state.messages = []
    # Reset other state variables
    st.session_state.active_suggestion = None
    st.session_state.content = {}
    st.session_state.insight_generation = False
    st.session_state.follow_up_suggestions = False
    st.session_state.analyst_time = 0
    st.session_state.actual_analyst_time =0
    st.session_state.cancel_execution = False  ## Added for stop button 
# def reset_session_state(state):
#     state['messages'] = []
#     state['active_suggestion'] = None
#     state['content'] = {}
#     state['insight_generation'] = False
#     state['follow_up_suggestions'] = False
#     state['analyst_time'] = 0
#     state['actual_analyst_time'] = 0
#     state['cancel_execution'] = False  

def enable_insight_generation():
    st.session_state.insight_generation = not st.session_state.insight_generation
    
    
def enable_follow_up_suggestions():
    st.session_state.follow_up_suggestions = not st.session_state.follow_up_suggestions

def show_header_and_sidebar():
    """Display the header and sidebar of the app."""
    # Set the title and introductory text of the app
    #st.title("RBP Insights Assistant")
    st.markdown(
        "Welcome to RBP Insights assistant chatbot! Type your questions below to interact with your data. "
    )
    #st.divider()

    # Sidebar with a reset button
    with st.sidebar:
        # st.selectbox(
        #     "Selected semantic model:",
        #     AVAILABLE_SEMANTIC_MODELS_PATHS,
        #     format_func=lambda s: s.split("/")[-1],
        #     key="selected_semantic_model_path",
        #     on_change=reset_session_state,
        # )
        
        
        #st.divider()

        # toggle for insight generation
        _, toggle_btn, _ = st.columns([2, 6, 2])
        toggle_btn = st.toggle("Enable Insight Generation", on_change=enable_insight_generation)
        if toggle_btn:
            #slider to control the number of rows passed to the LLM
            st.slider(
                "Max rows to process for insights",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="max_rows_for_llm",
                help="Limit the number of rows passed to the LLM for generating insights to avoid token size limits"
            )
            
            st.session_state.insight_generation = True
        
        # toggle for follow up suggestions
        _, toggle_btn_2, _ = st.columns([2, 6, 2])
        toggle_btn_2 = st.toggle("Enable Follow-up Suggestions", on_change=enable_follow_up_suggestions)
        if toggle_btn_2:
            st.session_state.follow_up_suggestions = True            
        
        # Center this button
        st.divider()
        _, btn_container, _ = st.columns([2, 6, 2])
        if btn_container.button("Clear Chat History", use_container_width=True):
            reset_session_state()


def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    # Handle chat input
    user_input = st.chat_input("What is your question?")
    if user_input:
        process_user_input(user_input)
    # Handle suggested question click
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    if st.session_state.get("fire_API_error_notify"):
        st.toast(f"An API error has occured! : {st.session_state['content']}", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    
    ### Normalize the input prompt for consistent comparison
    print("process_user_input")
    normalized_prompt = prompt.strip().lower()
    print("normalized_prompt" ,normalized_prompt)
    # Check if this question was asked before in the session
    previous_response = None
    previous_user_msg_index = None
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            previous_prompt = msg["content"][0]["text"].strip().lower()
            if previous_prompt == normalized_prompt:
                previous_user_msg_index = idx
                # Look for the next analyst response after this user message
                if idx + 1 < len(st.session_state.messages) and st.session_state.messages[idx + 1]["role"] == "analyst":
                    previous_response = st.session_state.messages[idx + 1]
                    st.write(idx,previous_response)
                break

    ### If the question was asked before and has a response with SQL, reuse it
    if previous_response is not None:
        # Check if the previous response contains SQL content
        has_sql = False
        for content_item in previous_response["content"]:
            if content_item.get("type") == "sql" and "statement" in content_item:
                has_sql = True
                break
        
        # Only reuse the response if it contains SQL (meaning Cortex successfully generated a query)
        if has_sql:
            # Add the user message to maintain conversation flow
            new_user_message = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
            st.session_state.messages.append(new_user_message)
            with st.chat_message("user"):
                user_msg_index = len(st.session_state.messages) - 1
                print("enetering the display message thru process")
                display_message(new_user_message["content"], user_msg_index)

            ### Reuse the previous analyst response
            reused_message = previous_response.copy()
            reused_message["content"] = reused_message["content"] + [{"type": "text", "text": "\n*(Reused from previous query)*"}]
            st.session_state.messages.append(reused_message)
            with st.chat_message("analyst"):
                display_message(reused_message["content"], len(st.session_state.messages) - 1)
            st.rerun()
            return

    # If not asked before, proceed with normal processing
    ## Check if the last message was an analyst message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "analyst":
        # Can safely add a user message
        new_user_message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
        st.session_state.messages.append(new_user_message)
        with st.chat_message("user"):
            user_msg_index = len(st.session_state.messages) - 1
            display_message(new_user_message["content"], user_msg_index)
    else:
        # Last message was already a user message or empty list
        # We need to replace the last message or add a new one
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            # Replace the last user message
            st.session_state.messages[-1] = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        else:
            # Add a new user message
            st.session_state.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            })
        
        with st.chat_message("user"):
            user_msg_index = len(st.session_state.messages) - 1
            display_message(st.session_state.messages[-1]["content"], user_msg_index)

    ## Reset cancellation flag
    st.session_state.cancel_execution = False

    
    # Show progress indicator inside analyst chat message while waiting for response
    with st.chat_message("analyst"):
        stop_col1, stop_col2, stop_col3 = st.columns([1, 1, 1])
        with stop_col2:
            if st.button("🛑 Stop Execution", key="stop_button", use_container_width=True):
                st.session_state.cancel_execution = True
                st.warning("Cancelling operation...")
                # Add the cancel message but don't rerun yet
                cancel_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": "Operation cancelled by user."}],
                }
                st.session_state.messages.append(cancel_message)
                st.rerun()
        
        with st.spinner("Waiting for Analyst's response..."):
            if not st.session_state.cancel_execution:
                start_time = time.time()
                # Pass the entire conversation history (including previous messages) to the API
                #response, error_msg = get_analyst_response(st.session_state.messages) 
                
                ## Validate the message sequence
                validated_messages = validate_message_sequence(st.session_state.messages)
                filtered_messages = []
                for d in validated_messages:
                    filtered_d = {k: v for k, v in d.items() if k not in ["judge_result", "llm_insights", "result_df", "insights_enabled"]}
                    filtered_messages.append(filtered_d)
                print("filtered_messages",filtered_messages)
                result= session.call("RAP_UAT.RBP_REPORTS.get_analyst_response", filtered_messages)
                if isinstance(result, str):
                    result = json.loads(result)

                # Now you can safely access the keys
                response = result.get("parsed_content")
                error_msg = result.get("error_msg")
                # Check if execution was cancelled during API call
                if st.session_state.cancel_execution:
                    # Don't add another message here - just rerun
                    st.rerun()
                    
                end_time = time.time()
                st.session_state.analyst_time = end_time-start_time

                if error_msg is None:
                    analyst_message = {
                        "role": "analyst",
                        "content": response["message"]["content"],
                        # "request_id": response["request_id"],
                    }
                else:
                    analyst_message = {
                        "role": "analyst",
                        "content": [{"type": "text", "text": error_msg}],
                        # "request_id": response["request_id"],
                    }
                    st.session_state["fire_API_error_notify"] = True
                
                # Append the analyst response to the conversation history
                st.session_state.messages.append(analyst_message)

                if st.session_state.follow_up_suggestions:
                    # Show additional followup questions if the last message contains SQL, and the proper feature flag is set
                    if (last_chat_message_contains_sql()):
                        if not st.session_state.cancel_execution:
                            get_and_display_smart_followup_suggestions()
            # else:
            #     return {"request_id": "cancelled"}, "Operation cancelled by user."
    st.rerun()

##
def cleanup_message_sequence():
    """
    Ensures the message sequence ends with a user message,
    removing any trailing analyst messages if needed.
    """
    if st.session_state.messages:
        # If the last message is from the analyst, we're good
        if st.session_state.messages[-1]["role"] == "analyst":
            return
        
        # If the last message is from the user, we need to make sure
        # there's only one user message at the end
        if st.session_state.messages[-1]["role"] == "user":
            # Start from the second-to-last message and move backward
            for i in range(len(st.session_state.messages) - 2, -1, -1):
                if st.session_state.messages[i]["role"] == "analyst":
                    # Found an analyst message, we can stop here
                    break
                if st.session_state.messages[i]["role"] == "user":
                    # Found another user message, remove it
                    st.session_state.messages.pop(i)



## to validate the message sequence before sending it to the API:
def validate_message_sequence(messages):
    """
    Ensures that the message sequence alternates between 'user' and 'analyst' roles.
    Returns a cleaned version of the messages.
    """
    if not messages:
        return []
    
    validated_messages = [messages[0]]  # Start with the first message
    
    for i in range(1, len(messages)):
        current_message = messages[i]
        prev_message = validated_messages[-1]
        
        # If the current message has the same role as the previous one, skip it
        if current_message["role"] == prev_message["role"]:
            continue
        
        validated_messages.append(current_message)
    
    # Ensure the last message is from the user before sending to API
    if validated_messages and validated_messages[-1]["role"] != "user":
        validated_messages = validated_messages[:-1]
    
    return validated_messages
            
# def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
#     """
#     Send chat history to the Cortex Analyst API and return the response.

#     Args:
#         messages (List[Dict]): The conversation history.

#     Returns:
#         Optional[Dict]: The response from the Cortex Analyst API.
#     """
    
#     ## Check if execution has been cancelled
#     if st.session_state.cancel_execution:
#         return {"request_id": "cancelled"}, "Operation cancelled by user."
    
#     ## Validate the message sequence
#     validated_messages = validate_message_sequence(messages)

    
#     # Prepare the request body with the user's prompt and full conversation history
#     filtered_messages = [{k: v for k,v in d.items() if k not in ["judge_result","llm_insights","result_df","insights_enabled"]} for d in messages]
#     request_body = {
#         "messages": filtered_messages,  # Pass the entire conversation history here
        
#        # "semantic_model_file": f"@{st.session_state.selected_semantic_model_path}",
        
#         "semantic_models": {"semantic_model_file": "@RAP_UAT.RBP_REPORTS.RBP_KIPI_STAGE/rbp_semantic_gold.yaml"},
#     }

#     ## diversion_prompt = f"""Given a user input, your job is to identify whether the input requires a forecasting/prediction or not. 
#     ## If we require forecasting/prediction return output as forecasting:1, Else just return forecasting:0. 
#     ## user input : {filtered_messages[-1]['content'][0]['text']} """

#     ## completion_response = cortex.complete(
#     ##        model="llama3.1-70b",
#     ##        prompt=diversion_prompt,
#     ##        session=session,
#     ##        options=cortex.CompleteOptions(temperature=0.2),
#     ##    )

#     # hardcoding forecasting variable until forecasting features are implemented
#     completion_response = 'forecasting:0'

#     if "forecasting:1" in completion_response:
#         resp = {
#             "message": {
#               "role": "analyst",
#               "content": [
#                 {
#                   "type": "text",
#                   "text": "This is our interpretation of your question:\n\n Forecast the sales for next 30 days"
#                 },
#                 {
#                   "type": "sql",
#                   "statement": "SELECT * FROM Forecast_Results where ts >= '2025-05-01' and ts <= '2025-05-30';",
#                   "confidence": {
#                     "verified_query_used": None
#                   }
#                 }
#               ]
#             },
#             "request_id": "e54379db-9cef-4b96-a081-7925a810a6ed"
#           }

#         return resp,None
#     else:
#         # Send a POST request to the Cortex Analyst API endpoint
#         # Adjusted to use positional arguments as per the API's requirement
#         analyst_time_start_time =time.time()
#         # resp = _snowflake.send_snow_api_request(
#         #     "POST",  # method
#         #     API_ENDPOINT,  # path
#         #     {},  # headers
#         #     {},  # params
#         #     request_body,  # body
#         #     None,  # request_guid
#         #     API_TIMEOUT,  # timeout in milliseconds
#         # )
#         host_name = session.conf.get("host")
#         token = session.conf.get("rest").token
#         resp = requests.post(
#             url=f"https://{host_name}{API_ENDPOINT}",
#             json=request_body,
#             headers={
#                 "Authorization": f'Snowflake Token="{token}"',
#                 "Content-Type": "application/json",
#             },
#         )
#         print(resp)
        
#         analyst_time_end_time =time.time()
#         st.session_state.actual_analyst_time =analyst_time_end_time -analyst_time_start_time

#         st.session_state["debug_variable"] = completion_response
            
#         # Content is a string with serialized JSON object
#         #parsed_content = json.loads(resp["content"])
    
#         # Check if the response is successful
#         if resp.status_code < 400:
#             # Return the content of the response as a JSON object
#             return resp, None
#         else:
#             st.session_state.content['resp'] = resp
#             #st.session_state.content['parsed_content'] = resp
            
#             # Craft readable error message
#             error_msg = f"""
#     🚨 An Analyst API error has occurred 🚨
    
#     * response code: `{resp.status_code}`
#     * error code: `{resp.raise_for_status()}`
    
#     Message:
    
#             """
#             return resp, error_msg

### Set of functions to display suggestions for follow up questions

def get_last_chat_message_idx() -> str:
    """Get message index for the last message in chat."""
    #st.write(f"get_last_chat_message_idx : {len(st.session_state.messages) - 1}")
    return len(st.session_state.messages) - 1
    
def last_chat_message_contains_sql() -> str:
    """Check if the last message in chat contains SQL content."""
    last_msg = st.session_state.messages[get_last_chat_message_idx()]
    msg_content_types = {c["type"] for c in last_msg["content"]}
    return "sql" in msg_content_types

def message_idx_to_question(idx: int) -> str:
    """
    Retrieve the question text from a message in the session state based on its index.

    This function checks the role of the message and returns the appropriate text:
    * If the message is from the user, it returns the prompt content
    * If the message is from the analyst and contains an interpretation of the question, it returns
    the interpreted question
    * Otherwise, it returns the previous user prompt

    Args:
        idx (int): The index of the message in the session state.

    Returns:
        str: The question text extracted from the message.
    """
    msg = st.session_state.messages[idx]

    # if it's user message, just return prompt content
    if msg["role"] == "user":
        return str(msg["content"][0]["text"])

    # If it's analyst response, if it's possibleget question interpretation from Analyst
    if msg["content"][0]["text"].startswith(
        "This is our interpretation of your question:"
    ):
        return str(
            msg["content"][0]["text"]
            .strip("This is our interpretation of your question:\n")
            .strip("\n")
            .strip("_")
        )

    # Else just return previous user prompt
    return str(st.session_state.messages[idx - 1]["content"][0]["text"])

def get_semantic_model_desc_from_messages() -> str:
    """Retrieve semantic model description from chat history.

    It assumes that in history there was a descritpion provided by Cortex Analyst,
    and it starts with "This semantic data model contains information about".
    """
    for msg in st.session_state.messages:
        for content in msg["content"]:
            if content["type"] == "text" and content["text"].startswith(
                "This semantic data model contains information about"
            ):
                return content["text"]
    return ""

def get_question_suggestions(
    previous_question: str, semantic_model_desc: str, requested_suggestions: int = 3
) -> Tuple[List[str], Optional[str]]:
    """
    Generate follow-up questions based on the previous question asked by the user and the semantic model description.

    This function utilizes the Snowflake Cortex Compleate to generate follow-up questions that encourage the user
    to explore the data in more depth.

    Args:
        previous_question (str): The last question asked by the user.
        semantic_model_desc (str): The description of the underlying semantic model provided by the analyst.
        requested_suggestions (int, optional): The number of follow-up questions to generate. Defaults to 3.

    Returns:
        Tuple[List[str], Optional[str]]: A tuple containing a list of generated follow-up questions and an optional error message.
                                         If an error occurs, the list of suggestions will be empty and the error message will be provided.
    """
    global session
    prompt = f"""
You will suggest follow-up questions to the Business User who is interacting with data via Cortex Analyst - "Talk to your data" solution from Snowflake. Here is the description provided by Analyst on the underlying semantic model:
{semantic_model_desc}

The user's goal is to gain insights into underlying data. They have previously asked:
{previous_question}

Now generate {requested_suggestions} follow-up questions that encourage the user to explore the data in more depth. The tone should be formal and concise. Please provide questions that are precise and non-ambiguous.

Some examples of good follow-up questions might include: "What are the top 3 factors contributing to [specific trend]?" or "How does [specific variable] affect [outcome]?"

Output your answer as a JSON list of strings, like this:
["suggestion 1", "suggestion 2", "suggestion 3"]

Refrain from adding any other text before or after the generated list.
Here is the answer:
[
"""
    try:
        completion_response_raw = cortex.complete(
            model="llama3.1-70b",
            prompt=prompt,
            session=session,
            #options=cortex.CompleteOptions(temperature=0.2),
        )
        completion_response_parsed = json.loads(completion_response_raw)
        #st.write(f"completion_response_parsed : {completion_response_parsed}")
        # https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex#returns
        # parsed_sugggedtions = json.loads(
        #     completion_response_parsed["choices"][0]["messages"]
        # )
    except SnowparkSQLException as e:
        err_msg = f"Error while generating suggestions thtough cortex.Complete: {e}"
        return [], err_msg
    except json.JSONDecodeError as e:
        err_msg = f"Error while parsing reponse from cortex.Compleate: {e}"
        return [], err_msg

    return completion_response_parsed, None
    
def get_and_display_smart_followup_suggestions():
    """Get smart followup questions for the last message and update the session state."""
    with st.spinner("Generating followup questions..."):

        ## Check if execution has been cancelled
        if st.session_state.cancel_execution:
            return

        
        
        question = message_idx_to_question(get_last_chat_message_idx())
        sm_description = get_semantic_model_desc_from_messages()
        suggestions, error_msg = get_question_suggestions(question, sm_description)

        ## Check again if execution was cancelled during API call
        if st.session_state.cancel_execution:
            return

        
        # If suggestions were successfully generated update the session state
        if error_msg is None:
            st.session_state.messages[-1]["content"].append(
                {"type": "text", "text": "__Suggested followups:__"}
            )
            st.session_state.messages[-1]["content"].append(
                {"type": "suggestions", "suggestions": suggestions}
            )

            
def display_conversation():
    """
    Display the conversation history between the user and the assistant.
    """
    print("messages..................", st.session_state.messages)
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            print("idx..................", idx)
            display_message(content, idx)


def display_message(content: List[Dict[str, str]], message_index: int):
    """
    Display a single message content.

    Args:
        content (List[Dict[str, str]]): The message content.
        message_index (int): The index of the message.
    """
    #st.write(content)
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            # Display suggestions as buttons
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, key=f"suggestion_{message_index}_{suggestion_index}"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            # Display the SQL query and results
            display_sql_query(item, message_index)
            # st.write(session.get_current_user())
            #Add thumbs-up and thumbs-down buttons for feedback
            thumbs_up, thumbs_down = st.columns([1, 1])
            if hasattr(st, "experimental_user") and st.experimental_user and "user_name" in st.experimental_user:
                current_user = st.experimental_user["user_name"]
            else:
                current_user = session.sql(f'SELECT CURRENT_USER()').collect()[0][0]
            ################
            save_query(current_user,
                        st.session_state.messages[message_index]["content"][0]["text"],
                        item["statement"],st.session_state.messages[message_index-1]["content"][0]["text"]
                     )    
            


def display_sql_query(item: dict, message_index: int):
    """
    Executes the SQL query and displays the results in form of data frame and charts.

    Args:
        sql (str): The SQL query.
        message_index (int): The index of the message.
    """
    sql = item["statement"]
    # Display the SQL query
    with st.expander("SQL Query", expanded=False):
        st.code(sql, language="sql")

    # Display the results of the SQL query
    with st.expander("Results", expanded=True):
        content = st.session_state.messages[message_index]['content']
        print("content", content)
        # print("initial content", content)
        with st.spinner("Running SQL..."):
            has_result_df = any("result_df" in d for d in content)
            print("has_result_df", has_result_df)
            if has_result_df is False:   
                df1, err_msg = get_query_exec_result(sql, message_index)
                #print("sql",sql)
                # result = session.call("RAP_UAT.RBP_REPORTS.execute_query", sql)
                # print(type(result))
                # #print(result)
                # df1 = None
                # if isinstance(result, dict) and "error" in result:
                #     print("Error:", result["error"])
                # elif isinstance(result, str):
                #     # Result is a JSON string

                #     if result.strip() == "" or result.strip() == "[]":
                #         print("No data returned from stored procedure.")
                #         df1 = pd.DataFrame()
                #     else:
                #         try:
                #             data = json.loads(result)
                #             # Then, convert to DataFrame
                #             df1 = pd.DataFrame(data)
                #             for item in content:
                #                 if item.get("type") == "sql":
                #                     item["result_df"] = df1
                #                     item["insights_enabled"] = st.session_state.insight_generation
                #             st.session_state.messages[message_index]['content'] = content
                #         except Exception as e:
                #             print("Could not parse JSON string:", e)
                #             print("Raw result:", result)
                    
                # elif isinstance(result, list):
                #     # Result is a list of dicts (VARIANT array)
                #     df1 = pd.DataFrame(result)
                #     for item in content:
                #         if item.get("type") == "sql":
                #             item["result_df"] = df1
                #             item["insights_enabled"] = st.session_state.insight_generation
                #     st.session_state.messages[message_index]['content'] = content
                # else:
                #     print("Unexpected return type from stored procedure:", type(result))
                #     for item in content:
                #         if item.get("type") == "sql":
                #             item["result_df"] = None
                #             item["insights_enabled"] = st.session_state.insight_generation
                #     st.session_state.messages[message_index]['content'] = content
            else:
                # Will get the value of 'result_df' if it exists, else None
                df_json = next((d["result_df"] for d in content if "result_df" in d), None)
                df1 = pd.read_json(df_json)
                
                # Define known date fields
                date_fields = [
                    "RAP_CALL_CREATED_DATE_TIME", "RAP_CALL_UPDATED_DATE_TIME", "SENT_ON", 
                    "CREATED_AT_DATE_TIME", "UPDATED_AT_DATE_TIME","PTA","ETA",
                    "START_PERIOD", "END_PERIOD", "START_DATE_TIME", "END_DATE_TIME"
                ]
    
                # Convert known date columns back to datetime
                for col in df1.columns:
                    if col in date_fields:
                        try:
                            df1[col] = pd.to_datetime(df1[col])
                        except Exception as e:
                            # If conversion fails, log it but continue
                            print(f"Failed to convert column {col} to datetime: {e}")
                
            if df1 is None:
                st.error(f"Could not execute generated SQL query. Error: {err_msg}")
                return

            if df1.empty:
                st.write("Query returned no data")
                return
            # total_bytes = df1.memory_usage(index=True).sum()
            # # Convert bytes to megabytes (1 MB = 1024 * 1024 bytes)
            # bytes_in_mb = 1024 * 1024
            # size_in_mb = total_bytes / bytes_in_mb
            
            # Print the result
            # print(f"DataFrame size in Bytes: {total_bytes}")
            # print(f"DataFrame size in MB: {size_in_mb:.2f}")
            # total_records = len(df1)
            # if size_in_mb <= 32:
            #     df = df1         
            # else:
            #     st.write(f"***:red[Your request for data resulted in {total_records} records. Displaying records is limited to 50k records. You can download the complete data using the e-mail function]***")
            #     df= df1[:50000]
            # Show query results in three tabs - added insights tab
            # data_tab, chart_tab, insights_tab, LLM_Judge_Rating_tab = st.tabs(["Data 📄", "Chart 📈", "Insights 💡", "LLM_Judge_Rating :star:"])
            # data_tab, chart_tab, insights_tab, LLM_Judge_Rating_tab = st.tabs([  
            #     "Data 📄",  
            #     "Chart 📈",  
            #     "Insights 💡",  
            #     "Insights Evaluation ⭐"  
            # ]) 
            df=df1 
            data_tab, insights_tab = st.tabs([  
                "Data 📄",   
                "Insights 💡" 
            ])  
            # Content inside each tab  
            with data_tab:  
                st.info("This tab contains data generated by sql query.")  
            
            # with chart_tab:  
            #     st.info("This tab provides chart-related insights in the form of visual representations of the SQL query results to gain a deeper understanding of trends and patterns within the data.")  
            
            with insights_tab:  
                st.info("This tab highlights the key insights derived from the SQL query outputs. It dives into the key trends and patterns, notable outliers or anomalies, business implications and actionable insights that can aid in decision-making.") 
            
            # with LLM_Judge_Rating_tab:  
            #     st.info("This tab evaluates the quality of the insights generated by the SQL queries,using ratings and feedback from the LLM as a benchmark. It considers the relevance,depth of insight, clarity and readability and actionability")
            with data_tab:
                st.dataframe(df, use_container_width=True, hide_index=True)

            # with chart_tab:
            #     #st.markdown('Charts')
            #     display_charts_tab(df, message_index)
                
            with insights_tab:
               # st.markdown('Generated insights based on data',help ='Generated insights based on data')
                # Get the original user query from the message history
                user_query = st.session_state.messages[message_index-1]["content"][0]["text"]
                
                # Generate and display insights usng Snoiwflake COMPLETE
                with st.spinner("Generating insights..."):
                    has_llm_insights = any("llm_insights" in d for d in content)
                    if has_llm_insights is False:
                        if st.session_state.insight_generation:
                            if item.get("insights_enabled", False) is True:
                                #insights = generate_insights_with_snowflake_complete(df, user_query, message_index)
                                json_data = df.to_json(orient='records')
                                #print("json_data",type(json_data))
                                insights = session.call("RAP_UAT.RBP_REPORTS.generate_insights", json_data, user_query)
                            else:
                                insights = "Insights generation has been disabled during this conversation"
                        else:
                            insights = "Insights generation has been disabled during this conversation"
                    else:
                        # Will get the value of 'result_df' if it exists, else None
                        insights = next((d["llm_insights"] for d in content if "llm_insights" in d), None)  
                    st.markdown(insights)
            
            # with LLM_Judge_Rating_tab:
            #      #st.markdown('Rating the Generated insights')
            #      with st.spinner("Evaluating insights..."):
            #         has_judge_result = any("judge_result" in d for d in content)
            #         if has_judge_result is False: 
            #             if st.session_state.insight_generation:
            #                 if item.get("insights_enabled", False) is True:
            #                     rating=judge_LLM(df,insights,user_query,message_index)
            #                 else:
            #                     rating = "Rating generation has been disabled during this conversation"
            #             else:
            #                 rating = "Rating generation has been disabled during this conversation"
            #         else:
            #             # Will get the value of 'result_df' if it exists, else None
            #             rating = next((d["judge_result"] for d in content if "judge_result" in d), None)  
            #         st.markdown(rating)


            # Add feedback section
            # st.divider()
            # st.subheader("Provide Feedback")
            
            # Prepare the user and query information
            # user_name = st.experimental_user["user_name"]
            user_name = session.sql(f'SELECT CURRENT_USER()').collect()[0][0]
            refined_query = st.session_state.messages[message_index]["content"][0]["text"]
            raw_query = st.session_state.messages[message_index-1]["content"][0]["text"]
            
            st.markdown('How do you like the response?')
            # Add thumbs-up and thumbs-down buttons for feedback
            col1, col2 = st.columns([1,1])

            
            # Initialize session state for feedback if not present
            if f"feedback_active_{message_index}" not in st.session_state:
                st.session_state[f"feedback_active_{message_index}"] = False
                st.session_state[f"feedback_type_{message_index}"] = None
                
            # if f"trigger_email_{message_index}" not in st.session_state:
            #     st.session_state[f"trigger_email_{message_index}"] = False
            #     st.session_state[f"email_status_{message_index}"] = None
     
            
            # Thumbs up button
            with col1:
                if st.button("👍 Like it!", key=f"thumbs_up_{message_index}"):
                    st.session_state[f"feedback_active_{message_index}"] = True
                    st.session_state[f"feedback_type_{message_index}"] = "thumbs_up"
                    st.session_state.messages[message_index]["feedback"] = "thumbs_up"
                    # Optional feedback for thumbs up
                    st.rerun()
            
            # Thumbs down button
            with col2:
                if st.button("👎 Need improvment", key=f"thumbs_down_{message_index}"):
                    st.session_state[f"feedback_active_{message_index}"] = True
                    st.session_state[f"feedback_type_{message_index}"] = "thumbs_down"
                    st.session_state.messages[message_index]["feedback"] = "thumbs_down"
                    # Mandatory feedback for thumbs down
                    st.rerun()
                    
            # with col3:
            #     if st.button("✉️ Email", key=f"email_{message_index}"):
            #         st.session_state[f"trigger_email_{message_index}"] = True
            #         st.session_state[f"email_status_{message_index}"] = "triggered"
            #         #st.session_state[f"email_sent_{message_index}"] = "email_to_sent"
            #         st.rerun()
            # if st.session_state[f"trigger_email_{message_index}"]:
            #     # Set it to a non-triggering value BEFORE sending the email.
            #     # This prevents multiple emails on subsequent reruns.
            #     st.session_state[f"trigger_email_{message_index}"] = None
            #     st.session_state[f"email_status_{message_index}"] = "preparing"
            #     file_name, file_url = None, None
            #     try:
            #         # Use a spinner while file is being prepared and uploaded
            #         with st.spinner("Preparing report file..."):
            #              file_name, file_url = send_full_reports(session, message_index)
    
            #         # Check if the function successfully returned file details
            #         if file_name and file_url: # Checks if both are not None and not empty strings
            #             st.session_state[f"email_status_{message_index}"] = "sending"
            #             extracted_texts = [] # Default text if not found
            #             user_prompt_text = ""

            #             # Check if there is a previous message and if it's a user message
            #             if message_index > 0:
            #                 previous_message = st.session_state.messages[message_index - 1]
            #                 # Assuming user messages have role 'user'
            #                 if previous_message.get('role') == 'user':
            #                     content_list = previous_message.get('content')
            #                     for item in content_list:
            #                         extracted_texts.append(item['text'])
            #             if extracted_texts:
            #                 user_prompt_text = "\n".join(extracted_texts)
        
            #             email_body = f'''Hello user,
            #                         We are pleased to know you found the results of RAP Case Query Agent useful. 
            #                         <br>Your Query Was:<br>
            #                         "{user_prompt_text}"
            #                         <br><br>
            #                         Here is the URL to download a CSV version of the result - <a href="{file_url}">{file_name}</a> 
            #                         Thanks for using the AI Agent, hope you found it useful!'''
            #             #to_email = 'SAgarwal@national.aaa.com'
            #             user_name = session.sql(f'SELECT CURRENT_USER()').collect()[0][0]

            #             if user_name: 
            #                 current_user_email = user_name
            #                 to_email = (f"{current_user_email},SAgarwal@national.aaa.com,MJames@national.aaa.com")
            #                 email_subject = 'Snowflake RAP Case Query Agent | Export | {}'.format(datetime.utcnow().strftime('%Y-%m-%d'))
                                
            #                 session.sql("CALL SYSTEM$SEND_EMAIL('RBP_EMAIL_INT', '{}', '{}', '{}', '{}');".format(to_email, email_subject,email_body, "text/html")).collect()
            #                 st.session_state[f"email_status_{message_index}"] = "sent"
            #             else:
            #                 st.write("Email cannot be send outside Snowflake.")
            #         else:
            #             st.session_state[f"email_status_{message_index}"] = "no_data_or_error"
            #     #     st.write("There are no records for the user input to be sent as an email")# Mandatory feedback for thumbs down
            #     except Exception as e:
            #         # Catch any errors during the SYSTEM$SEND_EMAIL call
            #         st.error(f"Failed to send email via SYSTEM$SEND_EMAIL: {e}")
            #         st.session_state[f"email_status_{message_index}"] = "failed" # Final status: failed send
            # current_status = st.session_state.get(f"email_status_{message_index}")
            # if current_status == "triggered":
            #      # This state is very brief, might not be seen
            #      st.info("Email process triggered...")
            # elif current_status == "preparing":
            #      # Spinner handles this, but you could add text
            #      pass # Spinner is running
            # elif current_status == "sending":
            #      # Spinner handles this
            #      pass # Spinner is running
            # elif current_status == "sent":
            #      st.success("✅ Email sent successfully!")
            #      if hasattr(st, "experimental_user") and st.experimental_user and "user_name" in st.experimental_user:
            #         user_name = st.experimental_user["user_name"]
            #      else:
            #         user_name = getpass.getuser()
            #      email_log_history(user_name, 
            #                     None,
            #             st.session_state.messages[message_index-1]["content"][0]["text"],
            #            st.session_state.messages[message_index]["content"][0]["text"],
            #             st.session_state.messages[message_index]["content"][1]["statement"]
            #          ) 
            # elif current_status == "failed":
            #      # Error message shown by the except block
            #      st.error("❌ Email sending failed.")
            # elif current_status == "no_data_or_error":
            #      # Warning/Error shown by send_full_reports or this block
            #      st.warning("ℹ️ Email skipped: No data or error during preparation.")
                # Display feedback form if active
            if st.session_state[f"feedback_active_{message_index}"]:
                feedback_type = st.session_state[f"feedback_type_{message_index}"]
                
                with st.form(key=f"feedback_form_{message_index}"):
                    st.markdown(f"### {'Optional Feedback' if feedback_type == 'thumbs_up' else 'Please tell us what went wrong'}")
                    
                    # Create tabs for different feedback categories
                    feedback_tabs = st.tabs(["Overall", "SQL Query", "Data Results", "Insights"])
                    
                    # Initialize feedback dictionary
                    feedback_data = {
                        "overall": "",
                        "sql": "",
                        "data": "",
                        "chart": "",
                        "insight": "",
                        "llm_rating": ""
                    }
                    
                    # Overall feedback tab
                    with feedback_tabs[0]:
                        feedback_data["overall"] = st.text_area(
                            "Overall feedback",
                            key=f"overall_feedback_{message_index}",
                            placeholder="Please share your overall feedback..."
                        )
                    
                    # SQL Query feedback tab
                    with feedback_tabs[1]:
                        feedback_data["sql"] = st.text_area(
                            "SQL Query feedback",
                            key=f"sql_feedback_{message_index}",
                            placeholder="Please share feedback about the SQL query..."
                        )
                    
                    # Data Results feedback tab
                    with feedback_tabs[2]:
                        feedback_data["data"] = st.text_area(
                            "Data Results feedback",
                            key=f"data_feedback_{message_index}",
                            placeholder="Please share feedback about the data results..."
                        )
                    
                    # Charts feedback tab
                    # with feedback_tabs[3]:
                    #     feedback_data["chart"] = st.text_area(
                    #         "Charts feedback",
                    #         key=f"chart_feedback_{message_index}",
                    #         placeholder="Please share feedback about the charts..."
                    #     )
                    
                    # Insights feedback tab
                    with feedback_tabs[3]:
                        feedback_data["insight"] = st.text_area(
                            "Insights feedback",
                            key=f"insight_feedback_{message_index}",
                            placeholder="Please share feedback about the insights..."
                        )
                    
                    # LLM Rating feedback tab
                    # with feedback_tabs[5]:
                    #     feedback_data["llm_rating"] = st.text_area(
                    #         "Insights Evaluation feedback",
                    #         key=f"llm_rating_feedback_{message_index}",
                    #         placeholder="Please share feedback about the Insights Evaluation..."
                    #     )
                    
                    # Add thumbs-up and thumbs-down buttons for feedback
                    coll1, coll2 = st.columns([1,1], gap='small')
                    
                    # Submit button
                    with coll2:
                        submit_button = st.form_submit_button("Submit Feedback")
                    
                        if submit_button:
                            # For thumbs down, validate that at least one feedback category is filled
                            if feedback_type == "thumbs_down" and all(value.strip() == "" for value in feedback_data.values()):
                                st.error("Please provide feedback in at least one category for 'Not Helpful' ratings.")
                            else:
                                # Retrieve chart details from the current state
                                x_axis = st.session_state.get(f"x_col_select_{message_index}")
                                y_axis = st.session_state.get(f"y_col_select_{message_index}")
                                chart_type = st.session_state.get(f"chart_type_{message_index}")

                                chart_desc = [f"x_axis: {x_axis}\ny_axis: {y_axis}\nchart_type: {chart_type}"]
                                # Retrieve insight text
                                insight_text = next((d.get("llm_insights", "") for d in st.session_state.messages[message_index]['content'] if d.get("type") == "sql"), "")
                                
                                # Save feedback to database
                                save_detailed_feedback(
                                    user_name,
                                    raw_query,
                                    refined_query,
                                    sql,
                                    feedback_type,
                                    feedback_data,
                                    insight_text,
                                    chart_desc
                                )
                                # Reset feedback state
                                st.session_state[f"feedback_active_{message_index}"] = False
                                st.success("Thank you for your feedback!")
                                st.rerun()
                    
                    with coll1:
                        # Cancel button
                        if st.form_submit_button("Cancel"):
                            st.session_state[f"feedback_active_{message_index}"] = False
                            st.session_state[f"feedback_type_{message_index}"] = None
                            st.rerun()
                     

@st.cache_data(show_spinner=False)
def get_query_exec_result(query: str, message_index: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the SQL query and convert the results to a pandas DataFrame.

    Args:
        query (str): The SQL query.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: The query results and the error message.
    """
    global session
    content = st.session_state.messages[message_index]['content']    

    # Check if execution has been cancelled
    if st.session_state.cancel_execution:
        return None, "Operation cancelled by user."

    
    try:
      
        # Execute the query
        df = session.sql(query).to_pandas()
        
        for item in content:
            if item.get("type") == "sql":
                item["result_df"] = df.to_json(orient="records", date_format="iso")
                item["insights_enabled"] = st.session_state.insight_generation
        
        st.session_state.messages[message_index]['content'] = content
        # snowflake_df = session.create_dataframe(df)
        # snowflake_df.write.mode('overwrite').save_as_table('FILES_DB.CSV_FILES.RESULTS')
        return df, None
    except SnowparkSQLException as e:
        for item in content:
            if item.get("type") == "sql":
                item["result_df"] = None
                item["insights_enabled"] = st.session_state.insight_generation
        st.session_state.messages[message_index]['content'] = content
        return None, str(e)

def send_full_reports(session,message_index: int):
    temp_file_path = None
    try:
        # view_name = 'FILES_DB.CSV_FILES.RESULTS' #Provide Fully Qualified Name of the View or Table.
        content = st.session_state.messages[message_index]['content']
        with st.spinner("Sending Email..."):
            #df = any("result_df" in d for d in content)
            df_json = next((d["result_df"] for d in content if "result_df" in d), None)
            df = pd.read_json(df_json)
            if df is None or df.empty:
                st.write(f"Message {message_index}: No DataFrame found in content or DataFrame is empty.")
                st.warning("Cannot send email: No report data available.") # User feedback
                return None, None
            #st.write(type(df))
            # df_json = next((d["result_df"] for d in content if "result_df" in d), None)
            # df = pd.read_json(df_json)
            #df =  session.table(view_name).toPandas()
            stage_name = "@RAP_UAT.RBP_REPORTS.RBP_FILES"
            file_name = f'rap_case_query_agent_result' #Change the FileName Here.
            fd, temp_file_path = tempfile.mkstemp(prefix=file_name, suffix=".csv")
            with tempfile.NamedTemporaryFile(mode="w+t",prefix=file_name, suffix=".csv", delete=False) as t:
                df.to_csv(t.name, index=None)
                session.file.put(t.name, stage_name,auto_compress=False)
                exported_file_name = t.name.split("/")[-1]
                file_sql = f"select GET_PRESIGNED_URL(@RAP_UAT.RBP_REPORTS.RBP_FILES, '{exported_file_name}',8600) as signed_url;"
                print(file_sql)
                signed_url = session.sql(file_sql).collect()[0]['SIGNED_URL']
                st.write("Email sent successfully")
                return exported_file_name, signed_url
    except Exception as e:
        st.write(f"An error occurred while generating/sending report for message {message_index}: {e}")
        st.error(f"An error occurred while preparing the report email: {str(e)}") # User feedback
        return None, None
    finally:
        # Ensure the temporary file is deleted even if errors occur
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # print(f"Cleaned up temporary file: {temp_file_path}") # Optional debug print
            except Exception as cleanup_e:
                print(f"Error cleaning up temporary file {temp_file_path}: {cleanup_e}")
        
def generate_insights_with_snowflake_complete(df: pd.DataFrame, user_query: str, message_index: int) -> str:
    """
    Generate insights from data using Snowflake COMPLETE function.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the query results.
        
    Returns:
        str: The generated insights as text.
    """
    global session
    content = st.session_state.messages[message_index]['content']

    # Check if execution has been cancelled
    if st.session_state.cancel_execution:
        return "Operation cancelled by user."

    
    try:
        if df is None or df.empty:
            return "No data available to generate insights."
        
        # Convert the dataframe to JSON for the prompt
        json_data = df.to_json(orient='records')
        
        # Create a prompt for the Snowflake COMPLETE function
        insight_prompt = f"""
        You are an expert business analyst tasked with generating insights from the provided dataset. The user asked: "{user_query}". Use this question to guide your analysis and produce a concise, structured summary that includes:

        1. **Key Trends and Patterns**: Identify the most prominent trends, changes, or consistencies in the data (e.g., increases/decreases over time, correlations, or recurring values). Focus on what stands out in the context of the user's question.
        2. **Notable Outliers or Anomalies**: Highlight any unusual data points or unexpected results (e.g., extreme values, sudden spikes/drops) relevant to the question, and quantify them if possible.
        3. **Business Implications**: Explain what these findings mean for business performance, operations, or strategy, directly addressing the user's query.
        4. **Action Items**: Suggest specific, actionable steps based on the insights, tailored to the question's intent.

        ### Guidelines:
        - Tailor your analysis to the structure and columns of the data provided (e.g., case_uuid, case_id, call_status, percentages).
        - If the data includes time-based fields (e.g., weeks, months), emphasize temporal trends relevant to the question.
        - If the data includes aggregates (e.g., sums, averages), focus on what these totals or averages reveal in context.
        - If the data is a single row or small set, interpret the specific values in light of the question.
        - Use numbers or examples from the data to support your points.
        - Keep the response concise (150-200 words), clear, and prioritized by importance.
        - If the data is insufficient or unclear for the question, note limitations and suggest what additional data might help.
        STRICTLY while giving your insights , do not add the EPOCH TIME, generate your insights in language/date format which is clearly understandle by human.
        ### Data to Analyze:
        {json_data}
        """
        
        # Call Cortex complete function
        completion_response = cortex.complete(
            model="llama3.1-8b",
            prompt=insight_prompt,
            session=session,
            options=cortex.CompleteOptions(temperature=0.2),
        )
        
        # Store the insights in the session state
        for item in content:
            if item.get("type") == "sql":
                item["llm_insights"] = completion_response
        st.session_state.messages[message_index]['content'] = content
        
        return completion_response
    
    except Exception as e:
        error_message = f"Error generating insights: {str(e)}"
        for item in content:
            if item.get("type") == "sql":
                item["llm_insights"] = error_message
        st.session_state.messages[message_index]['content'] = content
        return error_message


########

def save_detailed_feedback(user_id: str, raw_query: str, refined_query: str, sql_query: str, 
                           feedback_type: str, feedback_data: dict, insight_text: str = None, chart_desc: list = None):
    """
    Save detailed feedback to the database.
    
    Args:
        user_id (str): The ID of the user providing feedback.
        raw_query (str): The raw query string.
        refined_query (str): The user's refined query.
        sql_query (str): The SQL query generated by the analyst.
        feedback_type (str): The type of feedback (thumbs_up/thumbs_down).
        feedback_data (dict): Detailed feedback for each category.
        insight_text (str, optional): The generated insight text.
        chart_desc (list, optional): Description of the chart.
    """
    table_name = 'RAP_UAT.RBP_REPORTS.FEEDBACK_TABLE'
    feedback_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Escape single quotes in strings
        user_id_escaped = user_id.replace("'", "''")
        raw_query_escaped = raw_query.replace("'", "''")
        refined_query_escaped = refined_query.replace("'", "''") 
        sql_query_escaped = sql_query.replace("'", "''")
        feedback_type_escaped = feedback_type.replace("'", "''")
        
        # Escape all feedback data fields
        overall_feedback_escaped = feedback_data["overall"].replace("'", "''")
        sql_feedback_escaped = feedback_data["sql"].replace("'", "''")
        data_feedback_escaped = feedback_data["data"].replace("'", "''")
        chart_feedback_escaped = feedback_data["chart"].replace("'", "''")
        insight_feedback_escaped = feedback_data["insight"].replace("'", "''")
        llm_rating_feedback_escaped = feedback_data["llm_rating"].replace("'", "''")

        # Handle insight_text and chart_desc
        insight_text_escaped = insight_text.replace("'", "''") if insight_text else "NULL"
        chart_desc_escaped = chart_desc[0].replace("'", "''") if chart_desc and chart_desc[0] else "NULL"
        
        query = f"""
            INSERT INTO {table_name} (
                user_id, 
                raw_query, 
                refined_query, 
                sql_query, 
                feedback, 
                feedback_time,
                overall_feedback,
                sql_feedback,
                data_feedback,
                chart_feedback,
                insight_feedback,
                llm_rating_feedback,
                generated_insight,
                chart_desc
            )
            VALUES (
                '{user_id_escaped}', 
                '{raw_query_escaped}', 
                '{refined_query_escaped}', 
                '{sql_query_escaped}',
                '{feedback_type_escaped}',
                '{feedback_time}',
                '{overall_feedback_escaped}',
                '{sql_feedback_escaped}',
                '{data_feedback_escaped}',
                '{chart_feedback_escaped}',
                '{insight_feedback_escaped}',
                '{llm_rating_feedback_escaped}',
                NULLIF('{insight_text_escaped}', 'NULL'),
                NULLIF('{chart_desc_escaped}', 'NULL')
            )
        """
        
        # Execute the query
        session.sql(query).collect()
        
    except SnowparkSQLException as e:
        st.error(f"Error saving feedback: {e}")

def save_query(user_id: str, user_query: str, sql_query: str, raw_query: str):
    """
    Save the user query and SQL query into the query history table.
    
    Args:
        user_id (str): The ID of the user saving the query.
        user_query (str): The user's original query.
        sql_query (str): The SQL query generated by the analyst.
        raw_query (str): The raw query string.
        session (Session): The Snowflake session object.
    """
    table_name = 'RAP_UAT.RBP_REPORTS.QUERY_HISTORY'
    query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Escape single quotes in strings
    user_id_escaped = user_id.replace("'", "''")
    raw_query_escaped = raw_query.replace("'", "''")
    user_query_escaped = user_query.replace("'", "''")
    sql_query_escaped = sql_query.replace("'", "''")
    
    try:
        query = f"""
            INSERT INTO {table_name} (user_id, raw_query, refined_query, sql_query, query_time)
            VALUES ('{user_id_escaped}', '{raw_query_escaped}', '{user_query_escaped}', '{sql_query_escaped}', '{query_time}')
        """
        
        # Execute the query
        session.sql(query).collect()
        st.success("Query and SQL generated Saved")
    except SnowparkSQLException as e:
        st.error(f"Error saving query: {e}")
        
#################################################
def email_log_history(user_id: str, user_email: str, user_query: str, refined_query:str, sql_query: str):
    """
    Save the user query and SQL query into the query history table.
    
    Args:
        user_id (str): The ID of the user saving the query.
        user_query (str): The user's original query.
        sql_query (str): The SQL query generated by the analyst.
        refined_query (str): The refined query string.
        session (Session): The Snowflake session object.
    """
    table_name = 'RAP_UAT.RBP_REPORTS.EMAIL_LOG_HISTORY'
    query_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Escape single quotes in strings
    user_id_escaped = user_id.replace("'", "''")
    user_email_escaped = user_email.replace("'", "''")
    refined_query_escaped = refined_query.replace("'", "''")
    user_query_escaped = user_query.replace("'", "''")
    sql_query_escaped = sql_query.replace("'", "''")
    
    try:
        query = f"""
            INSERT INTO {table_name} (user_id,user_email, user_raw_query, refined_query, sql_query, query_time)
            VALUES ('{user_id_escaped}','{user_email_escaped}', '{user_query_escaped}', '{refined_query_escaped}', '{sql_query_escaped}', '{query_time}')
        """
        
        # Execute the query
        session.sql(query).collect()
        st.success("Email query and user details saved")
    except SnowparkSQLException as e:
        st.error(f"Error saving email logs: {e}")
        
###### judge_LLM
def judge_LLM(df: pd.DataFrame,  insights, user_query, message_index) -> str:
    """
    Judge the quality of the insights generated by the LLM.

    Args:
        original_question (str): The input question provided by the user.
        modified_question (str): The question modified by the LLM.
        sql_query (str): The SQL query generated by the Cortex Analyst.
        dataframe (pd.DataFrame): The dataframe generated from the SQL execution.
        insights (str): The insights generated by the LLM.

    Returns:
        str: The evaluation score or feedback from the LLM.
    """
    global session
    content = st.session_state.messages[message_index]['content']
    
    try:
        if df is None or df.empty:
            return "No data available to judge insights."
        
        # Use only the number of rows specified in the slider
        max_rows = st.session_state.get('max_rows_for_llm', 100)
        df_limited = df.head(max_rows)
        
        # Convert the dataframe to JSON for the prompt
        json_data = df_limited.to_json(orient='records')
        
        # Prepare the evaluation prompt
        prompt = f"""
            Carefully evaluate the generated summary using a multifaceted approach:
            
            Evaluation Criteria:
            1. Relevance (provide score only in between 0-2.5, don't score beyond 2.5):
               - How directly does the summary address the original user query?
               - Are key aspects of the query reflected in the insights?
            
            2. Depth of Insight (provide score only in between 0-2.5, don't score beyond 2.5):
               - Does the summary go beyond surface-level observations?
               - Are there meaningful interpretations of the data?
            
            3. Clarity and Readability (provide score only in between 0-2.5, don't score beyond 2.5):
               - Is the summary clear and easy to understand?
               - Are complex ideas explained coherently?
            
            4. Actionability (provide score only in between 0-2.5, don't score beyond 2.5):
               - Do the insights provide potential actions or recommendations?
               - Can a decision-maker derive value from these insights?
            
            User Query: {user_query}
            SQL Output: {json_data}
            Generated Summary: {insights}

            FORMAT:
                Evaluation Summary
                 1. Relevance
                    - 2 sub points 
                 2. Depth of Insight
                    - 2 sub points 
                 3. Clarity and Readability
                    - 2 sub points 
                 4. Actionability
                    - 2 sub points 
    
                Total Score: 
    
                Breakdown of Scoring:
    
                Improvement Suggestions:
                    - 2-3 sub points 
            
            IMPORTANT: 
            - Strictly always use new line for description.
            - Strictly follow the given FORMAT only.
            - Display heading in bold.
            - Strictly display description in new line.
            - Ensure the sum of score stays within score of 10.
            - Provide a detailed breakdown of your scoring.
            - Explain why you chose each point value.
            - Avoid defaulting to the same score repeatedly for both score i.e total score and Evaluation Criteria scores 
            - If the summary lacks critical elements, explain specific improvements
        """
        
        # Call Cortex complete function
        evaluation = cortex.complete(
            model="llama3.1-70b",
            prompt=prompt,
            session=session,
            #options=cortex.CompleteOptions(temperature=0.1),
        )
        
        # Store the evaluation in the session state
        for item in content:
            if item.get("type") == "sql":
                item["judge_result"] = evaluation
        
        return evaluation
        
    except Exception as e:
        error_message = f"Error judging insights: {str(e)}"
        for item in content:
            if item.get("type") == "sql":
                item["judge_result"] = error_message
        return error_message


#########


def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display the charts tab.
    Args:
        df (pd.DataFrame): The query results.
        message_index (int): The index of the message.
    """
    # Allow user to limit the number of rows for chart rendering
    max_rows_for_chart = st.slider(
        "Max rows for chart", 
        min_value=10, 
        max_value=1000, 
        value=min(20, len(df)), 
        step=10,
        key=f"max_rows_chart_{message_index}"
    )
    
    # Apply the row limit for chart rendering
    df_limited = df.head(max_rows_for_chart)
   
    # Show the number of rows being displayed vs total
    st.caption(f"Showing {len(df_limited)} of {len(df)} rows in chart")
    
    # There should be at least 2 columns to draw charts
    if len(df_limited.columns) >= 2:
                
        all_cols_set = set(df_limited.columns)
        col1, col2 = st.columns(2)
        
        x_col = col1.selectbox(
            "X axis", list(all_cols_set), key=f"x_col_select_{message_index}"
        )
        
        # Make sure to update the available y-axis options when x-axis changes
        y_options = list(all_cols_set.difference({x_col}))
        
        # Use multiselect instead of selectbox to allow multiple Y columns
        y_cols = col2.multiselect(
            "Y axis (select one or more)",
            y_options,
            default=[y_options[0]] if y_options else [],  # Default to first option if available
            key=f"y_cols_select_{message_index}",
        )
        
        chart_type = st.selectbox(
            "Select chart type",
            options=["Line Chart 📈", "Bar Chart 📊", "Scatter Chart"],
            key=f"chart_type_{message_index}",
        )
        
        # Create the chart only if at least one Y column is selected
        if y_cols:
            try:
                if chart_type == "Line Chart 📈":
                    st.line_chart(data=df_limited, x=x_col, y=y_cols)  # Pass list of Y columns
                elif chart_type == "Bar Chart 📊":
                    st.bar_chart(data=df_limited, x=x_col, y=y_cols)  # Pass list of Y columns
                elif chart_type == "Scatter Chart":
                    # For scatter chart, we might need special handling since it typically expects single y value
                    if len(y_cols) == 1:
                        st.scatter_chart(data=df_limited, x=x_col, y=y_cols[0])
                    else:
                        st.warning("Please select only one Y column for scatter charts")
            except Exception as e:
                st.error(f"Error creating chart: {e}")
                st.info("Try selecting different columns or reducing the number of rows.")
        else:
            st.warning("Please select at least one Y column")
    else:
        st.write("At least 2 columns are required to create a chart")


if __name__ == "__main__":
    session = Session.builder.configs(connection_parameters).create()
    main()
    
    #st.write(st.session_state["messages"])
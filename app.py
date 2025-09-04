import re
import os
import ast
import sys
import time
import base64
import plotly
import logging
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
from croniter import croniter
from datetime import datetime
import plotly.graph_objects as go
import snowflake.permissions as perms
from cron_descriptor import get_description
from snowflake.snowpark.functions import col,expr,count
from snowflake.snowpark.context import get_active_session

# Initializing logger object
logger = logging.getLogger("Data_Quality_Control_Logs")

# Setting page config to wide
st.set_page_config(layout="wide", page_title="**DATA QUALITY CONTROL**")

def image_png(file, img_width, img_height):

    loc = os.path.abspath(file)

    # Create figure
    fig = go.Figure()

    # Constants
    img_width = img_width
    img_height = img_height
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
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
    
@st.cache_data(show_spinner=False)
def generate_test_summary(_session, selected_check_id):
    try:
        ca, cb, cd = st.columns([1,0.6,1])
        with cb:
            st.header(":blue[TEST REPORT] :clipboard:")

        if len(selected_check_id)==1:
            updated_dq_tests = _session.sql(f'select CHECK_ID,DATABASE_NAME,SCHEMA_NAME,TABLE_NAME,COLUMN_NAME,CHECK_NAME,RESULT from app_instance_schema.data_quality_checks where check_id  = {selected_check_id[0]}').to_pandas()
            failed_records = _session.sql(f'select * from app_instance_schema.data_quality_failed_records where check_id  = {selected_check_id[0]} order by EXECUTED_DATE desc')
        else:
            updated_dq_tests = _session.sql(f'select CHECK_ID,DATABASE_NAME,SCHEMA_NAME,TABLE_NAME,COLUMN_NAME,CHECK_NAME,RESULT from app_instance_schema.data_quality_checks where check_id in {tuple(selected_check_id)}').to_pandas()
            failed_records = _session.sql(f'select * from app_instance_schema.data_quality_failed_records where check_id in {tuple(selected_check_id)} order by EXECUTED_DATE desc')
        
        passed_tests = updated_dq_tests[updated_dq_tests['RESULT']=='PASS']
        failed_tests = updated_dq_tests[updated_dq_tests['RESULT']=='FAIL']
        passed_tests.reset_index(inplace = True, drop = True)
        failed_tests.reset_index(inplace = True, drop = True)

        st.write('###')

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="**:green[TOTAL INDIVIDUAL CHECKS] :ballot_box_with_check:**", value=str(updated_dq_tests.shape[0]))
        with c2:
            st.metric(label="**:green[TOTAL PASSED CHECKS] :heavy_check_mark:**", value=str(passed_tests.shape[0]))
        with c3:
            st.metric(label="**:green[TOTAL FAILED CHECKS] :x:**", value=str(failed_tests.shape[0]))
        with c4:
            st.metric(label="**:green[TOTAL FAILED RECORDS] :heavy_multiplication_x:**", value=str(failed_records.count()))
        st.write('###')
        
        # Bar charts
        ## DQ check wise failed records
        # check_Wise_failed_records = failed_records.group_by("CHECK_NAME").agg(count("FAILED_RECORDS").alias("FAILED_RECORDS")).to_pandas()
        # passed_failed = updated_dq_tests.groupby(['CHECK_NAME','RESULT'])['CHECK_ID'].count().reset_index()
        # passed_failed.rename(columns={'CHECK_ID': 'COUNT'}, inplace=True)
        # f1,f2 = st.columns(2)
        # with f1:
        #     st.subheader("**Failed records per each Data Quality check**")
        #     custom_color_scale_cont = ['yellow', 'red']
        #     fig1 = px.bar(check_Wise_failed_records, x='CHECK_NAME', y='FAILED_RECORDS', 
        #                   color = 'FAILED_RECORDS',color_continuous_scale = custom_color_scale_cont,
        #                   text_auto = True,
        #                  hover_data=['CHECK_NAME','FAILED_RECORDS'])
        #     fig1.update_xaxes(linewidth=2, linecolor='black', mirror=True)
        #     fig1.update_yaxes(linewidth=2, linecolor='black', mirror=True, type="log")
        #     fig1.update_layout(title='',yaxis_title="FAILED RECORDS",xaxis_title="DQ CHECK",xaxis={'categoryorder':'total descending'})
        #     st.plotly_chart(fig1,use_container_width=True)
        # with f2:
        #     st.subheader("**Passed vs Failed per each Data Quality check**")
        #     #custom_color_scale_disc = ['green','red']
        #     color_mapping = {'PASS': 'green', 'FAIL': 'red'}
        #     fig2 = px.bar(passed_failed, x='CHECK_NAME', y='COUNT', 
        #                   color = 'RESULT',
        #                   #color_discrete_sequence = custom_color_scale_disc,
        #                   color_discrete_map=color_mapping,
        #                   text_auto = True,
        #                  hover_data=['CHECK_NAME','COUNT'])
        #     fig2.update_xaxes(linewidth=2, linecolor='black', mirror=True)
        #     fig2.update_yaxes(linewidth=2, linecolor='black', mirror=True)
        #     fig2.update_layout(title='',yaxis_title="TOTAL TESTS",xaxis_title="DQ CHECK", xaxis={'categoryorder':'total descending'})
        #     st.plotly_chart(fig2,use_container_width=True)
        st.write('---')

        # Displaying passed tests
        if not passed_tests.empty:
            st.success("**✅ The following tests have PASSED**")
            st.dataframe(passed_tests)
            st.write('---')
        # Displaying Failed tests
        if not failed_tests.empty:
            st.error("**🚨 The following tests have FAILED**")
            st.dataframe(failed_tests)    
            st.write('---')
        # Displaying Failed records
        if failed_records.count() > 0:
            st.warning("**⚠️ The following are the DISTINCT FAILED records**")
            st.dataframe(failed_records.drop_duplicates().limit(10000).collect(), use_container_width=True)
    except Exception as E:
        logger.error(f"Unable to generate test summary due to exception : {E}")
        sys.exit(1)
                            
def dq_main(session):
    print("Testing the app for code review")
    t1,t2,t3,t4 = st.columns([2,1,1,0.8])
    with t1:
        st.title(':blue[DATA QC]')
    with t4:
        st.plotly_chart(image_png('kipi-ai_logo.png', img_width = 430, img_height = 150))

    # Global variables
    parameterized_checks = ['length_check','valid_values_check','range_check','regexp_check','custom_check']
    non_parameterized_checks = ['null_check','unique_check','json_check','xml_check']
    SQL_STRING_TYPES = ["char","character","varchar","text","string"]
    SQL_INT_TYPES = ["number","bigint","tinyint","smallint","int","integer","hugeint"]
    SQL_FLOAT_TYPES = ["decimal","dec","numeric","number","real","float","double","double precision"]
    SQL_DATE_TYPES = ["date"]
    SQL_TIME_TYPES = ["time"]
    SQL_TIMESTAMP_TYPES = ["datetime","timestamp","timestamp_ltz","timestamp_ntz","timestamp_tz"]
    
    # getting current db
    # current_db = session.get_current_database()

    # if 'null_check' not in st.session_state:
    #     st.session_state.null_check = None
    # if 'unique_check' not in st.session_state:
    #     st.session_state.unique_check = None
    # if 'length_check' not in st.session_state:
    #     st.session_state.length_check = None
    # if 'length_check_str_params' not in st.session_state:
    #     st.session_state.length_check_str_params = None
    # if 'valid_values_check' not in st.session_state:
    #     st.session_state.valid_values_check = None
    # if 'valid_values_check_str_params' not in st.session_state:
    #     st.session_state.valid_values_check_str_params = None
    # if 'range_check' not in st.session_state:
    #     st.session_state.range_check = None
    # if 'range_check_str_params' not in st.session_state:
    #     st.session_state.range_check_str_params = None

    # Store form submitted value in a session state variable
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Store form submitted value in a session state variable
    if 'task_form_submitted' not in st.session_state:
        st.session_state.task_form_submitted = False
        
    # Store selected_table value in a session state variable
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None

    # Store pandas_df in a session state variable
    if 'checks_df' not in st.session_state:
        st.session_state.checks_df = None

    # Initialize a session state variable to store the text inputs
    # if 'text_inputs' not in st.session_state:
    #     st.session_state.text_inputs = [""]

    # creating tabs
    tabs = st.tabs(['**About :pushpin:**','**Overview** :wave:','**Timeliness Assessment** :stopwatch:','**Build data quality checks** :hammer:','**Execute data quality checks** :heavy_check_mark:','**View Past Results :memo:**', '**Scheduler** :clock4:', '**Privacy and Debugging** :lock:'])
    
    #############
    ### About ###
    #############
    with tabs[0]:
        # Displaying the images and overview
        c9, c10 = st.columns([0.75,2])
        with c9:
            st.write('###')
            st.plotly_chart(image_png('Data-QC-Logo-1.png', img_width = 500, img_height = 500))
        with c10:
            st.write('###')
            st.write("**Welcome to :blue[Data QC] :snowflake:, a powerful solution designed to revolutionize the way you manage and improve data quality within the Snowflake environment.**")
            st.write("➥ In today's data-driven world, the accuracy and reliability of your data are paramount. Data inaccuracies can lead to misguided decisions, operational inefficiencies, and risks in compliance. Data Quality Control, built using snowpark for python and streamlit platform, provides a seamless integration with Snowflake, empowers you to take control of your data quality with simplicity and precision.")
            st.write("➥ Join us on a journey to elevate your data quality standards, drive informed decisions, and unlock the full potential of your data. Say goodbye to data headaches and hello to data excellence.")
        st.write('---')

        c11, c12 = st.columns([2,1])
        with c11:
            st.write("**➥ With :blue[Data QC's] intuitive interface and robust features, you can:**")
            st.write("ㅤㅤㅤㅤㅤ❖ **Assess Data:** Gain deep insights into your data's timeliness.")
            st.write("ㅤㅤㅤㅤㅤ❖ **Validate Data:** Implement predefined and custom quality checks and validations.")
            st.write("ㅤㅤㅤㅤㅤ❖ **Analyse Data:** Execute custom data quality checks and analyse the results.")
            st.write("ㅤㅤㅤㅤㅤ❖ **Orchestrate:** Schedule your important data quality checks with ease.")
        with c12:
            st.plotly_chart(image_png('data-quality.png', img_width = 450, img_height = 450))
            
        st.write('---')
    
    ####################
    ### Overview tab ###
    ####################
    with tabs[1]:
        st.write('##')
        dq_checks_df = session.table('DATA_QUALITY_CHECKS').to_pandas()
        dq_failed_df = session.table('DATA_QUALITY_FAILED_RECORDS')
        dq_tasks_df = session.table('"APP_INSTANCE_TASK_SCHEMA"."DATA_QUALITY_TASKS"')
        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.4,0.6,1.4,0.6,1.4])
            with c1:
                st.plotly_chart(image_png('check_colored.png', img_width = 200, img_height = 200))
            with c2:
                st.subheader(f":green[Total Checks]")
                st.subheader(f":black[**{dq_checks_df.shape[0]}**]")
                #st.metric(label="**:green[TOTAL CHECKS] :ballot_box_with_check:**", value=str(dq_checks_df.shape[0]))
                
            with c3:
                st.plotly_chart(image_png('execute_colored_2.png', img_width = 200, img_height = 200))
            with c4:
                st.subheader(f":green[Executed Checks]")
                st.subheader(f":black[**{dq_checks_df[~dq_checks_df['LAST_EXECUTED_DATE'].isna()].shape[0]}**]")
                # st.metric(label="**:green[TOTAL EXECUTED CHECKS] :heavy_check_mark:**", value=str(dq_checks_df[~dq_checks_df['LAST_EXECUTED_DATE'].isna()].shape[0]))
            
            with c5:
                st.plotly_chart(image_png('unexecuted_colored_2.png', img_width = 180, img_height = 200))
            with c6:
                st.subheader(f":green[Unexecuted Checks]")
                st.subheader(f":black[**{dq_checks_df[dq_checks_df['LAST_EXECUTED_DATE'].isna()].shape[0]}**]")

        st.write('##')

        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.4,0.6,1.4,0.6,1.4])
            with c1:
                st.plotly_chart(image_png('pass_colored_2.png', img_width = 200, img_height = 200))
            with c2:
                st.subheader(f":green[Passed Checks]")
                st.subheader(f":black[**{dq_checks_df[dq_checks_df['RESULT']=='PASS'].shape[0]}**]")
                #st.metric(label="**:green[TOTAL CHECKS] :ballot_box_with_check:**", value=str(dq_checks_df.shape[0]))
                
            with c3:
                st.plotly_chart(image_png('fail_colored_2.png', img_width = 200, img_height = 200))
            with c4:
                st.subheader(f":green[Failed Checks]")
                st.subheader(f":black[**{dq_checks_df[dq_checks_df['RESULT']=='FAIL'].shape[0]}**]")
                # st.metric(label="**:green[TOTAL EXECUTED CHECKS] :heavy_check_mark:**", value=str(dq_checks_df[~dq_checks_df['LAST_EXECUTED_DATE'].isna()].shape[0]))
            
            with c5:
                st.plotly_chart(image_png('incorrect_colored_2.png', img_width = 180, img_height = 200))
            with c6:
                st.subheader(f":green[Failed Records]")
                st.subheader(f":black[**{dq_failed_df.count()}**]")

        st.write('##')

        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.4,0.6,1.4,0.6,1.4])
            with c1:
                st.plotly_chart(image_png('table_colored_2.png', img_width = 200, img_height = 200))
            with c2:
                st.subheader(f":green[Tables Covered]")
                st.subheader(f":black[**{dq_checks_df[['DATABASE_NAME','SCHEMA_NAME','TABLE_NAME']].drop_duplicates().shape[0]}**]")
                #st.metric(label="**:green[TOTAL CHECKS] :ballot_box_with_check:**", value=str(dq_checks_df.shape[0]))
                
            with c3:
                st.plotly_chart(image_png('column_colored.png', img_width = 200, img_height = 200))
            with c4:
                st.subheader(f":green[Columns Covered]")
                st.subheader(f":black[**{dq_checks_df[['DATABASE_NAME','SCHEMA_NAME','TABLE_NAME','COLUMN_NAME']].drop_duplicates().shape[0]}**]")
                # st.metric(label="**:green[TOTAL EXECUTED CHECKS] :heavy_check_mark:**", value=str(dq_checks_df[~dq_checks_df['LAST_EXECUTED_DATE'].isna()].shape[0]))
            
            with c5:
                st.plotly_chart(image_png('task_colored_2.png', img_width = 180, img_height = 200))
            with c6:
                st.subheader(f":green[Running Tasks]")
                st.subheader(f":black[**{dq_tasks_df.filter(col('STATUS')=='resumed').count()}**]")
        
        st.write('---')
            
    ######################
    ### Timeliness tab ###
    ######################
    with tabs[2]:
        # st.write("**➥This section shows the tables that have been modified most recently. This information can be used to assess the freshness and timeliness of data.**")
        # database_query = f"select distinct table_catalog, table_schema, name as table_name from snowflake.account_usage.GRANTS_TO_ROLES where deleted_on is NULL and granted_on = 'TABLE' and grantee_name = 'data_quality_control_app_role' order by name"
        # #database_names = session.sql(database_query).to_pandas()
        # database_names = pd.DataFrame(session.sql('show tables history in account').collect())
        # # Trap for dropped tables
        # database_names= database_names[pd.isna(database_names['dropped_on'])]
        # database_names.rename(columns={'database_name':'TABLE_CATALOG', 'schema_name':'TABLE_SCHEMA','name':'TABLE_NAME'}, inplace=True)
        # database_list = sorted(database_names['TABLE_CATALOG'].unique().tolist())
        st.write("**➥This section shows the tables that have been modified most recently. This information can be used to assess the freshness and timeliness of data.**")
        
        # database_query = f"select distinct table_catalog, table_schema, name as table_name from snowflake.account_usage.GRANTS_TO_ROLES where deleted_on is NULL and granted_on = 'TABLE' and grantee_name = 'data_quality_control_app_role' order by name"
        database_query = f"select distinct table_catalog, table_schema, name as table_name from snowflake.account_usage.GRANTS_TO_ROLES where deleted_on is NULL and granted_on in ('TABLE', 'VIEW') and grantee_name = 'data_quality_control_app_role' order by name"
        #database_names = session.sql(database_query).to_pandas()
        # database_names = pd.DataFrame(session.sql('show tables history in account').collect())
        # # Trap for dropped tables
        # database_names= database_names[pd.isna(database_names['dropped_on'])]
        # database_names.rename(columns={'database_name':'TABLE_CATALOG', 'schema_name':'TABLE_SCHEMA','name':'TABLE_NAME'}, inplace=True)
        # database_list = sorted(database_names['TABLE_CATALOG'].unique().tolist())
        tables_df = pd.DataFrame(session.sql('SHOW TABLES IN ACCOUNT').collect())
        tables_df.rename(columns={'database_name':'TABLE_CATALOG', 'schema_name':'TABLE_SCHEMA','name':'TABLE_NAME'}, inplace=True)
        st.write("TABLES DF")
        st.dataframe(tables_df)
        # tables_df = tables_df[pd.isna(tables_df['dropped_on'])]
        
        # Fetch all views
        views_df = pd.DataFrame(session.sql('SHOW VIEWS IN ACCOUNT').collect())
        views_df.rename(columns={'database_name':'TABLE_CATALOG', 'schema_name':'TABLE_SCHEMA','name':'TABLE_NAME'}, inplace=True)
        st.write("VIEWS DF")
        st.dataframe(views_df)
        views_df['rows'] = 0  # Initialize with 0 or a placeholder
        views_df = views_df[(views_df["TABLE_SCHEMA"] != "INFORMATION_SCHEMA") & (views_df["TABLE_CATALOG"] != "SNOWFLAKE")]

        # Calculate row count for each view and update the DataFrame
        for index, row in views_df.iterrows():
            try:
                count_query = f'SELECT COUNT(*) FROM "{row["TABLE_CATALOG"]}"."{row["TABLE_SCHEMA"]}"."{row["TABLE_NAME"]}"'
                view_row_count = session.sql(count_query).collect()[0][0]
                views_df.loc[index, 'rows'] = view_row_count
            except Exception as e:
                print(f"Could not get row count for view {row['TABLE_NAME']}: {e}")
                views_df.loc[index, 'rows'] = -1 # Use a sentinel value for errors
        
        st.write("VIEWS DF UPDATED")
        st.dataframe(views_df)

        # Add a 'TABLE_TYPE' column to distinguish between tables and views
        tables_df['TABLE_TYPE'] = 'BASE TABLE'
        views_df['TABLE_TYPE'] = 'VIEW'
        
        # Concatenate the two DataFrames into one
        database_names = pd.concat([tables_df, views_df], ignore_index=True)
        
        # --- Step 3: Get the list of unique databases ---
        database_list = sorted(database_names['TABLE_CATALOG'].unique().tolist())
        if len(database_list) > 0:
            tc1, tc2 = st.columns(2)
            with tc1:
                selected_database_tb0 = st.selectbox(
                    '***Choose database name***',
                    database_list)
                
            schema_list_tb0=sorted(database_names.loc[database_names['TABLE_CATALOG']==selected_database_tb0]['TABLE_SCHEMA'].unique().tolist())
            with tc2:
                selected_schema_tb0 = st.selectbox(
                    '***Choose schema name***',
                    schema_list_tb0)
            freshness = session.sql(f"""select distinct table_catalog as database_name, table_schema as schema_name, table_name,last_altered, datediff(day, last_altered, current_timestamp()) as DAYS_SINCE_LAST_DML from "{selected_database_tb0}"."INFORMATION_SCHEMA"."TABLES" where database_name = '{selected_database_tb0}' and schema_name = '{selected_schema_tb0}' order by DAYS_SINCE_LAST_DML nulls last""").to_pandas()
            freshness['TABLE_TYPE'] = ['Hot' if x<=7 else 'Warm' if 8<=x<=31 else 'Cold' for x in freshness['DAYS_SINCE_LAST_DML']]
            # Displaying KPI metrics for timeliness assessment
            st.write("---")
            e1,e2,e3 = st.columns([0.4,1,0.3])
            with e2:
                st.header(":blue[TIMELINESS ASSESSMENT REPORT] :clipboard:")
            st.write('###')
            a1,a2,a3,a4 = st.columns(4)
            with a1:
                st.metric(label=":green[**TOTAL TABLES**] :books:", value=str(freshness['TABLE_NAME'].count()))
            with a2:
                st.metric(label=":green[**TOTAL HOT TABLES**] :closed_book:", value=str(freshness[freshness['DAYS_SINCE_LAST_DML']<=7]['TABLE_NAME'].count()))
            with a3:
                st.metric(label=":green[**TOTAL WARM TABLES**] :orange_book:", value=str(freshness[(freshness['DAYS_SINCE_LAST_DML']>=8) & (freshness['DAYS_SINCE_LAST_DML']<31)]['TABLE_NAME'].count()))
            with a4:
                st.metric(label=":green[**TOTAL COLD TABLES**] :blue_book:", value=str(freshness[freshness['DAYS_SINCE_LAST_DML']>31]['TABLE_NAME'].count()))

            st.write('###')
            a5,a6 = st.columns([1,0.4])
            with a5:
                st.write("**➥ Tabular Summary**")
                st.dataframe(freshness, use_container_width=True)
            with a6:
                st.write('###')
                st.write("###")
                st.caption("***:red[Hot Tables]     → Tables that were modified in last 1 week***")
                st.caption("***:orange[Warm Tables] → Tables that were modified in last 1 month***")
                st.caption("***:blue[Cold Tables]   → Tables that haven't been modified in last 1 month***")
                st.write("###")
                st.write("**Recommendation**:bulb:")
                st.caption("**➥ Generally tables which have been modified recently (hot tables) are good candidates for data quality checks, along with warm tables. However, feel free to add data quality checks on cold tables if needed.**")
        else:
            st.info("Please grant access to database, schema and tables to the application using below queries")
            st.code("""
            GRANT USAGE ON DATABASE <DATABASE_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT USAGE ON SCHEMA <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT SELECT ON ALL TABLES IN <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            """, language='sql')
            
    #################
    ### Build tab ###
    #################  
    with tabs[3]:
        #st.write("**➥ To add data quality checks on a table, pls grant select privilege on the table to the application using the below mentioned commands**")
        #st.code("""
        #GRANT USAGE ON DATABASE <DATABASE_NAME> TO APPLICATION <APPLICATION_NAME>;
        #GRANT USAGE ON SCHEMA <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
        #GRANT SELECT ON ALL TABLES IN <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
        #""", language='sql')
        #st.write('---')
        st.write("**➥ The build page is used to create data quality checks on a table, Make sure to grant read-only access on required tables to the application**")
        if len(database_list) > 0:
            st.write(database_list)
            d,s,t = st.columns(3)
            # database dropdown
            with d:
                selected_database = st.selectbox(
                    '***Select database name***',
                    database_list)
                 
            # get schemas with selected database
            #schema_query = f"select distinct table_schema from snowflake.account_usage.columns where table_catalog='{selected_database}' order by table_schema"
            #schema_name = session.sql(schema_query).to_pandas()
            schema_list=sorted(database_names.loc[database_names['TABLE_CATALOG']==selected_database]['TABLE_SCHEMA'].unique().tolist())
            st.write(f"SCHEMA LIST: {schema_list}")
            # schema dropdown
            with s:
                selected_schema = st.selectbox(
                    '***Select schema name***',
                    schema_list,
                    index=0)
            
            # get tables within selected database and schema
            # table_query = f"select distinct table_name from snowflake.account_usage.columns where table_catalog = '{selected_database}' and table_schema='{selected_schema}' order by table_name"
            # table_list = session.sql(table_query).to_pandas()['TABLE_NAME'].tolist()
            table_list=sorted(database_names.loc[(database_names['TABLE_CATALOG']==selected_database) & (database_names['TABLE_SCHEMA']==selected_schema)]['TABLE_NAME'].unique().tolist())
            st.write(f"table_list: {table_list}")
            # table dropdown
            with t:
                selected_table = st.selectbox(
                    '***Select table name***',
                    table_list,
                    #index=table_list.index(st.session_state.selected_table) if st.session_state.selected_table else 0
                )

            st.write('---')
            if selected_table:
                st.session_state.selected_table=selected_table
                st.write(f"**➥ Preview of :blue[{selected_table}] table**")
                # table preview
                # preview_df = session.sql(f'select * from "{selected_database}"."{selected_schema}"."{selected_table}"').limit(5)
                preview_df = session.table(f'"{selected_database}"."{selected_schema}"."{selected_table}"').limit(5)
                if preview_df.count() > 0:
                    st.dataframe(preview_df, use_container_width=True)
                else:
                    st.info(f"**The table :green[{selected_table}] does not contain any data. You can still create data quality checks on this table.**")
                # existing checks on the selected table
                st.write('###')
                existing_checks = session.sql(f"select * from data_quality_checks where table_name = '{selected_table}'")
                if existing_checks.count() > 0:
                    st.write(f"**➥ The :blue[{selected_table}] table has the following data quality checks defined**")
                    st.dataframe(existing_checks)
                else:
                    st.write(f"**➥ The :blue[{selected_table}] table has no existing data quality checks**")
                st.write('---')
            
            # get tables within selected database and schema
            column_query = f"select distinct COLUMN_NAME from {selected_database}.information_schema.columns where table_catalog = '{selected_database}' and table_schema='{selected_schema}' and table_name='{selected_table}' order by COLUMN_NAME"
            column_name = session.sql(column_query).to_pandas()
            column_list=column_name['COLUMN_NAME'].tolist()

            def convert_to_string(value):
                if value is None:
                    return None
                return str(value)
            
            def construct_test_df(dq_checks:dict, selected_database, selected_schema, selected_table):
                new_dict = {}
                keys_to_del = []
                # Given information
                for key, value in dq_checks.items():
                    if key in parameterized_checks and len(dq_checks[f'{key}_str_params'])>0:
                        if key.lower()=='length_check':
                            input_params_list = [int(item) for item in dq_checks[f'{key}_str_params'].split(',') if len(item.strip()) > 0]
                            new_dict[key]= dict(zip(dq_checks[key], input_params_list))
                            keys_to_del.append(f'{key}_str_params')
                        elif key.lower() in ['valid_values_check','range_check']:
                            input_params_list = ast.literal_eval(dq_checks[f'{key}_str_params'])
                            # checking whether input is a list (only applies to single column case)
                            if isinstance(input_params_list,list):
                                input_params_list = [input_params_list]
                            # checking whether input is a tuple (only applies to multi column case)
                            elif isinstance(input_params_list,tuple):
                                pass
                            new_dict[key]= dict(zip(dq_checks[key], input_params_list))
                            keys_to_del.append(f'{key}_str_params')
                        elif key.lower() in ['regexp_check', 'custom_check']:
                            input_params_list = dq_checks[f'{key}_str_params'].split(',')
                            new_dict[key] = dict(zip(dq_checks[key], input_params_list))
                            keys_to_del.append(f'{key}_str_params')
                    else:
                        new_dict[key]=value

                for k in keys_to_del:
                    del new_dict[f'{k}']
            
                # Creating pandas df
                check_id = session.sql('select max(check_id) as CID from DATA_QUALITY_CHECKS').collect()
                if check_id[0]['CID']:
                    check_id = check_id[0]['CID'] + 1
                else:
                    check_id = 1001
                database_name = selected_database
                schema_name = selected_schema
                table_name = selected_table
            
                # Create a list to store data for the DataFrame
                data = []
                
                # Iterate through each data quality check and its column list or parameters
                for dq_check, col_list_or_params in new_dict.items():
                    if isinstance(col_list_or_params, (list,str)) and col_list_or_params:
                        col_list = col_list_or_params
                        is_parameterized_check = False
                        input_parameter = None
                    elif isinstance(col_list_or_params, dict) and col_list_or_params:
                        col_list = list(col_list_or_params.keys())
                        is_parameterized_check = True
                        input_parameter = col_list_or_params  # Input parameter is a dictionary
                    else:
                        continue  # Skip empty checks
                
                    for column_name in col_list:
                        data.append({
                            "CHECK_ID": check_id,
                            "DATABASE_NAME": database_name,
                            "SCHEMA_NAME": schema_name,
                            "TABLE_NAME": table_name,
                            "COLUMN_NAME": column_name,
                            "CHECK_NAME": dq_check,
                            "IS_PARAMETERIZED_CHECK": is_parameterized_check,
                            "INPUT_PARAMETER": input_parameter.get(column_name, None) if is_parameterized_check else None,
                            "RESULT": None
                        })
                # Create a Pandas DataFrame from the list of dictionaries
                dataframe = pd.DataFrame(data)
                # Adding created date column
                dataframe['CREATED_DATE'] = pd.Timestamp.now()
                dataframe['CREATED_DATE']  = pd.to_datetime(dataframe['CREATED_DATE'] )
                dataframe['CREATED_DATE']  = dataframe['CREATED_DATE'] .dt.tz_localize("UTC")
                if not dataframe.empty:
                    dataframe['INPUT_PARAMETER'] = dataframe['INPUT_PARAMETER'].apply(convert_to_string)
                st.session_state.checks_df=dataframe
                return dataframe


            # storing form values
            def update_session_state(form_dict):
                for k,v in form_dict.items():
                    st.session_state[k] = v

            # storing form values
            def remove_session_state(form_dict):
                for k in form_dict:
                    del st.session_state[k]
                    
            ##############
            # Input Form #
            ##############
            c99, c100 = st.columns([1.5,1])
            with c99:
                st.write('**➥Add data quality checks using the below input form**')
            # with c100:
            #     st.caption('● Pls provide input parameters for parameterized checks')
            error_flag = False
            with st.form('Data quality check selection form', clear_on_submit=False):
                with st.expander('Non-Parameterized data quality checks'):
                    null_check = st.multiselect('Null check', column_list)
                    unique_check = st.multiselect('Unique check', column_list)
                    json_check = st.multiselect('JSON check', column_list)
                    xml_check = st.multiselect('XML check', column_list)
                    #st.session_state.null_check = null_check
                    #st.session_state.unique_check = unique_check
                with st.expander('Parameterized data quality checks'):
                    ##### Length check
                    c1,c2 = st.columns([1,1])
                    with c1:
                        length_check = st.multiselect('Length check', column_list)
                    with c2:
                        length_check_str_params = st.text_input('Input params for length check :red[✱]',help='Enter values separated by comma. Ex- :green[**For a single column : 20, For multiple columns : 20,30 etc**]', key='length_params')
                    #st.session_state.length_check = length_check
                    #st.session_state.length_check_str_params = length_check_str_params
                  
                    ##### Valid values check
                    c3,c4 = st.columns([1,1])
                    with c3:
                        valid_values_check = st.multiselect('Valid values check', column_list)
                    with c4:
                        valid_values_check_str_params = st.text_input('Input params for valid values check :red[✱]',help="Enter list of values separated by comma. Ex- :green[**For single column : ['male','female'], For multiple columns : ['male','female'],['abc','xyz']**]", key='valid_params')
                    #st.session_state.valid_values_check = valid_values_check
                    #st.session_state.valid_values_check_str_params = valid_values_check_str_params
                    
                    ##### Range check
                    c5,c6 = st.columns([1,1])
                    with c5:
                        range_check = st.multiselect('Range check', column_list)
                    with c6:
                        range_check_str_params = st.text_input('Input params for range check :red[✱]', help="Enter list of values (lower and upper limits) separated by comma. Ex- :green[**For single column : [1,10], For multiple columns : [1,10],[2,20]**]", key='range_params')
                    #st.session_state.range_check = range_check
                    #st.session_state.range_check_str_params = range_check_str_params
                    
                    #### Regexp check
                    c7,c8 = st.columns([1,1])
                    with c7:
                        regexp_check = st.multiselect('Regexp check', column_list)
                    with c8:
                        regexp_check_str_params = st.text_input('Input params for regexp check :red[✱]', help="Enter values separated by comma. Ex- :green[**For single column : ^[a-zA-Z0-9_.]\$ , For multiple columns : ^[a-zA-Z0-9_.]$,[a-zA-Z]**]", key='regexp_params')

                    #### custom check
                    c9,c10 = st.columns([1,1])
                    with c9:
                        custom_check = st.multiselect('Custom check', column_list, disabled=False)
                    with c10:
                        custom_check_str_params = st.text_input('Input params for custom check :red[✱]',disabled=False, key='custom_params', help="""Enter SQL expressions separated by comma. Column names are case sensitive, Hence enclose column names within the SQL expressions in double quotes. Ex- :green[**For single column : "Order_Date" <= "Delivery_Date" , For multiple columns : "Order_Date" <= "Delivery_Date","Gender" = 'female'**]""")
           
                st.session_state.submitted = st.form_submit_button(label="**:green[Submit]**")
                
            if st.session_state.submitted:
                form_dict = {
                    'null_check':null_check,
                    'unique_check':unique_check,
                    'json_check':json_check,
                    'xml_check':xml_check,
                    'length_check':length_check,
                    'length_check_str_params':length_check_str_params,
                    'valid_values_check':valid_values_check,
                    'valid_values_check_str_params':valid_values_check_str_params,
                    'range_check':range_check,
                    'range_check_str_params':range_check_str_params,
                    'regexp_check':regexp_check,
                    'regexp_check_str_params':regexp_check_str_params,
                    'custom_check':custom_check,
                    'custom_check_str_params':custom_check_str_params
                }
                #update_session_state(form_dict)
                #st.write(ast.literal_eval(valid_values_check_str_params), type(ast.literal_eval(valid_values_check_str_params)))
                # Store values in the session state
                st.session_state.selected_table = selected_table
                # st.session_state.null_check = null_check
                # st.session_state.unique_check = unique_check
                # st.session_state.length_check = length_check
                # st.session_state.length_check_str_params = length_check_str_params
                # st.session_state.valid_values_check = valid_values_check
                # st.session_state.valid_values_check_str_params = valid_values_check_str_params
                # st.session_state.range_check = range_check
                # st.session_state.range_check_str_params = range_check_str_params
                
                #### perform input validations for length check
                if length_check_str_params and length_check:
                    try:
                        input_params_list = [int(item) for item in length_check_str_params.split(',') if len(item.strip()) > 0]
                        
                        if len(length_check) != len(input_params_list):
                            error_flag=True
                            logger.error("Number of columns and input parameters do not match for length check")
                            st.error(f'**️🚨 Number of columns and input parameters do not match for length check**')
                    except:
                        error_flag=True
                        logger.error(f"Please provide valid numeric input parameters instead of {length_check_str_params},  for length check. Ex: :green[80] (If using single column) or :green[80,20] (If using multiple columns)")
                        st.error(f"**🚨 Please provide valid numeric input parameters for length check. Ex: :green[80] (If using single column) or :green[80,20] (If using multiple columns)**")
                elif length_check and not length_check_str_params:
                    error_flag=True
                    logger.error("Please provide input parameters for Length check")
                    st.error(f'**🚨 Please provide input parameters for Length check**')
                elif length_check_str_params and len(length_check)==0:
                    error_flag=True
                    logger.error("Please select columns for Length checks")
                    st.error(f'**🚨 Please select columns for Length checks**')
                else:
                    pass
                            
                #### perform input validations for valid values check
                # Creating a dictionary of col_names and dtypes from a list of lists
                preview_df_dtypes = {}
                for item in preview_df.dtypes:
                    key, value = item
                    preview_df_dtypes[key.replace('"','')] = value
                    
                if valid_values_check_str_params and valid_values_check:
                    valid_values_params_list =[]

                    # trap for cases where input params is not enclosed in single quotes. Ex - [male,female] instead of  ['male','female']
                    try:
                        input_params_list=ast.literal_eval(valid_values_check_str_params)
                        #st.write(f"input_params_list after eval : {input_params_list}"
                    
                        # checking whether input is a list (only applies to single column case)
                        if isinstance(input_params_list,list):
                            valid_values_params_list.append(input_params_list)
                            #st.write(f"input_params_list is a list : {valid_values_params_list}")
                        # checking whether input is a tuple (only applies to multi column case)
                        elif isinstance(input_params_list,tuple):
                            #st.write(f"input_params_list is a tuple : {valid_values_params_list}")
                            valid_values_params_list = input_params_list
                        else:
                            error_flag=True
                            logger.error(f"Unexpected Input type of valid values params : {type(input_params_list)}")
                            st.error(f'**🚨 Unexpected Input type of valid values params : {type(input_params_list)}**')
                        
                        # validating column names with input params length
                        if len(valid_values_check) != len(valid_values_params_list):
                            error_flag=True
                            logger.error("Number of columns and input parameters do not match for valid values check")
                            st.error(f'**🚨 Number of columns and input parameters do not match for valid values check**')
            
                        # Type casting the input params to the type of the selected columns
                        # Ex - a varchar column in snowflake expects the in condition to have values of same dtype
                        # customer_name column of varchar dtype would only work when customer_name in ('mary','john')
                        # it does not work and throws an error when customer_name in (15,20).
                        for column_name, params_list in dict(zip(valid_values_check,valid_values_params_list)).items():
                            #st.write(f"Inside for loop, column_name : {column_name}, params_list : {params_list}")
                            col_dtype = preview_df_dtypes[column_name]
                            try:
                                expected_dtype=None
                                if '(' in col_dtype:
                                    col_dtype=col_dtype.split('(')[0]
                                    if col_dtype.lower() in SQL_STRING_TYPES:
                                        expected_dtype = "string"
                                        valid_values_list = [str(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_INT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [int(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_FLOAT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [float(i) for i in params_list]
                                    elif col_dtype.lower()  in SQL_DATE_TYPES:
                                        expected_dtype = "date"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIME_TYPES:
                                        expected_dtype = "time"
                                        valid_values_list = [datetime.strptime(i, '%H:%M:%S') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIMESTAMP_TYPES:
                                        expected_dtype = "datetime"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in params_list]
                                else:
                                    if col_dtype.lower() in SQL_STRING_TYPES:
                                        expected_dtype = "string"
                                        valid_values_list = [str(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_INT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [int(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_FLOAT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [float(i) for i in params_list]
                                    elif col_dtype.lower()  in SQL_DATE_TYPES:
                                        expected_dtype = "date"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIME_TYPES:
                                        expected_dtype = "time"
                                        valid_values_list = [datetime.strptime(i, '%H:%M:%S') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIMESTAMP_TYPES:
                                        expected_dtype = "datetime"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in params_list]
                            except Exception as E:
                                error_flag=True
                                logger.error(f"Please enter input params of type {col_dtype}, for column {column_name} specified in valid values check")
                                st.error(f"**🚨 Please enter input params of type {col_dtype}, for column {column_name} specified in valid values check**")
                                #st.stop()

                    except:
                        error_flag=True
                        logger.error(f"Please enter correct input parameters instead of {valid_values_check_str_params}, for valid values check. If using string values for input parameters, make sure to enclose the values in single quotes")
                        st.error(f"**🚨 Please enter correct input parameters for valid values check. If using string values for input parameters, make sure to enclose the values in single quotes**")
                        #st.stop()
                        
                        
                elif valid_values_check and not valid_values_check_str_params:
                    error_flag=True
                    logger.error("Please provide input parameters for Valid values check")
                    st.error(f"**🚨 Please provide input parameters for Valid values check**")
                elif valid_values_check_str_params and len(valid_values_check)==0:
                    error_flag=True
                    logger.error("Please select columns for Valid values check")
                    st.error(f"**🚨 Please select columns for Valid values check**")
                else:
                    pass
                    
                #### perform input validations for range check
                # Creating a dictionary of col_names and dtypes from a list of lists
                preview_df_dtypes = {}
                for item in preview_df.dtypes:
                    key, value = item
                    preview_df_dtypes[key.replace('"','')] = value
                    
                if range_check_str_params and range_check:
                    range_params_list =[]

                    try:
                        input_params_list=ast.literal_eval(range_check_str_params)
                        
                        # checking whether input is a list (only applies to single column case)
                        if isinstance(input_params_list,list):
                            range_params_list.append(input_params_list)
                        # checking whether input is a tuple (only applies to multi column case)
                        elif isinstance(input_params_list,tuple):
                            range_params_list = input_params_list
                        else:
                            error_flag=True
                            logger.error(f"Unexpected Input type of Range params : {type(input_params_list)}")
                            st.error(f'**🚨 Unexpected Input type of Range params : {type(input_params_list)}**')
                            
                        # validating column names with input params length
                        if len(range_check) != len(range_params_list):
                            error_flag=True
                            logger.error("Number of columns and input parameters do not match for Range check")
                            st.error(f'**🚨 Number of columns and input parameters do not match for Range check**')

                        # Type casting the input params to the type of the selected columns
                        for column_name, params_list in dict(zip(range_check,range_params_list)).items():
                            col_dtype = preview_df_dtypes[column_name]
                            try:
                                expected_dtype=None
                                if '(' in col_dtype:
                                    col_dtype=col_dtype.split('(')[0]
                                    if col_dtype.lower() in SQL_STRING_TYPES:
                                        expected_dtype = "string"
                                        valid_values_list = [str(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_INT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [int(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_FLOAT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [float(i) for i in params_list]
                                    elif col_dtype.lower()  in SQL_DATE_TYPES:
                                        expected_dtype = "date"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIME_TYPES:
                                        expected_dtype = "time"
                                        valid_values_list = [datetime.strptime(i, '%H:%M:%S') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIMESTAMP_TYPES:
                                        expected_dtype = "datetime"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in params_list]
                                else:
                                    if col_dtype.lower() in SQL_STRING_TYPES:
                                        expected_dtype = "string"
                                        valid_values_list = [str(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_INT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [int(i) for i in params_list]
                                    elif col_dtype.lower() in SQL_FLOAT_TYPES:
                                        expected_dtype = "numeric"
                                        valid_values_list = [float(i) for i in params_list]
                                    elif col_dtype.lower()  in SQL_DATE_TYPES:
                                        expected_dtype = "date"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIME_TYPES:
                                        expected_dtype = "time"
                                        valid_values_list = [datetime.strptime(i, '%H:%M:%S') for i in params_list]
                                    elif col_dtype.lower()  in SQL_TIMESTAMP_TYPES:
                                        expected_dtype = "datetime"
                                        valid_values_list = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in params_list]
                            except Exception as E:
                                    error_flag=True
                                    logger.error(f"Please enter input params of type {col_dtype}, for column {column_name} specified in range check")
                                    st.error(f"**🚨 Please enter input params of type {col_dtype}, for column {column_name} specified in range check**")
                                    #st.stop()
                    except:
                        error_flag=True
                        logger.error(f"Please enter correct input parameters instead of {range_check_str_params}, for range check")
                        st.error(f"**🚨 Please enter correct input parameters for range check**")

                elif range_check and not range_check_str_params:
                    error_flag=True
                    logger.error("Please provide input parameters for Range check")
                    st.error('**🚨 Please provide input parameters for Range check**')
                elif range_check_str_params and len(range_check)==0:
                    error_flag=True
                    logger.error("Please select columns for Range check")
                    st.error('**🚨 Please select columns for Range check**')
                else:
                    pass
                    
                #### perform input validations for regexp check
                if regexp_check_str_params and regexp_check:
                    # checking if input is a proper regular expression
                    if sum([bool(re.compile(regexp_str)) for regexp_str in regexp_check_str_params.split(',')]) != len(regexp_check_str_params.split(',')):
                        error_flag=True
                        logger.error(f"Please provide a valid regular expression instead of {regexp_check_str_params}")
                        st.error("**🚨 Please provide a valid regular expression**")
                    input_params_list = regexp_check_str_params.split(',')
                    # validating column names with input params length
                    if len(regexp_check) != len(input_params_list):
                        error_flag=True
                        logger.error("Number of columns and input parameters do not match for Regexp check")
                        st.error('**🚨 Number of columns and input parameters do not match for Regexp check**')
                elif regexp_check and not regexp_check_str_params:
                    error_flag=True
                    logger.error("Please provide input parameters for Regexp check")
                    st.error('**🚨 Please provide input parameters for Regexp check**')
                elif regexp_check_str_params and len(regexp_check)==0:
                    error_flag=True
                    logger.error("Please select columns for Regexp checks")
                    st.error('**🚨 Please select columns for Regexp checks**')
                else:
                    pass

                ### perform input validations for custom check
                if custom_check_str_params and custom_check:
                    input_params_list = custom_check_str_params.split(',')
                    # Validating column names with input params length
                    if len(custom_check) != len(input_params_list):
                        error_flag=True
                        logger.error("Number of columns and input parameters do not match for Custom check")
                        st.error('**🚨 Number of columns and input parameters do not match for Custom check**')
                    # Validating column and sql expression order
                    for col_name, sql_expr in dict(zip(custom_check,input_params_list)).items():
                        if col_name.lower() not in sql_expr.lower():
                            error_flag=True
                            logger.error(f"Invalid expression : {sql_expr} specified for column : {col_name}")
                            st.error(f'**🚨 Invalid expression : {sql_expr} specified for column : {col_name}**')
                        # Here we validate the sql expression. Column names are case sensitive
                        else:
                            try:
                                session.sql(f"""select "{col_name}" from "{selected_database}"."{selected_schema}"."{selected_table}" where {sql_expr} """).collect()
                                #session.sql(f"""select "{col_name}" from "{selected_database}"."{selected_schema}"."{selected_table}" """).filter(expr(sql_expr)).show()
                            except Exception as E:
                                error_flag=True
                                logger.error(f"Invalid Expression : {sql_expr}, specified for column : {col_name}. Column names in custom expression are case sensitive, Hence column names in custom expression should be enclosed by double quotes")
                                st.error(f'**🚨 Invalid Expression : {sql_expr}, specified for column : {col_name}. Column names in custom expression are case sensitive, Hence column names in custom expression should be enclosed by double quotes**')
                        
                elif custom_check and not custom_check_str_params:
                    error_flag=True
                    logger.error("Please provide input parameters for Custom check")
                    st.error('**🚨 Please provide input parameters for Custom check**')
                elif custom_check_str_params and len(custom_check)==0:
                    error_flag=True
                    logger.error("Please select columns for Custom checks")
                    st.error('**🚨 Please select columns for Custom checks**')
                else:
                    pass

                # data quality summary
                dq_checks = {
                                "null_check":null_check, 
                                "unique_check":unique_check,
                                "json_check":json_check,
                                "xml_check":xml_check,
                                "length_check":length_check,
                                "length_check_str_params":length_check_str_params,
                                "valid_values_check":valid_values_check,
                                "valid_values_check_str_params":valid_values_check_str_params,
                                "range_check": range_check,
                                "range_check_str_params": range_check_str_params,
                                "regexp_check":regexp_check,
                                "regexp_check_str_params":regexp_check_str_params,
                                "custom_check":custom_check,
                                "custom_check_str_params":custom_check_str_params
                            }
                if not error_flag:
                    try:
                        df = construct_test_df(dq_checks, selected_database, selected_schema, selected_table)
                    except Exception as E:
                        logger.error(f"Unable to summarize DQ checks due to exception : {E}")
                        st.error(f"🚨 Unable to summarize DQ checks")
                        
            
            if st.session_state.submitted and not error_flag and 'checks_df' in st.session_state and st.session_state.checks_df is not None and not st.session_state['checks_df'].empty:
                st.write(f"**➥ Consolidated data quality checks**")
                st.dataframe(st.session_state['checks_df'])
                session.write_pandas(st.session_state['checks_df'],table_name='DATA_QUALITY_CHECKS',schema="APP_INSTANCE_SCHEMA",auto_create_table=True, overwrite=False, use_logical_type=True)
                logger.info(f"Successfully registered DQ Check with id : {st.session_state['checks_df']['CHECK_ID'].unique().tolist()}")
                st.session_state['checks_df'] = None
                
            
            st.session_state.submitted = False
            # st.session_state.null_check = None
            # st.session_state.unique_check = None
            # st.session_state.length_check = None
            # st.session_state.length_check_str_params = None
            # st.session_state.valid_values_check = None
            # st.session_state.valid_values_check_str_params = None
            # st.session_state.range_check = None
            # st.session_state.range_check_str_params = None
    
    ###################
    ### Execute tab ###
    ###################    
    with tabs[4]:
        st.write("**➥ The execute page displays all the available data quality checks and the users can execute multiple checks by using the execution form below**")
        
        data_quality_tests = session.sql('select * from app_instance_schema.data_quality_checks').to_pandas()
        if len(database_list) > 0:
            with st.expander('The following are the available data quality tests', expanded = True):
                st.dataframe(data_quality_tests)

            ######################
            ### Execution Form ###
            ######################
            show_failed_records = False
            selected_check_id = ()
            with st.form('Select a check ID and execute'):
                st.write("Select Check ID's to execute")
                available_check_ids = data_quality_tests['CHECK_ID'].unique().tolist()
                selected_check_id = st.multiselect('Select check IDs', label_visibility='collapsed', options=available_check_ids)
                st.write(f"available_check_ids: {available_check_ids}")
                st.write(f"selected_check_id: {selected_check_id}")
                form_2_submitted = st.form_submit_button(label="**:green[Execute]**")
                
            if form_2_submitted:
                
                st.write('---')
                # check if user has selected a checkid
                if len(selected_check_id) == 0:
                    st.warning(f"**⚠️ Please select a checkID to execute**")
                else:
                    progress_bar = st.progress(0, text="**Fetching the details...**")
                    time.sleep(1)
                    st.write("FORM SUBMITTED and inside Else")
                    # Check if the tables corresponding to the selected checkid's has data
                    progress_bar.progress(25, text="**Scanning for empty tables...**")
                    st.write("REACHED 25%.")
                    time.sleep(1)
                    st.write("data_quality_tests")
                    st.dataframe(data_quality_tests)
                    tables_for_selected_check_id = data_quality_tests[data_quality_tests['CHECK_ID'].isin(selected_check_id)][['CHECK_ID','DATABASE_NAME','SCHEMA_NAME','TABLE_NAME']]
                    st.write("tables_for_selected_check_id")
                    st.dataframe(tables_for_selected_check_id)
                    tables_row_count = database_names[database_names['TABLE_NAME'].isin(tables_for_selected_check_id['TABLE_NAME'].unique().tolist())][['TABLE_CATALOG','TABLE_SCHEMA','TABLE_NAME','rows']].rename(columns={'TABLE_CATALOG':'DATABASE_NAME','TABLE_SCHEMA':'SCHEMA_NAME'},inplace=False)
                    st.write("database_names")
                    st.dataframe(database_names)
                    st.write("tables_row_count")
                    st.dataframe(tables_row_count)
                    result = pd.merge(tables_for_selected_check_id, tables_row_count, on =['DATABASE_NAME','SCHEMA_NAME','TABLE_NAME'], how='inner')
                    st.write("result")
                    st.dataframe(result)
                    check_ids_with_empty_tables = result[result['rows']==0]['CHECK_ID'].unique().tolist()
                    st.write(f"check_ids_with_empty_tables: {check_ids_with_empty_tables}")
                    check_ids_with_non_empty_tables = result[result['rows']>0]['CHECK_ID'].unique().tolist()
                    st.write(f"check_ids_with_non_empty_tables: {check_ids_with_non_empty_tables}")

                    if len(check_ids_with_empty_tables) == 1:
                        st.write("INSIDE IF STATEMENT")
                        st.write('###')
                        st.warning(f"**⚠️ The table :green[{result[result['CHECK_ID'].isin(check_ids_with_empty_tables)]['TABLE_NAME'].unique().tolist()[0]}] associated with check ID :green[{check_ids_with_empty_tables[0]}] do not contain any data**")
                    elif len(check_ids_with_empty_tables) > 1:
                        st.write("INSIDE ELSE STATEMENT")
                        tbl_result = ', '.join(map(str, result[result['CHECK_ID'].isin(check_ids_with_empty_tables)]['TABLE_NAME'].unique().tolist()))
                        id_result =  ', '.join(map(str, check_ids_with_empty_tables))
                        st.write('###')
                        st.warning(f"**⚠️ The tables :green[{tbl_result}] associated with check ID's :green[{id_result}] do not contain any data**")
                    
                    if len(check_ids_with_non_empty_tables) > 0:
                        show_failed_records = True
                        try:
                            progress_bar.progress(50, text="**Executing the data quality checks...**")
                            session.sql(f'call DATA_QUALITY_CHECK({check_ids_with_non_empty_tables})').collect()
                            
                            # Displaying the results
                            if show_failed_records:
                                if len(check_ids_with_non_empty_tables) == 1:
                                    st.info(f"**Below is the test summary for check ID :green[{check_ids_with_non_empty_tables[0]}]**")
                                else:
                                    result = ', '.join(map(str,check_ids_with_non_empty_tables))
                                    st.info(f"**Below is the test summary for check ID's :green[{result}]**")
                                
                                progress_bar.progress(75, text="**Generating test summary report...**")
                                st.write('##')
                                generate_test_summary(session, check_ids_with_non_empty_tables)
                                st.write('###')
                                st.write("**➥ Due to current data size limitations with streamlit in snowflake, only first 10000 failed records are displayed. To view full list of failed records, please execute the below query using the role that was used to create the application**")
                                st.code("""
                                -- Replace <ROLE NAME> with the role used to create the application
                                -- Replace <APPLICATION NAME> with the name of the application
                                GRANT APPLICATION ROLE DATA_QUALITY_CONTROL_APP_ROLE TO ROLE <ROLE NAME>;
                                SELECT * FROM "<APPLICATION NAME>"."APP_INSTANCE_SCHEMA"."DATA_QUALITY_FAILED_RECORDS";
                                """, language='sql')
                                progress_bar.progress(100, text="**Execution complete**")
                                show_failed_records = False
                                
                        except Exception as E:
                            logger.error(f"Unable to execute DQ check due to exception : {E}")
                            st.error(f"**🚨 Unable to execute DQ check**")
                        
                        
    ########################
    ### Past results tab ###
    ########################                 
    with tabs[5]:
        st.write("**➥ The view past results tab allows users to view past execution summary for selected data quality check IDs without re-executing the checks**")
        available_check_ids = data_quality_tests['CHECK_ID'].unique().tolist()
        with st.form("Select Check ID's to view past test results"):
            selected_ids = st.multiselect("Select Check ID's to view past test results", options=available_check_ids)
            generate = st.form_submit_button(label="**:green[View Results]**")
        
        with st.spinner('Generating Test Report'):
            if generate:
                if len(selected_ids) > 0:
                    st.write('---')
                    if len(selected_ids) == 1:
                        pd_df = session.sql(f"select distinct check_id, case when Result is Null then False else True end as Past_execution from data_quality_checks where check_id = {selected_ids[0]}").to_pandas()
                        executed_ids = pd_df[pd_df['PAST_EXECUTION']==True]['CHECK_ID'].tolist()
                        not_executed_ids = pd_df[pd_df['PAST_EXECUTION']==False]['CHECK_ID'].tolist()
                    else:
                        pd_df = session.sql(f"select distinct check_id, case when Result is Null then False else True end as Past_execution from data_quality_checks where check_id in {tuple(selected_ids)}").to_pandas()
                        executed_ids = pd_df[pd_df['PAST_EXECUTION']==True]['CHECK_ID'].tolist()
                        not_executed_ids = pd_df[pd_df['PAST_EXECUTION']==False]['CHECK_ID'].tolist()
            
                    if len(not_executed_ids) == 1:
                        st.warning(f'**⚠️ The CheckID :green[{not_executed_ids[0]}] does not have any past execution history**')
                    elif len(not_executed_ids) > 1:
                        result = ', '.join(map(str, not_executed_ids))
                        st.warning(f"**⚠️ The CheckID's :green[{result}] does not have any past execution history**")
            
                    if len(executed_ids) > 0:
                        result = ', '.join(map(str, executed_ids))
                        st.info(f"**Generating test report for :green[{result}]**")
                        generate_test_summary(session, executed_ids)
                        st.write('###')
                        st.write("**➥ Due to current data size limitations with streamlit in snowflake, only first 10000 failed records are displayed. To view full list of failed records, please execute the below query using the role that was used to create the application**")
                        st.code("""
                        -- Replace <ROLE NAME> with the role used to create the application
                        -- Replace <APPLICATION NAME> with the name of the application
                        GRANT APPLICATION ROLE DATA_QUALITY_CONTROL_APP_ROLE TO ROLE <ROLE NAME>;
                        SELECT * FROM "<APPLICATION NAME>"."APP_INSTANCE_SCHEMA"."DATA_QUALITY_FAILED_RECORDS";
                        """, language='sql')
                else:
                    st.warning(f"**⚠️ Please select a checkID to view past results**")
            
    
    ####################
    ### Schedule tab ###
    ####################
    with tabs[6]:
        st.write('**➥ The Scheduler page is used to schedule data quality checks based on cron expression entered using the below form**')

        # Get a list of available warhoueses to the current role
        #available_warehouses = pd.DataFrame(session.sql('show warehouses').collect())['name'].tolist()
        #available_warehouses = "reference('consumer_warehouse')"
        #available_warehouses = perms.get_reference_associations("consumer_warehouse")
        #available_warehouses = available_warehouses if isinstance(available_warehouses, list) else [available_warehouses]
        
        # Get a list of available check id's
        data_quality_tests = session.sql('select * from data_quality_checks').to_pandas()
        #data_quality_tests.columns = [col.upper() for col in data_quality_tests.columns.tolist()]
        available_check_ids = data_quality_tests['CHECK_ID'].unique().tolist()
        
        # Getting the maximum task id
        max_task_id = session.sql('select max(task_id) as tid from app_instance_task_schema.DATA_QUALITY_TASKS').collect()
        if max_task_id[0]['TID']:
            max_task_id = max_task_id[0]['TID'] + 1
        else:
            max_task_id = 1

        current_ts = session.sql("select current_timestamp()").collect()[0][0]
            
        # Task creation
        def create_cron_expression(minute, hour, day_of_month, month, day_of_week):
            # Trap for scenarios like Feb 30,31 etc
            d = {'2':['30','31'],
               '4':['31'],
               '6':['31'],
               '9':['31'],
               '11':['31']
               }
            if month in d and day_of_month in d[month]:
                st.error(f"**🚨 The month number : {month} does not contain the day : {day_of_month}**")
                sys.exit(0)

            # """Creates a cron expression from the given inputs."""
            cron_expression = ''
            cron_expression += '* ' if minute == '*' else f'{minute} '
            cron_expression += '* ' if hour == '*' else f'{hour} '
            cron_expression += '* ' if day_of_month == '*' else f'{day_of_month} '
            cron_expression += '* ' if month == '*' else f'{month} '
            cron_expression += '* ' if day_of_week == '*' else f'{day_of_week} '
        
            return cron_expression
        
        with st.form('Task creation form'):
            minute = st.text_input('Minute', value='*')
            hour = st.text_input('Hour', value='*')
            day_of_month = st.text_input('Day of Month', value='*')
            month = st.text_input('Month', value='*')
            day_of_week = st.text_input('Day of Week', value='*')
            selected_check_id_for_scheduling = st.multiselect('Select checkid',options=available_check_ids)
            st.session_state.task_form_submitted = st.form_submit_button(label="**:green[Create task]**")

        if st.session_state.task_form_submitted:
            # Validate check_id
            if len(selected_check_id_for_scheduling)==0:
                st.warning('**⚠️ Please select a checkid to schedule**')
                sys.exit(0)
                
            # creating the cron expression
            cron_expression = create_cron_expression(minute, hour, day_of_month, month, day_of_week)
            # Validating the cron expression
            if croniter.is_valid(cron_expression):
                try:
                    #st.write(f"Valid cron expression {cron_expression}")
                    st.info(f"**⏳ The above cron expression translates to : :green[{get_description(cron_expression)}]**")
                    task_query = f"""CREATE TASK app_instance_task_schema.DQ_TASK_{max_task_id} SCHEDULE = "USING CRON {cron_expression} America/Los_Angeles" warehouse = reference('consumer_warehouse') AS CALL app_instance_schema.DATA_QUALITY_CHECK ({selected_check_id_for_scheduling})"""
                    #st.write(f"Creating task {task_query}")
                    session.sql(task_query).collect()
                    #session.sql(f"""GRANT OWNERSHIP ON TASK app_instance_task_schema.DQ_TASK_{max_task_id} TO APPLICATION ROLE data_quality_control_app_role""").collect()
                    session.sql(f"""GRANT OPERATE ON TASK app_instance_task_schema.DQ_TASK_{max_task_id} TO APPLICATION ROLE data_quality_control_app_role""").collect()
                    #session.sql(f"""GRANT EXECUTE TASK ON ACCOUNT TO APPLICATION ROLE data_quality_control_app_role""").collect()
                    
                    # Find the index of the substring "warehouse = reference('consumer_warehouse')"
                    start_index = task_query.find("warehouse = reference('consumer_warehouse')")
                    end_index = start_index + len("warehouse = reference('consumer_warehouse')")
                    # Remove the substring from the input string
                    output_string = task_query[:start_index] + task_query[end_index:]
                    
                    session.sql(f"""insert into app_instance_task_schema.DATA_QUALITY_TASKS(TASK_ID, TASK_NAME, CREATED_ON, DEFINITION, STATUS) values ({max_task_id},'DQ_TASK_{max_task_id}', '{current_ts}'::timestamp, '{output_string}', 'suspended')""").collect()
                    st.success(f"**✅ Task DQ_TASK_{max_task_id} has been successfully created**")
                except Exception as E:
                    st.write(E)
                    logger.error(f"Unable to create snowflake task for cron : {cron_expression} due to exception : {E}")
                    st.error("**🚨 Unable to create a task with the specified inputs, Please make sure to provide a valid inputs for cron expression**")
                    sys.exit(0)
            else:
                logger.error(f"The input parameters did not yield a valid cron expression for {cron_expression}. Please correct the inputs")
                st.error("**🚨 The input parameters did not yield a valid cron expression. Please correct the inputs**")
                sys.exit(0)
            st.session_state.task_form_submitted= False
        
        st.write('---')
        
        # Task status
        task_df = pd.DataFrame(session.table('"APP_INSTANCE_TASK_SCHEMA"."DATA_QUALITY_TASKS"').collect())
        st.session_state.task_df = task_df
        if not st.session_state.task_df.empty:
            # Displaying the existing tasks
            with st.expander('**➥ Task Configuration form**', expanded=True):
                with st.form('Task Config Form', clear_on_submit=True):
                    c1,c2 = st.columns(2)
                    with c1:
                        task_names = st.multiselect("Select task name", options=task_df['TASK_NAME'].unique().tolist())
                    with c2:
                        action = st.selectbox('Select an action',options=['Suspend', 'Resume', 'Drop'])
                    st.session_state.task_config_form_submitted = st.form_submit_button(label="**:green[Submit]**")
                
                if st.session_state.task_config_form_submitted:
                    action = list([action])
                    
                    # validating inputs
                    if len(task_names)>0 and len(action)>0:
                        action = action * len(task_names)
                    elif len(task_names)>0 and len(action)==0:
                        st.error('**🚨 Please select an action**')
                    elif len(action)>0 and len(task_names)==0:
                        st.error('**🚨 Pls select a task name**')

                    task_dict = dict(zip(task_names, action))
                    # executing task actions
                    try:
                        for k,v in task_dict.items():
                            if v.lower() == 'suspend':
                                session.sql(f'alter task APP_INSTANCE_TASK_SCHEMA.{k} {v}').collect()
                                session.sql(f"""update APP_INSTANCE_TASK_SCHEMA.DATA_QUALITY_TASKS set status = 'suspended' where task_name = '{k}' """).collect()
                                st.success(f'**✅ Task :blue[{k}] suspended successfully!**')
                            elif v.lower() == 'resume':
                                session.sql(f'alter task APP_INSTANCE_TASK_SCHEMA.{k} {v}').collect()
                                session.sql(f"""update APP_INSTANCE_TASK_SCHEMA.DATA_QUALITY_TASKS set status = 'resumed' where task_name = '{k}' """).collect()
                                st.success(f'**✅ Task {k} resumed successfully!**')
                            elif v.lower() == 'drop':
                                session.sql(f'drop task APP_INSTANCE_TASK_SCHEMA.{k}').collect()
                                session.sql(f"""delete from APP_INSTANCE_TASK_SCHEMA.DATA_QUALITY_TASKS where task_name = '{k}' """).collect()
                                st.success(f'**✅ Task :red[{k}] dropped successfully!**')
                    except:
                        logger.error("Unable to resume task due to internal error")
                        st.error('**🚨 Internal Error, Please re-open the application and try again**')
                        sys.exit(0)
                    st.session_state.task_df = pd.DataFrame(session.table('"APP_INSTANCE_TASK_SCHEMA"."DATA_QUALITY_TASKS"').collect())
                    st.session_state.task_config_form_submitted = False
                    
                # display the tasks
                if not st.session_state.task_df.empty:
                    st.write('**➥ The following are the existing tasks**')
                    st.dataframe(st.session_state.task_df.set_index(st.session_state.task_df.columns[0]), use_container_width=True)
        else:
            st.write("**➥ There are no existing tasks scheduled**")
    
    #############################
    ### Privacy and Debugging ###
    ############################# 
    with tabs[7]:      
        st.write(f"""**➥ We are working diligently to enhance the debugging capabilities of our application to provide you with an even better user experience. Our sole purpose is to improve the performance and reliability of our application. As part of this effort, we kindly request your permission to access the `Event Table` within your Snowflake account. The Event Table is used for logging purposes, capturing information related to the application's errors, and user interactions. The request for access to the Event Table is strictly for debugging purposes, and no personal or sensitive data will be shared with any third-party, including the application provider (Kipi.bi). This request is applicable only for customers on `AWS US WEST` Region.**""")
        st.write('###')
        st.info('**:wave: To grant access to the event table, please follow the provided instructions**')
        st.warning('**⚠️ To use an existing event table, Please execute the below statements using accountadmin role in a worksheet**')
        st.code('''
        -- Set Event Table at account level and share application logs with provider
        -- <FULLY QUALIFIED NAME OF EVENT TABLE> refers to database_name.schema_name.event_table_name
        ALTER ACCOUNT SET EVENT_TABLE = <FULLY QUALIFIED NAME OF EVENT TABLE>;
        ALTER APPLICATION "<application name>" SET SHARE_EVENTS_WITH_PROVIDER=TRUE;
        ''', language='sql')
        l,m,r = st.columns([1,0.3,1])
        with m:
            st.subheader("**or**")
        st.warning('**⚠️ To create a new event table, Please execute the below statements using accountadmin role in a worksheet**')

        st.code('''
        -- Create Event Database
        CREATE DATABASE EVENT_DB;
        USE DATABASE EVENT_DB;
        USE SCHEMA Public;

        -- Create Event Table
        CREATE EVENT TABLE event_table;

        -- Set Event Table at account level and share application logs with provider
        ALTER ACCOUNT SET EVENT_TABLE = EVENT_DB.PUBLIC.EVENT_TABLE;
        ALTER APPLICATION "<application name>" SET SHARE_EVENTS_WITH_PROVIDER=TRUE;
        ''', language='sql')

privileges = [
        "IMPORTED PRIVILEGES ON SNOWFLAKE DB", "EXECUTE MANAGED TASK","EXECUTE TASK"
    ]
        
try:
    missing_privileges = perms.get_missing_account_privileges(privileges)
    if len(missing_privileges) > 0:
        perms.request_account_privileges(missing_privileges)
    else:
        # getting current session
        session = get_active_session()
        
        wh_flag= True
        table_flag = True
        
        # Check if the app has usage privs on a warehouse
        if not perms.get_reference_associations("consumer_warehouse"):
            perms.request_reference("consumer_warehouse")
            wh_flag=False
        #Check If the app has access to database tables
        table_access_check = pd.DataFrame(session.sql('show tables history in account').collect())
        # Trap for dropped tables
        table_access_check= table_access_check[pd.isna(table_access_check['dropped_on'])]
        table_access_check = table_access_check[~table_access_check['name'].isin(['DATA_QUALITY_CHECKS','DATA_QUALITY_FAILED_RECORDS','DATA_QUALITY_TASKS'])]
        if table_access_check.empty:
            table_flag=False
            
        if (not wh_flag and not table_flag):
            st.warning("**⚠️ Access Control Requirements**")
            st.write('**➥ The application needs usage privilege on a warehouse to schedule data quality checks in the scheduler page. Pls grant usage privilege using the below command and refresh the page to proceed.**')
            st.code("""
            -- Replace <APPLICATION_NAME> with name of the application, <warehouse_name> with the appropriate warehouse
            Grant usage on warehouse <warehouse_name> to application <application_name>;
            """, language='sql')
            st.write('---')
            st.write("**➥ The application needs read-only access to tables in order to create and execute data quality checks, pls grant select access on the tables to the application using the below mentioned commands and refresh the page to proceed.**")
            st.code("""
            -- Replace <APPLICATION_NAME> with name of the application
            -- Replace <DATABASE_NAME> with appropriate database name
            -- Replace <SCHEMA_NAME> with appropriate schem name
            GRANT USAGE ON DATABASE <DATABASE_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT USAGE ON SCHEMA <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT SELECT ON ALL TABLES IN SCHEMA <DATABASE_NAME>.<SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            """, language='sql')
        elif not table_flag and wh_flag:
            st.warning("**⚠️ Access Control Requirements**")
            st.write("**➥ The application needs read-only access to tables in order to create and execute data quality checks, pls grant select access on the tables to the application using the below mentioned commands and refresh the page to proceed.**")
            st.code("""
            -- Replace <APPLICATION_NAME> with name of the application
            -- Replace <DATABASE_NAME> with appropriate database name
            -- Replace <SCHEMA_NAME> with appropriate schem name
            GRANT USAGE ON DATABASE <DATABASE_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT USAGE ON SCHEMA <SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            GRANT SELECT ON ALL TABLES IN SCHEMA <DATABASE_NAME>.<SCHEMA_NAME> TO APPLICATION <APPLICATION_NAME>;
            """, language='sql')
        elif not wh_flag and table_flag:
            st.warning("**⚠️ Access Control Requirements**")
            st.write('**➥ The application needs usage privilege on a warehouse to schedule data quality checks in the scheduler page. Pls grant usage privilege using the below command and refresh the page to proceed.**')
            st.code("""
            -- Replace <APPLICATION_NAME> with name of the application, <warehouse_name> with the appropriate warehouse
            Grant usage on warehouse <warehouse_name> to application <application_name>;
            """, language='sql')
        else:   
            dq_main(session)
            
except IndexError:
    st.error('**🚨 Some error occured, Please refresh the page**')

    

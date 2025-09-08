CREATE OR REPLACE PROCEDURE KIPI.ADMIN.INFERENCING_SP_2("TGT_DB_NAME" VARCHAR(16777216), "TGT_SCHEMA_NAME" VARCHAR(16777216), "TGT_TABLE_NAME" VARCHAR(16777216))
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('joblib==1.2.0','pycaret==3.0.2','pycaret-models==3.0.2','snowflake-snowpark-python==*')
HANDLER = 'main'
IMPORTS = ('@KIPI.ADMIN.MODEL_STAGE/Final_Model_WITHOUT_CAT_FEAT.pkl')
EXECUTE AS OWNER
AS '# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
#from snowflake.snowpark.types import StructType, StructField, DoubleType, StringType
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import * 
import snowflake.snowpark.types as T
from snowflake.snowpark.types import * 
from pycaret.classification import *  
import pandas as pd 

# misc
import json


def main(session: snowpark.Session, tgt_db_name: str, tgt_schema_name: str, tgt_table_name: str):
    # Your code goes here, inside the "main" handler.
    pass  # Replace with actual code


    #################### MODEL PREDICTION CODE STARTS HERE ###################
    try:
        ############ LOADING THE SECOND CHUNK OF 35M DATASET##################
        test_table = f"{tgt_db_name}.{tgt_schema_name}.TEST_35M_PART2"
        df= session.table(test_table).toPandas()
        
        #################LOADING THE MODEL ############################
        session.file.get(''@"KIPI"."ADMIN"."MODEL_STAGE"/Final_Model_WITHOUT_CAT_FEAT'',''/tmp'')
        top_models = load_model(''/tmp/Final_Model_WITHOUT_CAT_FEAT'')
        print("Model loaded successfully.")
        
        # # ######################## PREDICTIONS ON TEST DATA ##############################
        pred2 = predict_model(top_models,data = df,raw_score=True)
        pred2= pred2.rename(columns={''prediction_score_1'': ''PREDICTION_SCORE'',''prediction_label'':''PREDICTION_LABEL''}).drop(''prediction_score_0'',axis=1)
        sf_df = session.create_dataframe(data=pred2)
        sf_df.write.mode(''append'').save_as_table(f"{tgt_db_name}.{tgt_schema_name}.{tgt_table_name}")

    
        return "Success" 
    except Exception as e:
        return f"Error: {str(e)}" '
    ;

call PREDICTION_MODEL_SP('KIPI','ADMIN','PRED_FINAL');
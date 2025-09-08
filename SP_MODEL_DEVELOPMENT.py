CREATE OR REPLACE PROCEDURE KIPI.ADMIN.RETRAIN_MODEL_SP()
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('joblib==1.2.0','pycaret==3.0.2','pycaret-models==3.0.2','snowflake-snowpark-python==*')
HANDLER = 'main'
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


def main(session: snowpark.Session):
    # Your code goes here, inside the "main" handler.
    pass  # Replace with actual code


    #################### MODEL DEVELOPMENT CODE STARTS HERE ###################
    ## Table reading
    tgt_table_full_name_feature_engg = "KIPI.ADMIN..PYCARET___TRAIN___2_5M"

    ## Loading the table having 2.5M data for Model Development
    df = session.table(tgt_table_full_name_feature_engg).toPandas()
   
    #Dropping the un-necssary columns before Model development
    df = df.drop([''SCHOOL_SIZE_RATIO'',''IRU_COUNT_LAST_90_DAYS'',''IRU_COUNT_LAST_120_DAYS'',
                                   ''IRU_COUNT_LAST_365_DAYS'',''IRU_COUNT_LAST_730_DAYS'',''PHOTO_COUNT'',''LOGIN_COUNT_LAST_7_DAYS'',
                                  ''HINOTE_COUNT_LAST_90_DAYS'',''HINOTE_COUNT_LAST_120_DAYS'',''HINOTE_COUNT_LAST_365_DAYS'',''HINOTE_COUNT_LAST_730_DAYS'',
                            ''AGE_GROUP'',''LOGIN_COUNT_BUCKETS'',''PUBLISHER_OWNER_NAME'',''PUBLISHER_NAME'',''GB_MOMENTUM''
                           ,''SCHOOL_NAME'',''SCHOOL_CITY'',''SCHOOL_STATE'',''ACQUISITION_SOURCE'',''MEMBERSHIP_STATUS'',''EMAIL_DOMAIN'', ''GENDER'',''MEMBERSHIP_STATUS_HISTORY'', ''EMAIL_DOMAIN_GROUP''],  axis=1)

    #Set up method for splitting the data, imbalance handling, ignoring the features not used for model development
    data = setup(df,target=''CLICKED_OR_NOT'',                              
                categorical_features=[], 
                  fold=5, train_size=0.995 ,
            data_split_stratify=True, session_id=123,
         ignore_features=[''REGISTRATION_ID''],fix_imbalance = True)

    #Compare method which by default compare all the algorithms for the model development and gives you the top model along with evaluation metrics results on training data
    top_model = compare_models(sort='AUC',include = [''xgboost'',''lightgbm''])
    
    #Finalize_model method to finalize your model
    top_models = finalize_model(top_model)
    
     ################# MODEL SAVING CODE ###################
     # saving th model to MODEL_STAGE
    import_dir = sys._xoptions.get("snowflake_import_directory")
    save_model(top_models, os.path.join(import_dir, ''/tmp/Final_Model_WITHOUT_CAT_FEAT''))
    session.file.put(
        os.path.join(import_dir, ''/tmp/Final_Model_WITHOUT_CAT_FEAT.pkl''),
        "@MODEL_STAGE",
        auto_compress=False,
        overwrite=True
    )

    # ################PULLING TRAIN MODEL METRICS ###########################
    train_metrics = pull()
    train_metrics_sp = session.create_dataframe(train_metrics)
    train_metrics_sp.write.mode(''overwrite'').save_as_table("KIPI.ADMIN.TRAIN_METRICS")

    return "Success" '
    ;

call PROCEDURE KIPI.ADMIN.RETRAIN_MODEL_SP();


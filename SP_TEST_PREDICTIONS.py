CREATE OR REPLACE PROCEDURE KIPI.ADMIN.PREDICTIONS_CODE()
RETURNS TABLE ()
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
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

# data science libs
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# misc
import json
import warnings
pd.set_option(''display.max_columns'', 1200)
pd.set_option(''display.max_rows'', None)
pd.set_option(''display.max_colwidth'', None)

warnings.filterwarnings("ignore")


def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler.

    ## You can use the below code in case you want create a dataset  from MASTER_MARKETING_DATA2 table ##############
    tableName = ''KIPI.PUBLIC.MASTER_MARKETING_DATA2''
    click_t = ''KIPI.PUBLIC.CMMAILCLICKTHROUGH_1_EVT''
    
    df = session.sql(f''''''select * from {tableName} mm
    WHERE DATE(mm.current_timestamp) = CURRENT_DATE()-30 and bad_email=0'''''').drop([''S_NUMBER'',''USER_TYPES'',''CLASSIC_SKU_ID'',''SCHOOL_ID'', ''MOST_RECENT_EMAIL_SEGMENT_ID''])

    # Print a sample of the dataframe to standard output.
    print(len(df.columns),df.count())

    #### NOTE ----> ####### If yo have test table handy with you , you can directly take that table and drop the columns [''S_NUMBER'',''USER_TYPES'',''CLASSIC_SKU_ID'',''SCHOOL_ID'', ''MOST_RECENT_EMAIL_SEGMENT_ID''] same as above statetment################
    
    ############################### DATA CLEANING STEPS ##################
    
    miss = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().T
    miss = miss.rename(columns={0:''Missing_Values_%''})
    miss[''Missing_Values_%''] = (miss[''Missing_Values_%'']/df.count())*100
    miss = miss.sort_values(''Missing_Values_%'',ascending=False).reset_index().rename(columns={''index'':''Feature_Name''})
    drop_cols = miss[miss[''Missing_Values_%'']>90][''Feature_Name''].values.tolist()
    print("Drop columns list: ",drop_cols)
    for col_name in drop_cols:
        df = df.drop(col(col_name))
    print(len(df.columns))

    ####Data Cleaning
    dt = df.dtypes
    bool_cols = [item[0] for item in dt if item[1].startswith(''bool'')]
    string_cols = [item[0] for item in dt if item[1].startswith(''string'')]
    date_cols = [item[0] for item in dt if item[1].startswith(''date'')]
    num_cols = list(set(df.columns).difference(set(string_cols)).difference(set(date_cols)).difference(set(bool_cols)))
    
    print(len(num_cols), len(string_cols), len(date_cols),len(bool_cols), len(num_cols)+ len(string_cols)+ len(date_cols)+len(bool_cols))
    
    str_exc_list = [''REUNION_YEAR'',''DNER_FLAG'',''COMMERCIAL_IND'',''WEEKLY_DIGEST_IND'',
     ''MY_PROFILE_VISITS_IND'',
     ''MY_REMINDER_IND'',
     ''SCHOOL_COMMUNITY_IND'',
     ''MY_REMEMBERS_IND'',
     ''MY_INBOX_IND'',
     ''MY_PROFILE_NOTES_IND'',
     ''MY_PRIVATE_MESSAGES_IND'',
     ''SCHOOL_PROFILE_IND'',
     ''SCHOOL_REMINDER_IND'',
     ''SCHOOL_YEARBOOK_IND'',
     ''NEW_CLASSMATES_FEATURES_IND'',
     ''DO_NOT_EMAIL_IND'',''MOBILE_APP_USER'',''UPLOADED_PHOTOS_OR_NOT'',
     ''TAGGED_IN_YEARBOOK_OR_NOT'',''REUNION_ORGANIZER'',''REUNION_INVITEE'',''FACEBOOK_TOKEN_EXPIRED'',''FACEBOOK_TOKEN_AVAILABLE'']
    
    
    string_cols = [x for x in string_cols if x not in str_exc_list]
    bool_cols.extend(str_exc_list)
    print("string_cols: " ,string_cols)
    print("bool_cols: " ,bool_cols)
    # Return value will appear in the Results tab.

    ## Numerical Variables
    count_num = [s for s in df.select(num_cols).columns if ''COUNT'' in s]
    non_count_num = [x for x in df.select(num_cols).columns if x not in count_num]
    df = df.fillna(0, subset=count_num)
    
    zero_per_cols = miss[miss[''Missing_Values_%'']==0.000000][''Feature_Name''].values.tolist()
    test = [x for x in non_count_num if x not in zero_per_cols]
    print(test)
    
    a = df.select([count(when(col(c).isNull(), c)).alias(c) for c in test]).toPandas().T
    a = a.rename(columns={0:''Missing_Values_%''})
    a[''Missing_Values_%''] = (a[''Missing_Values_%'']/df.count())*100
    
    a.sort_values(''Missing_Values_%'',ascending=False)
    df = df.fillna(0, subset=''LIFETIME_IRU_TAGS'')

    ## String Columns Imputation
    df = df.fillna("Others", subset=string_cols)

    ## Remove the two features named REUNION_ORGANIZER and MOBILE_APP_USER as they have almost all same values
    df = df.drop([''REUNION_ORGANIZER'',''MOBILE_APP_USER''])

    ## Date cols
    df = df.drop([''FIRST_CLASSIC_UPGRADE_DATE'',''LAST_UPDATE_DATE''])
    df = df.withColumn("days_since_last_login",
      datediff(''day'',col("LAST_LOGIN_DATE"),current_date()-30)
                      )
    
    df = df.withColumn("days_bw_last_two_logins",
      datediff(''day'',col("PRIOR_LAST_LOGIN_DATE"), col("LAST_LOGIN_DATE"))
                      )
    
    df = df.drop(''LAST_LOGIN_DATE'',''PRIOR_LAST_LOGIN_DATE'')
    df = df.fillna(0, subset=[''DAYS_SINCE_LAST_LOGIN'',''DAYS_BW_LAST_TWO_LOGINS''])
    df = df.dropna()
    clean = df.drop(''CURRENT_TIMESTAMP'')
    print(len(clean.columns))

    click = session.sql(f''''''select ec.reg_id as registration_id_click
    FROM {click_t} ec
    WHERE DATE(ec.event_date) >= CURRENT_DATE()-30
    GROUP BY 1'''''')
    # click.printSchema()
    
    clean = clean.join(click, (clean.REGISTRATION_ID==click.REGISTRATION_ID_CLICK), ''left'')
    clean = clean.withColumn("clicked_or_not",when(clean.REGISTRATION_ID_CLICK.isNull(),''False'').otherwise(''True'')).drop(''REGISTRATION_ID_CLICK'')
    print(clean.groupBy(''clicked_or_not'').count().show())
    ############# SAVING THE TABLE WITH IMPUTED DATA ###################################
    clean.write.mode(''overwrite'').save_as_table(''KIPI.ADMIN.DATASET'')


    ########################### FEATURE ENGG - DERIVING NEW COLUMNS ###################
    
    clean = session.sql(''select * from KIPI.ADMIN.DATASET'')

    bool_cols_v2, string_cols_v2, date_cols_v2,num_cols_v2, count_num_v2, non_count_num_v2 = cols_list(clean)
    
    list_new = [x for x in num_cols_v2 if x not  in (''BAD_EMAIL'',''REGISTRATION_ID'',''CREATION_YEAR'',''COUNT_OF_EXPIRATIONS'',''DAYS_BW_LAST_TWO_LOGINS'',
                                                 ''IRU_COUNT_LAST_30_DAYS'',''SCHOOL_SIZE'',''MAX_GRADE_YEAR'',''HINOTE_COUNT_LAST_730_DAYS'',''HINOTE_COUNT_LAST_365_DAYS'',
                                                    ''HINOTE_COUNT_LAST_120_DAYS'',''HINOTE_COUNT_LAST_90_DAYS'',''HINOTE_COUNT_LAST_60_DAYS'',''HINOTE_COUNT_LAST_30_DAYS'',
                                                    ''LOGIN_COUNT_LAST_7_DAYS'',''LOGIN_COUNT_LAST_14_DAYS'',''IRU_COUNT_LAST_90_DAYS'',''IRU_COUNT_LAST_60_DAYS'')]
    print(len(clean.columns))

    ## New Feature Creation
    # 1.0 Lifetime Slot Count
    
    slot = session.sql(f''''''select reg_id as REGISTRATION_ID, count(SLOT_NAME) as "LIFETIME_SLOTS_COUNT" from 
    CMMAILCLICKTHROUGH_1_EVT group by reg_id'''''') 
    
    # 2. DAILY_CUSTOMER_TRANSACTIONS for last 90 Days
    
    all_renewals = session.sql(f''''''SELECT registration_id, SUM(dct.transactions) as renewals

    FROM (
    /*CLASSIC*/
    SELECT
    dos.registration_id
    , sku.term_length
    , DATE(dos.start_date) as start_date
    , ''renewals'' as transaction_type
    , count(distinct dos.daily_customer_trans_id) as transactions
    FROM
     (SELECT a.*
                         FROM daily_customer_transactions a 
                         WHERE 1=1                          
                         AND last_renewed_date IS NOT NULL  
                         AND transaction_type = ''PAYMENT''  
                         AND item_class_type = ''permissionCommerceItem''  
                         AND transaction_status = 1 
                         AND order_state <> ''REMOVED''  
                         AND ( cancelled_date IS NULL OR date(cancelled_date) = date(start_date)) 
           ) dos
           
   
    JOIN MASTER_SKU_LIST sku
    ON dos.sku_id = sku.sku_id
   
   
    WHERE 1=1
           AND DATE(dos.start_date) < CURRENT_DATE()
        AND dos.sku_id IN (
                SELECT
                        sku_id
                FROM
                        sku dpc,
                        PRODUCT dp
                WHERE
                        dpc.product_id = dp.product_id
                        AND dp.display_name IN(''Gold Membership'', ''Work Gold Membership'')
        )
        AND dos.sku_id in (
                select
                        sku_id
                from
                        sku_type
                where
                        type = 0
        )
    GROUP BY 1,2,3,4
    /*CLASSIC*/
    
    UNION ALL 
    
    /*MAGENTO*/
    SELECT 
     
    dct.cmates_registration_id as registration_id
    , sku.term_length
    , date(SUBSCRIPTION_LAST_RUN_DATE) as start_date
    , ''renewals'' as transaction_type
    , count(DISTINCT SUBSCRIPTION_ID) as transactions  
    FROM  MAGENTO_DAILY_CUSTOMER_TRANSACTIONSALL dct
    JOIN MASTER_SKU_LIST sku
    ON dct.reference_sku = sku.sku_id
    where
          run_count > 1
          and subscription_id not in (
                select subscription_id from PARADOXLABS_SUBSCRIPTION where date(LAST_RUN_DATE)= CURRENT_DATE()-1 and run_count>1
            and subscription_id   in (
                select subscription_id from PARADOXLABS_SUBSCRIPTIONLOG where date(created_at)= CURRENT_DATE()-1 and description like (''%retry%'')
                )
            )
            and date(SUBSCRIPTION_LAST_RUN_DATE) < current_date()

    GROUP BY 1,2,3,4
    /*MAGENTO*/
    )dct
    GROUP BY 1'''''')

    # 3. Daily Transactions Count
    trn_count = session.sql(f''''''select registration_id, count(daily_customer_trans_id) as "CUSTOMER_TRANSALL_COUNT" 
    from DAILY_CUSTOMER_TRANSACTIONSALL group by registration_id'''''')
    
    # 4. Generated Email Messages Counts
    gen_counts = session.sql(f'''''' select registration_id, count(generated_email_msg_id) as GENERATED_EMAIL_MSG_COUNT
    from GENERATED_EMAIL_MSGS
    where DATE(creation_date) >= current_date() - 60
    AND DATE(creation_date) <= current_date()-30
    group by registration_id '''''')
    # Print a sample of the dataframe to standard output.
    clean = clean.join(gen_counts, ''REGISTRATION_ID'', how=''left'')
    #clean = clean.fillna(0, subset=[''GENERATED_EMAIL_MSG_COUNT'',''EMAIL_SEND_COUNT_LAST_30_DAYS''])
    clean = clean.join(trn_count, ''REGISTRATION_ID'', how=''left'')
    #clean = clean.fillna(0, subset=[''CUSTOMER_TRANSALL_COUNT''])
    clean = clean.join(slot, ''REGISTRATION_ID'', how=''left'')
    #clean = clean.fillna(0, subset=[''LIFETIME_SLOTS_COUNT''])
    clean = clean.join(all_renewals, ''REGISTRATION_ID'', how=''left'')

    print("Lenth of clean table columns after adding new features",len(clean.columns))

    miss = clean.select([count(when(col(c).isNull(), c)).alias(c) for c in clean.columns]).toPandas().T
    miss = miss.rename(columns={0:''Missing_Values_%''})
    miss[''Missing_Values_%''] = (miss[''Missing_Values_%'']/clean.count())*100
    miss = miss.sort_values(''Missing_Values_%'',ascending=False).reset_index().rename(columns={''index'':''Feature_Name''})
    print(miss.head())
    
     
    drop_cols = miss[miss[''Missing_Values_%'']>90][''Feature_Name''].values.tolist()
    for cols in drop_cols:
        clean = clean.drop(cols)
    print("Lenth of clean table columns after dropping 90% missing features",len(clean.columns)) 

    #### SAVING THE TABLE AGAIN WITH NEW COLUMNS ###########
    clean.write.mode(''overwrite'').save_as_table(''KIPI.ADMIN.DATASET'')

    #################### MODEL PREDICTION CODE STARTS HERE ###################
    complete_df = session.sql(''select * from DATASET'').toPandas()
    df = session.sql(''select * from PYCARET___TRAIN___2_5M'').toPandas()
    df_test = complete_df.subtract(df) ## here we are removing the REG_ID on which model is trained or you can use the below code in want to predict on complete dataset
    #df_test = session.sql(''select * from KIPI.ADMIN.DATASET'').toPandas() 

    df = df.drop([''SCHOOL_SIZE_RATIO'',''BAD_EMAIL'',''IRU_COUNT_LAST_90_DAYS'',''IRU_COUNT_LAST_120_DAYS'',
                      ''IRU_COUNT_LAST_365_DAYS'',''IRU_COUNT_LAST_730_DAYS'',''PHOTO_COUNT'',''LOGIN_COUNT_LAST_7_DAYS'',
                      ''HINOTE_COUNT_LAST_90_DAYS'',''HINOTE_COUNT_LAST_120_DAYS'',''HINOTE_COUNT_LAST_365_DAYS'',''HINOTE_COUNT_LAST_730_DAYS'',
                  ''AGE_GROUP'',''LOGIN_COUNT_BUCKETS'', ''PUBLISHER_OWNER_NAME'',''PUBLISHER_NAME'',
                                      ''GB_MOMENTUM'',  ''CONVERTED_MEMBERSHIP_STATUS_HISTORY'', ''CONVERTED_EMAIL_DOMAIN_GROUP''
                  ,''SCHOOL_NAME'',''SCHOOL_CITY'',''SCHOOL_STATE'',''ACQUISITION_SOURCE'',''MEMBERSHIP_STATUS'',''EMAIL_DOMAIN'', ''GENDER''
                   ],axis=1)
    
    df_test = df_test.drop([''SCHOOL_SIZE_RATIO'',''BAD_EMAIL'',''IRU_COUNT_LAST_90_DAYS'',''IRU_COUNT_LAST_120_DAYS'',
                                   ''IRU_COUNT_LAST_365_DAYS'',''IRU_COUNT_LAST_730_DAYS'',''PHOTO_COUNT'',''LOGIN_COUNT_LAST_7_DAYS'',
                                  ''HINOTE_COUNT_LAST_90_DAYS'',''HINOTE_COUNT_LAST_120_DAYS'',''HINOTE_COUNT_LAST_365_DAYS'',''HINOTE_COUNT_LAST_730_DAYS'',
                            ''AGE_GROUP'',''LOGIN_COUNT_BUCKETS'',''PUBLISHER_OWNER_NAME'',''PUBLISHER_NAME'',''GB_MOMENTUM''
                           ,''SCHOOL_NAME'',''SCHOOL_CITY'',''SCHOOL_STATE'',''ACQUISITION_SOURCE'',''MEMBERSHIP_STATUS'',''EMAIL_DOMAIN'', ''GENDER''],   axis=1)

    data = setup(df,target=''CLICKED_OR_NOT'',                              
                categorical_features=[], 
                 fold=5,
           data_split_stratify=True, session_id=123,
         ignore_features=[''REGISTRATION_ID'',''EMAIL_CLICK_COUNT_LAST_90_DAYS'',''EMAIL_CLICK_COUNT_LAST_120_DAYS''],fix_imbalance = True)
    
    session.file.get(''@"KIPI"."ADMIN"."MODEL_STAGE"/Final_Model_WITHOUT_CAT_FEAT'',''/tmp'')
    top_models = load_model(''/tmp/Final_Model_WITHOUT_CAT_FEAT'')
    print("Model loaded successfully.")
  
    # ################TRAIN MODEL METRICS ###########################
    train_metrics = pull()
    train_metrics_sp = session.create_dataframe(train_metrics)
    train_metrics_sp.write.mode(''overwrite'').save_as_table(''KIPI.ADMIN.FINAL_TRAIN_METRICS'')

    # # ######################## PREDICTIONS ON TEST DATA ##############################
    pred1 = predict_model(top_models,data = df_test,raw_score=True)
    pred1= pred1.rename(columns={''"prediction_score_1"'': ''PREDICTION_SCORE'',''"prediction_label"'':''PREDICTION_LABEL''}).drop(''"prediction_score_0"'',axis=1)
    sf_df = session.create_dataframe(data=pred1)
    sf_df.write.mode(''overwrite'').save_as_table(''KIPI.ADMIN.FINAL_PRED_67M'')
   
    test_metrics = pull()
    test_metrics_sp = session.create_dataframe(test_metrics)
    test_metrics_sp.write.mode(''overwrite'').save_as_table(''KIPI.ADMIN.FINAL_TEST_METRICS'')

    return "Success"';
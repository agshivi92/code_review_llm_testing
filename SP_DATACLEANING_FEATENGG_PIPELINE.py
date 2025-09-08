--This script can be executed inside Snowflake SQL Worksheets
-------Author : Kipi DS Team (Kipi.bi)

CREATE OR REPLACE PROCEDURE KIPI.ADMIN.SP_DATACLEANING_FEATENGG_PIPELINE("SRC_DB_NAME" VARCHAR(16777216), "SRC_SCHEMA_NAME" VARCHAR(16777216), "SRC_TABLE_NAME" VARCHAR(16777216), "TGT_DB_NAME" VARCHAR(16777216), "TGT_SCHEMA_NAME" VARCHAR(16777216))
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
EXECUTE AS OWNER
AS '# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import * 
from snowflake.snowpark.types import * 

def cols_list(df):
    dt_v2 = df.dtypes
    bool_cols_v2 = [item[0] for item in dt_v2 if item[1].startswith(''bool'')]
    string_cols_v2 = [item[0] for item in dt_v2 if item[1].startswith(''string'')]
    date_cols_v2 = [item[0] for item in dt_v2 if item[1].startswith(''date'')]
    num_cols_v2 = list(set(df.columns).difference(set(string_cols_v2)).difference(set(date_cols_v2)).difference(set(bool_cols_v2)))
    count_num_v2 = [s for s in df.select(num_cols_v2).columns if ''COUNT'' in s]
    non_count_num_v2 = [x for x in df.select(num_cols_v2).columns if x not in count_num_v2]
    return bool_cols_v2, string_cols_v2, date_cols_v2,num_cols_v2, count_num_v2, non_count_num_v2

    ############################### MAIN FUNCTION ####################################

def main(session: snowpark.Session, src_db_name: str, src_schema_name: str, src_table_name: str, tgt_db_name: str, tgt_schema_name: str): 
    # SQL query to fetch all the data from the "Master Marketing table" for a "30 days back" timestamp
    df = session.sql(f''''''select mm.*, CASE WHEN (comp.reg_id IS NOT NULL) THEN TRUE ELSE FALSE END AS CLICKED_OR_NOT from {src_db_name}.{src_schema_name}.{src_table_name} mm
    left join
    (SELECT ec.reg_id FROM {src_db_name}.{src_schema_name}.CMMAILCLICKTHROUGH_1_EVT ec WHERE 1=1 AND DATE(ec.event_date) >= CURRENT_DATE() -30
    GROUP BY 1 ) comp
    on mm.registration_id = comp.reg_id
    WHERE DATE(mm.current_timestamp) = CURRENT_DATE()-30 and bad_email=0 
    '''''').drop([''S_NUMBER'',''USER_TYPES'',''CLASSIC_SKU_ID'',''SCHOOL_ID'', ''MOST_RECENT_EMAIL_SEGMENT_ID'',''REG_ID'',''BAD_EMAIL''])

    ########################### DERIVING NEW COLUMNS ###################

    # New Feature Creation
    # 1.0 Lifetime Slot Count
    
    slot = session.sql(f''''''select reg_id as REGISTRATION_ID, count(SLOT_NAME) as "LIFETIME_SLOTS_COUNT" from 
    CMMAILCLICKTHROUGH_1_EVT group by reg_id'''''') 
    
    # 2. DAILY_CUSTOMER_TRANSACTIONS for last 90 Days
    #SQL query to get the count of "RENEWALS" for both "CLASSIC" & "MAGENTO" registration ids
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
    # SQL query to get the count of the "TRANSACTIONS" done by all the registration ids 
    trn_count = session.sql(f''''''select registration_id, count(daily_customer_trans_id) as "CUSTOMER_TRANSALL_COUNT" 
    from DAILY_CUSTOMER_TRANSACTIONSALL group by registration_id'''''')
    
    # 4. Generated Email Messages Counts
    # SQL query to get the count of all the "GENERATED EMAIL MESSAGES" for the registration ids for the time period of last 30 days 
    gen_counts = session.sql(f'''''' select registration_id, count(generated_email_msg_id) as GENERATED_EMAIL_MSG_COUNT
    from GENERATED_EMAIL_MSGS
    where DATE(creation_date) >= current_date() - 60
    AND DATE(creation_date) <= current_date()-30
    group by registration_id '''''')
    
    # Joining the newly created features with the Master table - "LIFETIME_SLOTS_COUNT", "RENEWALS", "CUSTOMER_TRANSALL_COUNT", "GENERATED_EMAIL_MSG_COUNT"
    df = df.join(slot, ''REGISTRATION_ID'', how=''left'')
    df = df.join(all_renewals, ''REGISTRATION_ID'', how=''left'')
    df = df.join(trn_count, ''REGISTRATION_ID'', how=''left'')
    df = df.join(gen_counts, ''REGISTRATION_ID'', how=''left'')

    # Creating more new features - "YEARS_SINCE_CREATION" using the column "CREATION_DATE" and "YEARS_SINCE_MAXGRAD" using the column "MAX_GRAD_YEAR"  
    df = df.withColumn("YEARS_SINCE_CREATION", round(months_between(current_date(), col("CREATION_DATE")) / lit(12), 1))
    df = df.withColumn("YEARS_SINCE_MAXGRAD", year(current_date())-col("MAX_GRAD_YEAR"))
    
    # Dropping the unnecessary columns from the Master table
    df= df.drop([''CREATION_DATE'',''CREATION_YEAR'',''MAX_GRAD_YEAR'',''FIRST_CLASSIC_UPGRADE_DATE'',''LAST_UPDATE_DATE'',''CURRENT_TIMESTAMP''])

    # Dropping features which mostly has Null values and wont contribute much
    df = df.drop([''REUNION_ORGANIZER'',''MOBILE_APP_USER''])

    # Creating a new feature DAYS_SINCE LAST_LOGIN using the date column LAST_LOGIN_DATE
    df = df.withColumn("days_since_last_login",
      datediff(''day'',col("LAST_LOGIN_DATE"),current_date()-30)
                      )
    # Dropping the unnecessary columns from the Master table
    df = df.withColumn("days_bw_last_two_logins",
      datediff(''day'',col("PRIOR_LAST_LOGIN_DATE"), col("LAST_LOGIN_DATE")))

    # Replacing the "Null" values in the columns "DAYS_SINCE_LAST_LOGIN" and "DAYS_BW_LAST_TWO_LOGINS" with 0
    df = df.drop(''LAST_LOGIN_DATE'',''PRIOR_LAST_LOGIN_DATE'')
    clean = df.fillna(0, subset=[''DAYS_SINCE_LAST_LOGIN'',''DAYS_BW_LAST_TWO_LOGINS''])

    ############################### DATA CLEANING STEPS ##################

    ## Dropping columns with missing values greater than 90%
    miss = clean.select([count(when(col(c).isNull(), c)).alias(c) for c in clean.columns]).toPandas().T
    miss = miss.rename(columns={0:''Missing_Values_%''})
    miss[''Missing_Values_%''] = (miss[''Missing_Values_%'']/clean.count())*100
    miss = miss.sort_values(''Missing_Values_%'',ascending=False).reset_index().rename(columns={''index'':''Feature_Name''})
    drop_cols = miss[miss[''Missing_Values_%'']>90][''Feature_Name''].values.tolist()
    for col_name in drop_cols:
        clean = clean.drop(col(col_name))

    ## Data Imputation
    bool_cols, string_cols, date_cols,num_cols, count_num, non_count_num = cols_list(clean)
    
    # List of all the Boolean columns from the Master table (Columns with String values having two classes)
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
    
    # Creating a list of columns with "String" datatype excluding the "Boolean" columns
    string_cols = [x for x in string_cols if x not in str_exc_list]
    
    # Adding the columns with "Boolean String" values into the boolean columns list 
    bool_cols.extend(str_exc_list)

    # Replacing the "NULL" values in the Numerical columns with columns names having "COUNT" 
    clean = clean.fillna(0, subset=count_num)

    # check for missing values after removing NON_COUNT features and Numerical features with no-missing values
    zero_per_cols = miss[miss[''Missing_Values_%'']==0.000000][''Feature_Name''].values.tolist()
    rem_noncount_num_cols = [x for x in non_count_num if x not in zero_per_cols]
    miss_rem_noncount = miss[miss.Feature_Name.isin(rem_noncount_num_cols)]
    noncount_num_impute_cols = miss_rem_noncount.Feature_Name.values.tolist()
    noncount_num_impute_cols_v2 = [i for i in noncount_num_impute_cols if i not in (''REGISTRATION_ID'',''SCHOOL_SIZE_RATIO'',''YEARS_SINCE_CREATION'',''YEARS_SINCE_MAXGRAD'')]

    # Replacing the "NULL" values in the Numerical columns with columns names not having "COUNT"
    clean = clean.fillna(0, subset=noncount_num_impute_cols_v2)
    
    # Replacing the missing values in the string columns as "Others"
    clean = clean.fillna("Others", subset=string_cols)
    
    # Mapping all the string values to numerical values in the boolean columns 
    clean = clean.withColumn(''PRINTABLE_SENIOR_YEARBOOK_NOT_ON_SITE'', F.when(clean.PRINTABLE_SENIOR_YEARBOOK_NOT_ON_SITE == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''PRINTABLE_JUNIOR_YEARBOOK_ON_SITE'', F.when(clean.PRINTABLE_JUNIOR_YEARBOOK_ON_SITE == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''PRINTABLE_SOPHOMORE_YEARBOOK_ON_SITE'', F.when(clean.PRINTABLE_SOPHOMORE_YEARBOOK_ON_SITE ==''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''PRINTABLE_FRESHMAN_YEARBOOK_ON_SITE'', F.when(clean.PRINTABLE_FRESHMAN_YEARBOOK_ON_SITE ==''FALSE'', 0).otherwise(1))
    
    clean = clean.withColumn(''REUNION_YEAR'', F.when(clean.REUNION_YEAR == ''No'', 0).otherwise(1))
    clean = clean.withColumn(''DNER_FLAG'', F.when(clean.DNER_FLAG == ''No'', 0).otherwise(1))
    clean = clean.withColumn(''COMMERCIAL_IND'', F.when(clean.COMMERCIAL_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''WEEKLY_DIGEST_IND'', F.when(clean.WEEKLY_DIGEST_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_PROFILE_VISITS_IND'', F.when(clean.MY_PROFILE_VISITS_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_REMINDER_IND'', F.when(clean.MY_REMINDER_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''SCHOOL_COMMUNITY_IND'', F.when(clean.SCHOOL_COMMUNITY_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_REMEMBERS_IND'', F.when(clean.MY_REMEMBERS_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_INBOX_IND'', F.when(clean.MY_INBOX_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_PROFILE_NOTES_IND'', F.when(clean.MY_PROFILE_NOTES_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''MY_PRIVATE_MESSAGES_IND'', F.when(clean.MY_PRIVATE_MESSAGES_IND ==''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''SCHOOL_PROFILE_IND'', F.when(clean.SCHOOL_PROFILE_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''SCHOOL_REMINDER_IND'', F.when(clean.SCHOOL_REMINDER_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''SCHOOL_YEARBOOK_IND'', F.when(clean.SCHOOL_YEARBOOK_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''NEW_CLASSMATES_FEATURES_IND'', F.when(clean.NEW_CLASSMATES_FEATURES_IND == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''DO_NOT_EMAIL_IND'', F.when(clean.DO_NOT_EMAIL_IND ==''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''UPLOADED_PHOTOS_OR_NOT'', F.when(clean.UPLOADED_PHOTOS_OR_NOT == ''NO_UPLOADED_PHOTOS'', 0).otherwise(1))
    clean = clean.withColumn(''TAGGED_IN_YEARBOOK_OR_NOT'', F.when(clean.TAGGED_IN_YEARBOOK_OR_NOT == ''NOT_TAGGED_IN_YEARBOOK'', 0).otherwise(1))
    clean = clean.withColumn(''REUNION_INVITEE'', F.when(clean.REUNION_INVITEE == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''FACEBOOK_TOKEN_EXPIRED'', F.when(clean.FACEBOOK_TOKEN_EXPIRED == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''FACEBOOK_TOKEN_AVAILABLE'', F.when(clean.FACEBOOK_TOKEN_AVAILABLE == ''FALSE'', 0).otherwise(1))
    clean = clean.withColumn(''CLICKED_OR_NOT'', F.when(clean.CLICKED_OR_NOT == ''FALSE'', 0).otherwise(1))

    
    # Dropping the un-necssary columns before inferencing
    clean= clean.drop([''SCHOOL_SIZE_RATIO'',''IRU_COUNT_LAST_90_DAYS'',''IRU_COUNT_LAST_120_DAYS'',
                                   ''IRU_COUNT_LAST_365_DAYS'',''IRU_COUNT_LAST_730_DAYS'',''PHOTO_COUNT'',''LOGIN_COUNT_LAST_7_DAYS'',
                                  ''HINOTE_COUNT_LAST_90_DAYS'',''HINOTE_COUNT_LAST_120_DAYS'',''HINOTE_COUNT_LAST_365_DAYS'',''HINOTE_COUNT_LAST_730_DAYS'',
                            ''AGE_GROUP'',''LOGIN_COUNT_BUCKETS'',''PUBLISHER_OWNER_NAME'',''PUBLISHER_NAME'',''GB_MOMENTUM''
                               ,''SCHOOL_NAME'',''SCHOOL_CITY'',''SCHOOL_STATE'',''ACQUISITION_SOURCE'',''MEMBERSHIP_STATUS'',''EMAIL_DOMAIN'',         ''GENDER'',''MEMBERSHIP_STATUS_HISTORY'', ''EMAIL_DOMAIN_GROUP''])

    # Writing the cleaned data created from the pipeline above
    tgt_table_full_name_feature_engg = f"{tgt_db_name}.{tgt_schema_name}.FEAT_ENGG_DATASET_FINAL"
    clean.write.mode(''overwrite'').save_as_table(f"{tgt_table_full_name_feature_engg}")

    # Reading the cleaned data created from the pipeline above
    df= session.table(tgt_table_full_name_feature_engg)
    
    # Creating the first chunk and writing it to SF
    df_test = df.sample(n=35000000)
    df_test.write.mode(''overwrite'').save_as_table(f"{tgt_db_name}.{tgt_schema_name}.TEST_35M")
    df1= session.table(f"{tgt_db_name}.{tgt_schema_name}.TEST_35M")

    # Creating the second chunk and writing it to SF
    df2= df.subtract(df1)
    df2.write.mode(''overwrite'').save_as_table(f"{tgt_db_name}.{tgt_schema_name}.TEST_35M_PART2")    
    
    # Return value will appear in the Results tab.
    return "Data Cleaning & Feature Engineering Pipeline Ran Successfully!"';


    call  KIPI.ADMIN.SP_DATACLEANING_FEATENGG_PIPELINE('KIPI','PUBLIC','MASTER_MARKETING_DATA2','KIPI','ADMIN');
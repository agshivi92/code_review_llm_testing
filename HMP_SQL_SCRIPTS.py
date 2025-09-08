-- Main Sql query to create the mastertable considering few filter conditions on PROD_HMPFPD_DATA_POINTS_FLAT table. This query is supposed to run first.

CREATE OR REPLACE TRANSIENT TABLE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW AS
WITH filtered_data AS (
    SELECT *
    FROM HMP_FPD.PRODUCTION.PROD_HMPFPD_DATA_POINTS_FLAT
    WHERE  (FIX_Spec is not null AND  FIX_Spec <> '' AND  FIX_Spec <> 'null') and VOCAB IN ('drug', 'disease')
      AND (Value IS NOT NULL AND  VALUE <> '' AND  VALUE <> 'null') and EVENTDATE >=('2024-05-01') 
), 
pivoted_data AS (
    SELECT *
    FROM filtered_data
    PIVOT (
        COUNT(BC_ID) 
        FOR source IN ('ENL', 'nlp', 'riddle', 'js', '0', 'NEI_APP')
    ) AS pivot_table_alias
),
aggregated_data AS (
    SELECT VALUE, NPPES_NPI,VOCAB,FIX_SPEC,
           LISTAGG(DISTINCT (replace(a.EMAIL,'''','')), ';\n') WITHIN GROUP (ORDER BY replace(a.EMAIL,'''','')) AS ENGAGED_EMAIL,
           MAX(EVENTDATE) AS EVENTDATE,
           max(INGESTED_TIME) as INGESTED_TIME,
           SUM("'ENL'") AS email_count,
           SUM("'nlp'") AS nlp_count, 
           SUM("'riddle'") AS riddle_count,
           SUM("'js'") AS js_count,
           SUM("'0'") AS others_count,
           SUM("'NEI_APP'") AS NEI_APP_count,
           (SUM("'ENL'") + SUM("'nlp'") + SUM("'riddle'") + SUM("'js'") + SUM("'0'") + SUM("'NEI_APP'")) AS total_count
    FROM pivoted_data a
    GROUP BY NPPES_NPI, VALUE,VOCAB,FIX_SPEC
),
joined_data AS (
    SELECT c.NPPES_NPI, c.FIX_SPEC, c.VALUE,c.vocab,
           MAX(np.PROVIDER_FIRST_NAME) AS PROVIDER_FIRST_NAME,
           MAX(np.PROVIDER_LAST_NAME_LEGAL_NAME) AS PROVIDER_LAST_NAME,
           MAX(np.PROVIDER_ORGANIZATION_NAME_LEGAL_BUSINESS_NAME) AS PROVIDER_ORGANIZATION_NAME,
           MAX(REPLACE(REPLACE(np.PROVIDER_CREDENTIAL_TEXT, '.', ''), ' ', '')) AS PROVIDER_DEGREE,
           --MAX(np.PROVIDER_CREDENTIAL_TEXT) AS PROVIDER_DEGREE,
           COALESCE(MAX(PROVIDER_FIRST_LINE_BUSINESS_MAILING_ADDRESS), '') || ', ' ||
           COALESCE(NULLIF(MAX(PROVIDER_SECOND_LINE_BUSINESS_MAILING_ADDRESS), ''), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_CITY_NAME), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_STATE_NAME), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_POSTAL_CODE), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_COUNTRY_CODE_IF_OUTSIDE_US), '') AS Business_Mailing_Address,
           MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_TELEPHONE_NUMBER) AS Provider_Telephone_Number,
           c.ENGAGED_EMAIL,
           c.email_count, c.nlp_count, c.riddle_count, c.js_count, c.others_count, c.NEI_APP_count, c.total_count,
           bb.classification,
           c.EVENTDATE,c.INGESTED_TIME,
           DATE(MAX(bu.lastvisitdate)) AS MAX_LASTVISITDATE
    FROM aggregated_data c
    LEFT JOIN CUSTOMERS_DEV.BLUECONIC_DEV.FPD_BLUECONIC_BASE bb ON c.NPPES_NPI = bb.NPPES_NPI
    LEFT JOIN web.public.BLUECONIC_USERS bu ON c.NPPES_NPI = bu.NPI
    LEFT JOIN NATIONAL_PROVIDER_IDENTIFIER_NPI_REGISTRY.HLS_NPPES.NPI_DATA_MONTHLY np ON c.NPPES_NPI = np.npi
    WHERE bu.NPI_DEACTIVATION_DATE IS NULL   
    GROUP BY c.VALUE, c.NPPES_NPI,c.vocab, c.email_count, c.nlp_count, c.riddle_count, c.js_count, 
             c.others_count, c.NEI_APP_count, c.total_count, bb.classification, c.FIX_SPEC, c.EVENTDATE,c.INGESTED_TIME, c.ENGAGED_EMAIL
),
listagg_data AS (
    SELECT jd.*,
           LISTAGG(DISTINCT (replace(bu.EMAIL,'''','')), ';\n') WITHIN GROUP (ORDER BY replace(bu.EMAIL,'''','')) AS ALL_EMAILS_BLUECONIC
    FROM joined_data jd
    LEFT JOIN web.public.BLUECONIC_USERS bu ON jd.NPPES_NPI = bu.NPI
    GROUP BY jd.NPPES_NPI, jd.FIX_SPEC, jd.VALUE, jd.vocab,jd.PROVIDER_FIRST_NAME, jd.PROVIDER_LAST_NAME, 
             jd.PROVIDER_ORGANIZATION_NAME, jd.PROVIDER_DEGREE, jd.Business_Mailing_Address, jd.Provider_Telephone_Number,
             jd.ENGAGED_EMAIL, jd.email_count, jd.nlp_count, jd.riddle_count, jd.js_count, 
             jd.others_count, jd.NEI_APP_count, jd.total_count, jd.classification, jd.EVENTDATE,jd.INGESTED_TIME, jd.MAX_LASTVISITDATE
)
SELECT * FROM listagg_data;

-- This below sql query is for data cleaning, removes the data points where drug/disease name has numerical values and has function supports in its name. This query is supoosed to run 2nd in the sequence.

DELETE FROM CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW WHERE REGEXP_LIKE(VALUE, '^[0-9]+$') and value like '%function supports%';

--this sql query given below is supposed to run 3rd in the sequence, and is used to add a new column embedding_value in the mastertable.
ALTER TABLE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW ADD COLUMN embedding_value VECTOR(FLOAT, 768);

--this sql query given below is supposed to run 4th in the sequence, and is used to insert data in embedding_value column of the mastertable.
UPDATE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW SET embedding_value = snowflake.cortex.EMBED_TEXT_768('snowflake-arctic-embed-m', value);

--this sql query given below is supposed to run 5th in the sequence, and is used to add a new column HASH_COLUMN in the mastertable required for data pipeline merge condition.
alter table  CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW add column HASH_COLUMN varchar;

--this sql query given below is supposed to run 6th in the sequence, and is used to add a insert data in the HASH_COLUMN in the mastertable.
update  CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW set HASH_COLUMN = hash(NPPES_NPI,VALUE,VOCAB, FIX_SPEC);

------------------------------------------------------------------------------------------------------------------


--TASK (use this only if you want to automate the mastertable creation once a week)
-- if you want to schedule the mastertable creation once a week you can also do that, for this the below task can be used
-- Note this task is not created only script is written for you to use if needed in future.

create or replace task CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_CREATION_TASK
	warehouse=DATA_SCIENCE_WH
	schedule='USING CRON 0 19 * * 1 UTC' --scheduling the run at 7pm UTC time on every Monday 
	as 
BEGIN 

CREATE OR REPLACE TRANSIENT TABLE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW AS
WITH filtered_data AS (
    SELECT *
    FROM HMP_FPD.PRODUCTION.PROD_HMPFPD_DATA_POINTS_FLAT
    WHERE  (FIX_Spec is not null AND  FIX_Spec <> '' AND  FIX_Spec <> 'null') and VOCAB IN ('drug', 'disease')
      AND (Value IS NOT NULL AND  VALUE <> '' AND  VALUE <> 'null') and EVENTDATE >=('2024-05-01') 
), 
pivoted_data AS (
    SELECT *
    FROM filtered_data
    PIVOT (
        COUNT(BC_ID) 
        FOR source IN ('ENL', 'nlp', 'riddle', 'js', '0', 'NEI_APP')
    ) AS pivot_table_alias
),
aggregated_data AS (
    SELECT VALUE, NPPES_NPI,VOCAB,FIX_SPEC,
           LISTAGG(DISTINCT (replace(a.EMAIL,'''','')), ';\n') WITHIN GROUP (ORDER BY replace(a.EMAIL,'''','')) AS ENGAGED_EMAIL,
           MAX(EVENTDATE) AS EVENTDATE,
           max(INGESTED_TIME) as INGESTED_TIME,
           SUM("'ENL'") AS email_count,
           SUM("'nlp'") AS nlp_count, 
           SUM("'riddle'") AS riddle_count,
           SUM("'js'") AS js_count,
           SUM("'0'") AS others_count,
           SUM("'NEI_APP'") AS NEI_APP_count,
           (SUM("'ENL'") + SUM("'nlp'") + SUM("'riddle'") + SUM("'js'") + SUM("'0'") + SUM("'NEI_APP'")) AS total_count
    FROM pivoted_data a
    GROUP BY NPPES_NPI, VALUE,VOCAB,FIX_SPEC
),
joined_data AS (
    SELECT c.NPPES_NPI, c.FIX_SPEC, c.VALUE,c.vocab,
           MAX(np.PROVIDER_FIRST_NAME) AS PROVIDER_FIRST_NAME,
           MAX(np.PROVIDER_LAST_NAME_LEGAL_NAME) AS PROVIDER_LAST_NAME,
           MAX(np.PROVIDER_ORGANIZATION_NAME_LEGAL_BUSINESS_NAME) AS PROVIDER_ORGANIZATION_NAME,
           MAX(REPLACE(REPLACE(np.PROVIDER_CREDENTIAL_TEXT, '.', ''), ' ', '')) AS PROVIDER_DEGREE,
           --MAX(np.PROVIDER_CREDENTIAL_TEXT) AS PROVIDER_DEGREE,
           COALESCE(MAX(PROVIDER_FIRST_LINE_BUSINESS_MAILING_ADDRESS), '') || ', ' ||
           COALESCE(NULLIF(MAX(PROVIDER_SECOND_LINE_BUSINESS_MAILING_ADDRESS), ''), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_CITY_NAME), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_STATE_NAME), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_POSTAL_CODE), '') || ', ' ||
           COALESCE(MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_COUNTRY_CODE_IF_OUTSIDE_US), '') AS Business_Mailing_Address,
           MAX(PROVIDER_BUSINESS_MAILING_ADDRESS_TELEPHONE_NUMBER) AS Provider_Telephone_Number,
           c.ENGAGED_EMAIL,
           c.email_count, c.nlp_count, c.riddle_count, c.js_count, c.others_count, c.NEI_APP_count, c.total_count,
           bb.classification,
           c.EVENTDATE,c.INGESTED_TIME,
           DATE(MAX(bu.lastvisitdate)) AS MAX_LASTVISITDATE
    FROM aggregated_data c
    LEFT JOIN CUSTOMERS_DEV.BLUECONIC_DEV.FPD_BLUECONIC_BASE bb ON c.NPPES_NPI = bb.NPPES_NPI
    LEFT JOIN web.public.BLUECONIC_USERS bu ON c.NPPES_NPI = bu.NPI
    LEFT JOIN NATIONAL_PROVIDER_IDENTIFIER_NPI_REGISTRY.HLS_NPPES.NPI_DATA_MONTHLY np ON c.NPPES_NPI = np.npi
    WHERE bu.NPI_DEACTIVATION_DATE IS NULL   
    GROUP BY c.VALUE, c.NPPES_NPI,c.vocab, c.email_count, c.nlp_count, c.riddle_count, c.js_count, 
             c.others_count, c.NEI_APP_count, c.total_count, bb.classification, c.FIX_SPEC, c.EVENTDATE,c.INGESTED_TIME, c.ENGAGED_EMAIL
),
listagg_data AS (
    SELECT jd.*,
           LISTAGG(DISTINCT (replace(bu.EMAIL,'''','')), ';\n') WITHIN GROUP (ORDER BY replace(bu.EMAIL,'''','')) AS ALL_EMAILS_BLUECONIC
    FROM joined_data jd
    LEFT JOIN web.public.BLUECONIC_USERS bu ON jd.NPPES_NPI = bu.NPI
    GROUP BY jd.NPPES_NPI, jd.FIX_SPEC, jd.VALUE, jd.vocab,jd.PROVIDER_FIRST_NAME, jd.PROVIDER_LAST_NAME, 
             jd.PROVIDER_ORGANIZATION_NAME, jd.PROVIDER_DEGREE, jd.Business_Mailing_Address, jd.Provider_Telephone_Number,
             jd.ENGAGED_EMAIL, jd.email_count, jd.nlp_count, jd.riddle_count, jd.js_count, 
             jd.others_count, jd.NEI_APP_count, jd.total_count, jd.classification, jd.EVENTDATE,jd.INGESTED_TIME, jd.MAX_LASTVISITDATE
)
SELECT * FROM listagg_data;

DELETE FROM CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW WHERE REGEXP_LIKE(VALUE, '^[0-9]+$') and value like '%function supports%';

ALTER TABLE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW ADD COLUMN embedding_value VECTOR(FLOAT, 768);

UPDATE CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW SET embedding_value = snowflake.cortex.EMBED_TEXT_768('snowflake-arctic-embed-m', value);

alter table  CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW add column HASH_COLUMN varchar;

update  CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW set HASH_COLUMN = hash(NPPES_NPI,VALUE,VOCAB, FIX_SPEC);

END;

ALTER TASK CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_CREATION_TASK resume;

-----------------------------------------------------------------------------------------------------------------------------
-- DATAPIPELINE SP and TASK -- this SP and task are created and task is scheduled to run daily at 4 am UTC time

--creating the email integration and allowing the email id which should be notified upon SP failure
CREATE or replace NOTIFICATION INTEGRATION Task_Fail_Email_Notifications
  TYPE=EMAIL
  ENABLED=TRUE
  ALLOWED_RECIPIENTS=('shivangi.j.agarwal@kipi.bi','swati.a.dey@kipi.ai','arindam.t.chatterjee@kipi.bi','yash.s.dixit@kipi.bi');

SHOW NOTIFICATION INTEGRATIONS;

-- Stored Procedure
CREATE OR REPLACE PROCEDURE CUSTOMERS_DEV.DATA_SCIENCE_DEV.SP_INSERT_UPDATE_MASTER_TABLE_ALL_SPECS()
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'main'
EXECUTE AS CALLER
AS $$

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def main(session: snowpark.Session):
    try:
        session.sql("BEGIN").collect()
        # takes the latest ingested time from MASTERTABLE_ALL_SPECS_NEW and from that time it captures the delta record.
        npi_eventdate = session.sql(f"select max(INGESTED_TIME) from CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW ").collect()  
        
        if not npi_eventdate or not npi_eventdate[0][0]:
            session.sql("COMMIT").collect()
            
            return "No new records found in the source table"
                  
        else:
            latest_eventdate = npi_eventdate[0][0]
    
            session.sql(f"""
                CREATE OR REPLACE TEMPORARY TABLE TEMP_FILTERED_DATA_ALL_SPECS AS 
                    WITH filtered_data AS (
                        SELECT *
                        FROM HMP_FPD.PRODUCTION.PROD_HMPFPD_DATA_POINTS_FLAT
                        WHERE INGESTED_TIME > '{latest_eventdate}'
                          AND (FIX_Spec is not null AND  FIX_Spec <> '' AND  FIX_Spec <> 'null') 
                          AND VOCAB IN ('drug', 'disease')
                          AND (Value IS NOT NULL AND  VALUE <> '' AND  VALUE <> 'null')
                    ),
                    aggregated_data AS (
                        SELECT 
                            VALUE, 
                            NPPES_NPI, 
                            VOCAB, 
                            FIX_SPEC,
                            LISTAGG(DISTINCT REPLACE(email, '''', '')) WITHIN GROUP (ORDER BY REPLACE(email, '''', '')) AS ENGAGED_EMAIL,
                            MAX(EVENTDATE) AS EVENTDATE,
                            MAX(INGESTED_TIME) AS INGESTED_TIME,
                            SUM(CASE WHEN SOURCE = 'ENL' THEN 1 ELSE 0 END) AS email_count,
                            SUM(CASE WHEN SOURCE = 'nlp' THEN 1 ELSE 0 END) AS nlp_count,
                            SUM(CASE WHEN SOURCE = 'riddle' THEN 1 ELSE 0 END) AS riddle_count,
                            SUM(CASE WHEN SOURCE = 'js' THEN 1 ELSE 0 END) AS js_count,
                            SUM(CASE WHEN SOURCE = '0' THEN 1 ELSE 0 END) AS others_count,
                            SUM(CASE WHEN SOURCE = 'NEI_APP' THEN 1 ELSE 0 END) AS NEI_APP_count,
                            COUNT(*) AS total_count
                        FROM filtered_data
                        GROUP BY VALUE, NPPES_NPI, VOCAB, FIX_SPEC
                    ),
                    pivoted_data AS (
                        SELECT 
                            a.NPPES_NPI, 
                            REPLACE(a.VALUE, '''', '') AS VALUE, 
                            a.VOCAB, 
                            a.FIX_SPEC,
                            HASH(a.NPPES_NPI, a.VALUE, a.VOCAB, a.FIX_SPEC) AS HASH_COLUMN,
                            a.ENGAGED_EMAIL AS New_Email_List,
                            MAX(REPLACE(np.PROVIDER_FIRST_NAME, '''', '')) AS PROVIDER_FIRST_NAME,
                            MAX(REPLACE(np.PROVIDER_LAST_NAME_LEGAL_NAME, '''', '')) AS PROVIDER_LAST_NAME,
                            MAX(REPLACE(np.PROVIDER_ORGANIZATION_NAME_LEGAL_BUSINESS_NAME, '''', '')) AS PROVIDER_ORGANIZATION_NAME,
                            MAX(REPLACE(REPLACE(np.PROVIDER_CREDENTIAL_TEXT, '.', ''), ' ', '')) AS PROVIDER_DEGREE,
                            COALESCE(MAX(REPLACE(np.PROVIDER_FIRST_LINE_BUSINESS_MAILING_ADDRESS, '''', '')), '') || ', ' ||
                            COALESCE(NULLIF(MAX(REPLACE(np.PROVIDER_SECOND_LINE_BUSINESS_MAILING_ADDRESS, '''', '')), ''), '') || ', ' ||
                            COALESCE(MAX(REPLACE(np.PROVIDER_BUSINESS_MAILING_ADDRESS_CITY_NAME, '''', '')), '') || ', ' ||
                            COALESCE(MAX(REPLACE(np.PROVIDER_BUSINESS_MAILING_ADDRESS_STATE_NAME, '''', '')), '') || ', ' ||
                            COALESCE(MAX(REPLACE(np.PROVIDER_BUSINESS_MAILING_ADDRESS_POSTAL_CODE, '''', '')), '') || ', ' ||
                            COALESCE(MAX(REPLACE(np.PROVIDER_BUSINESS_MAILING_ADDRESS_COUNTRY_CODE_IF_OUTSIDE_US, '''', '')), '') AS Business_Mailing_Address,
                            MAX(np.PROVIDER_BUSINESS_MAILING_ADDRESS_TELEPHONE_NUMBER) AS Provider_Telephone_Number,
                            a.email_count, 
                            a.nlp_count, 
                            a.riddle_count, 
                            a.js_count, 
                            a.others_count, 
                            a.NEI_APP_count, 
                            a.total_count, 
                            bb.classification,
                            a.EVENTDATE,
                            a.INGESTED_TIME,
                            DATE(MAX(bu.lastvisitdate)) AS MAX_LASTVISITDATE,
                            LISTAGG(DISTINCT REPLACE(bu.email, '''', ''), ';\n') WITHIN GROUP (ORDER BY REPLACE(bu.email, '''', '')) AS ALL_EMAILS_BLUECONIC
                        FROM aggregated_data a
                        LEFT JOIN web.public.BLUECONIC_USERS bu 
                            ON a.NPPES_NPI = bu.NPI
                            
                        LEFT JOIN CUSTOMERS_DEV.BLUECONIC_DEV.FPD_BLUECONIC_BASE bb 
                            ON a.NPPES_NPI = bb.NPPES_NPI
                        LEFT JOIN NATIONAL_PROVIDER_IDENTIFIER_NPI_REGISTRY.HLS_NPPES.NPI_DATA_MONTHLY np 
                            ON a.NPPES_NPI = np.npi where bu.NPI_DEACTIVATION_DATE IS NULL
                        GROUP BY 
                            a.NPPES_NPI, 
                            a.VALUE, 
                            a.VOCAB, 
                            a.FIX_SPEC, 
                            bb.classification, 
                            a.email_count, 
                            a.nlp_count, 
                            a.riddle_count, 
                            a.js_count, 
                            a.others_count, 
                            a.NEI_APP_count, 
                            a.total_count, 
                            a.EVENTDATE, 
                            a.INGESTED_TIME, 
                            a.ENGAGED_EMAIL
                    )
                    SELECT *  FROM pivoted_data;   
        """).collect()

            session.sql(f"""DELETE FROM TEMP_FILTERED_DATA_ALL_SPECS
                WHERE REGEXP_LIKE(VALUE, '^[0-9]+$') and value like '%function supports%' """).collect()
                
            session.sql(f"""
                CREATE OR REPLACE TEMPORARY TABLE TEMP_ENGAGED_EMAIL_ALL_SPECS AS
                    SELECT 
                        HASH_COLUMN,
                        LISTAGG(DISTINCT email, ';\n') WITHIN GROUP (ORDER BY email) AS MERGED_ENGAGED_EMAIL
                    FROM (
                            SELECT HASH_COLUMN,  SPLIT_RESULT.VALUE AS email
                            FROM TEMP_FILTERED_DATA_ALL_SPECS,
                            LATERAL SPLIT_TO_TABLE(NEW_EMAIL_LIST, ';\n') AS SPLIT_RESULT
                            WHERE REGEXP_LIKE( SPLIT_RESULT.VALUE, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
                            UNION ALL
                            SELECT HASH_COLUMN,  SPLIT_RESULT.VALUE AS email
                            FROM CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW,
                            LATERAL SPLIT_TO_TABLE(ENGAGED_EMAIL, ';\n') as SPLIT_RESULT
                            WHERE REGEXP_LIKE( SPLIT_RESULT.VALUE , '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
                        
                    )
                    GROUP BY HASH_COLUMN; """).collect()
                    
            session.sql(f"""
                CREATE OR REPLACE TEMPORARY TABLE TEMP_ALL_EMAIL_ALL_SPECS AS
                SELECT 
                    HASH_COLUMN,
                    LISTAGG(DISTINCT email, ';\n') WITHIN GROUP (ORDER BY email) AS MERGED_ENGAGED_EMAIL
                FROM (
                        SELECT HASH_COLUMN,  SPLIT_RESULT.VALUE AS email
                        FROM TEMP_FILTERED_DATA_ALL_SPECS,
                        LATERAL SPLIT_TO_TABLE(ALL_EMAILS_BLUECONIC, ';\n') AS SPLIT_RESULT
                        WHERE REGEXP_LIKE( SPLIT_RESULT.VALUE, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
                        UNION ALL
                        SELECT HASH_COLUMN,  SPLIT_RESULT.VALUE AS email
                        FROM CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW,
                        LATERAL SPLIT_TO_TABLE(ALL_EMAILS_BLUECONIC, ';\n') as SPLIT_RESULT
                        WHERE REGEXP_LIKE( SPLIT_RESULT.VALUE , '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
                    
                )
                GROUP BY HASH_COLUMN; """).collect()
                    
            session.sql(f"""
                    MERGE INTO CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW AS target
                    USING (
                        SELECT 
                            src.*, 
                            tmp.MERGED_ENGAGED_EMAIL AS NEW_ENGAGED_EMAIL,
                            tmp_all_email.MERGED_ENGAGED_EMAIL AS NEW_ALL_EMAILS_BLUECONIC
                        FROM TEMP_FILTERED_DATA_ALL_SPECS AS src
                        LEFT JOIN TEMP_ENGAGED_EMAIL_ALL_SPECS AS tmp
                        ON src.HASH_COLUMN = tmp.HASH_COLUMN
                        LEFT JOIN TEMP_ALL_EMAIL_ALL_SPECS AS tmp_all_email  on src.HASH_COLUMN = tmp_all_email.HASH_COLUMN
                    ) AS source
                    ON target.HASH_COLUMN = source.HASH_COLUMN
                    WHEN MATCHED THEN
                        UPDATE SET
                            EVENTDATE = GREATEST(target.EVENTDATE, source.EVENTDATE),
                            INGESTED_TIME = GREATEST(target.INGESTED_TIME, source.INGESTED_TIME),
                            ENGAGED_EMAIL = source.NEW_ENGAGED_EMAIL,
                            EMAIL_COUNT = target.EMAIL_COUNT + source.email_count,
                            NLP_COUNT = target.NLP_COUNT + source.nlp_count,
                            RIDDLE_COUNT = target.RIDDLE_COUNT + source.riddle_count,
                            JS_COUNT = target.JS_COUNT + source.js_count,
                            OTHERS_COUNT = target.OTHERS_COUNT + source.others_count,
                            NEI_APP_COUNT = target.NEI_APP_COUNT + source.NEI_APP_count,
                            TOTAL_COUNT = target.TOTAL_COUNT + source.total_count,
                            ALL_EMAILS_BLUECONIC = source.NEW_ALL_EMAILS_BLUECONIC
                    WHEN NOT MATCHED THEN
                        INSERT (NPPES_NPI, FIX_SPEC, VALUE, VOCAB, PROVIDER_FIRST_NAME, PROVIDER_LAST_NAME, PROVIDER_ORGANIZATION_NAME,
                                PROVIDER_DEGREE, BUSINESS_MAILING_ADDRESS, PROVIDER_TELEPHONE_NUMBER, EVENTDATE, INGESTED_TIME, ENGAGED_EMAIL, EMAIL_COUNT, NLP_COUNT, 
                                RIDDLE_COUNT, JS_COUNT, OTHERS_COUNT, NEI_APP_COUNT, TOTAL_COUNT, CLASSIFICATION, MAX_LASTVISITDATE, ALL_EMAILS_BLUECONIC, EMBEDDING_VALUE, HASH_COLUMN)
                        VALUES (source.NPPES_NPI, source.FIX_SPEC, source.VALUE, source.VOCAB, source.PROVIDER_FIRST_NAME, source.PROVIDER_LAST_NAME, source.PROVIDER_ORGANIZATION_NAME,
                                source.PROVIDER_DEGREE, source.BUSINESS_MAILING_ADDRESS, source.PROVIDER_TELEPHONE_NUMBER, source.EVENTDATE, source.INGESTED_TIME, 
                                source.NEW_ENGAGED_EMAIL, source.email_count, source.nlp_count, source.riddle_count, source.js_count, 
                                source.others_count, source.NEI_APP_count, source.total_count, source.CLASSIFICATION, source.MAX_LASTVISITDATE, 
                                source.NEW_ALL_EMAILS_BLUECONIC, (SELECT snowflake.cortex.EMBED_TEXT_768('snowflake-arctic-embed-m', source.VALUE)), 
                                HASH(source.NPPES_NPI, source.VALUE, source.VOCAB, source.FIX_SPEC));
                    
                            """).collect()
                   
                    
                    
        session.sql("COMMIT").collect()

        
    except Exception as e:
        # Rollback in case of an error
        error_message = str(e).replace("'", "''")
        session.sql("rollback").collect()
        session.sql(f""" call system$send_email('Task_Fail_Email_Notifications','shivangi.j.agarwal@kipi.bi,swati.a.dey@kipi.ai,arindam.t.chatterjee@kipi.bi,yash.s.dixit@kipi.bi','Email Alert:Error while merging data to Mastertable all specs', 'Please check the error: {error_message}') """).collect()
        
        return f"Failed with error: {error_message}"
    return "Success"
$$    

;


CREATE or replace TASK TASK_INSERT_UPDATE_MASTER_TABLE_ALL_SPECS
    WAREHOUSE = DATA_SCIENCE_WH
  SCHEDULE = 'USING CRON 0 4 * * * UTC' --scheduling the run at 4am UTC time on everyday 
  USER_TASK_TIMEOUT_MS = 86400000
  AS
    call CUSTOMERS_DEV.DATA_SCIENCE_DEV.SP_INSERT_UPDATE_MASTER_TABLE_ALL_SPECS();

ALTER TASK TASK_INSERT_UPDATE_MASTER_TABLE_ALL_SPECS resume;

-----------------------------------------------------------------------------------------------------------------------
-- Note -- you can have the Task for table creation once a week that is on Monday or any other day or the task for data pipeline to run on
--Tue, Wed, Thu, Fri , which will in turn save the cost and maintain a balance between accuracy and credit consumption.
-- for this requirement only the Task schedule has to be changed as per your requirement.

---------------------------------------------------------------------------------------------------------------------------------------
select count(*) from CUSTOMERS_DEV.DATA_SCIENCE_DEV.MASTERTABLE_ALL_SPECS_NEW;

CREATE OR REPLACE PROCEDURE ml_preprocessing(TABLENAME VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-ml-python', 'numpy', 'pandas')
HANDLER = 'ml_preprocessing_handler'
AS
$$
import os
import pickle
import numpy as np
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import lit, map_from_arrays, get, col, array_construct, coalesce, when, row_number
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import StringType, FloatType

def apply_industry_grouping(df, industry_column='INDUSTRY', mapping_dict=None, unmapped_category='Other/Unspecified'):
            if mapping_dict is None:
                raise ValueError("mapping_dict must be provided.")
            case_expr = None
            for broader_cat, specific_cats in mapping_dict.items():
                for specific_cat in specific_cats:
                    condition = F.col(industry_column) == F.lit(specific_cat)
                    if case_expr is None:
                        case_expr = F.when(condition, F.lit(broader_cat))
                    else:
                        case_expr = case_expr.when(condition, F.lit(broader_cat))
            case_expr = case_expr.otherwise(F.lit(unmapped_category))
            df = df.with_column('BROADER_INDUSTRY', case_expr)
            df = df.drop(industry_column)
            return df
            
# Helper function to cap outliers
def cap_outliers_snowpark(df, col_name):
    Q1 = df.select(F.expr(f"percentile_cont(0.25) within group (order by {col_name})")).collect()[0][0]
    Q3 = df.select(F.expr(f"percentile_cont(0.75) within group (order by {col_name})")).collect()[0][0]
    IQR = float(Q3) - float(Q1)
    lower_bound = float(Q1) - 1.5 * IQR
    upper_bound = float(Q3) + 1.5 * IQR
    capped_col = F.when(F.col(col_name) > upper_bound, upper_bound)\
                  .when(F.col(col_name) < lower_bound, lower_bound)\
                  .otherwise(F.col(col_name))
    return df.with_column(col_name, capped_col)

# Load preprocessor object from stage
def load_object_from_stage(session, stage_name, filename_on_stage, folder_name):
    local_temp_path = '/tmp/'
    full_stage_path = f"{stage_name.strip('/')}/{folder_name}/{filename_on_stage}"
    session.file.get(full_stage_path, local_temp_path)
    downloaded_file_path = os.path.join(local_temp_path, filename_on_stage)
    with open(downloaded_file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    os.remove(downloaded_file_path)
    return loaded_object
        
def ml_preprocessing_handler(session, table_name):

    try:
        # Load data from table 
        snow_df = session.table(table_name)
        
        # Cast boolean columns to string (if exist)
        boolean_columns = [
            'FINANCE_SEGMENT', 'HAS_CSM', 'ENT_', 'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_1_MONTHS',
            'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_3_MONTHS', 'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_6_MONTHS',
            'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS', 'HAS_PSP', 'HAD_EBR',
            'PRIORITY_ACCOUNT_GC', 'PLG_SOURCED', 'FORMERLY_ON_PLUS', 'FORMERLY_ON_SCHOLARSHIP'
        ]
        for col_name in boolean_columns:
            if col_name in snow_df.columns:
                snow_df = snow_df.with_column(col_name, snow_df[col_name].cast(StringType()))
        
        # Define target variable and drop rows with missing target
        TARGET_VARIABLE = 'CHURN_TYPE'
        snow_df = snow_df.dropna(subset=[TARGET_VARIABLE])
        
        # Encode target variable
        target_mapping = {'no_churn': 0, 'partial_churn': 1, 'full_churn': 2}
        ENCODED_TARGET_VARIABLE = 'CHURN_TYPE_ENCODED'
        case_expr = None
        for category, encoded_value in target_mapping.items():
            if case_expr is None:
                case_expr = when(col(TARGET_VARIABLE) == category, lit(encoded_value))
            else:
                case_expr = case_expr.when(col(TARGET_VARIABLE) == category, lit(encoded_value))
        snow_df = snow_df.with_column(ENCODED_TARGET_VARIABLE, case_expr)
        
        # Drop unwanted columns
        cols_to_drop = [
            'AVG_NPSSCORE_PAST_1_MONTHS', 'AVG_NPSSCORE_PAST_3_MONTHS', 'AVG_NPSSCORE_PAST_6_MONTHS',
            'AVG_NPSSCORE_PAST_9_MONTHS', 'AVG_CSATSCORE_PAST_1_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_1_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_6_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_1_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_6_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_1_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_6_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_1_MONTHS',
            'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_6_MONTHS',
            'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_1_MONTHS',
            'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_6_MONTHS',
            'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_LESSONS_COUNT_PAST_1_MONTHS',
            'AVG_MONTHLY_LESSONS_COUNT_PAST_3_MONTHS', 'AVG_MONTHLY_LESSON_COMPLETION_RATIO_PAST_1_MONTHS', 'AVG_MONTHLY_LESSON_COMPLETION_RATIO_PAST_3_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_LAST_ACTIVITY_PAST_1_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_LAST_ACTIVITY_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_ENROLLED_PAST_1_MONTHS',
            'AVG_MONTHLY_DAYS_SINCE_ENROLLED_PAST_3_MONTHS', 'AVG_MONTHLY_TICKETS_OPEN_SINCE_90DAYS_PAST_1_MONTHS', 'AVG_MONTHLY_TICKETS_OPEN_SINCE_90DAYS_PAST_3_MONTHS',
            'HAS_PSP', 'FORMERLY_ON_SCHOLARSHIP'
        ]
        snow_df = snow_df.drop(*cols_to_drop)
        
        # Prepare feature columns and data types
        feature_cols = [c for c in snow_df.columns]
        train_dtypes = dict(snow_df.dtypes)
        numerical_features = []
        categorical_features = []
        for c in feature_cols:
            dtype_str = train_dtypes.get(c, '').lower()
            if any(x in dtype_str for x in ['int', 'float', 'double', 'decimal', 'numeric']):
                numerical_features.append(c)
            elif any(x in dtype_str for x in ['string', 'varchar', 'char', 'boolean', 'bool']):
                categorical_features.append(c)
        
        # Standardize country names using mapping
        COUNTRY_STANDARDIZATION_MAP = {
            # (same mapping as provided in user code)
            "Antigua and Barbuda": "Antigua and Barbuda", "Argentina": "Argentina", "Armenia": "Armenia",
            "Australia": "Australia", "Austria": "Austria", "Azerbaijan": "Azerbaijan", "Bahrain": "Bahrain",
            "Belarus": "Belarus", "Belgium": "Belgium", "Brazil": "Brazil", "Bulgaria": "Bulgaria",
            "Canada": "Canada", "Cayman Islands": "Cayman Islands", "Chile": "Chile", "China": "China",
            "Colombia": "Colombia", "Croatia": "Croatia", "Cyprus": "Cyprus", "Czech Republic": "Czech Republic",
            "Denmark": "Denmark", "Dominican Republic": "Dominican Republic", "Ecuador": "Ecuador",
            "Egypt": "Egypt", "El Salvador": "El Salvador", "Estonia": "Estonia", "Finland": "Finland",
            "France": "France", "Georgia": "Georgia", "Germany": "Germany", "Gibraltar": "Gibraltar",
            "Greece": "Greece", "Guatemala": "Guatemala", "Guernsey": "Guernsey", "Hong Kong": "Hong Kong",
            "Hungary": "Hungary", "Iceland": "Iceland", "India": "India", "Indonesia": "Indonesia",
            "Iraq": "Iraq", "Ireland": "Ireland", "Israel": "Israel", "Italy": "Italy",
            "Jamaica": "Jamaica", "Japan": "Japan", "Kazakhstan": "Kazakhstan", "Kenya": "Kenya",
            "Korea": "Korea, Republic of", "Korea, Democratic People's Republic of": "Korea, Democratic People's Republic of",
            "Korea, Republic of": "Korea, Republic of", "Kuwait": "Kuwait", "Kyrgyzstan": "Kyrgyzstan",
            "Latvia": "Latvia", "Lebanon": "Lebanon", "Lithuania": "Lithuania", "Luxembourg": "Luxembourg",
            "Malaysia": "Malaysia", "Malta": "Malta", "Mauritius": "Mauritius", "Mexico": "Mexico",
            "Moldova, Republic of": "Moldova, Republic of", "Morocco": "Morocco", "Netherlands": "Netherlands",
            "New Zealand": "New Zealand", "Nigeria": "Nigeria", "Norway": "Norway", "Pakistan": "Pakistan",
            "Peru": "Peru", "Philippines": "Philippines", "Poland": "Poland", "Portugal": "Portugal",
            "Qatar": "Qatar", "Romania": "Romania", "Russia": "Russian Federation",
            "Russian Federation": "Russian Federation", "Saudi Arabia": "Saudi Arabia", "Senegal": "Senegal",
            "Serbia": "Serbia", "Seychelles": "Seychelles", "Singapore": "Singapore", "Slovenia": "Slovenia",
            "South Africa": "South Africa", "South Korea": "Korea, Republic of", "Spain": "Spain",
            "Sweden": "Sweden", "Switzerland": "Switzerland", "Taiwan": "Taiwan",
            "Taiwan, Province of China": "Taiwan", "Thailand": "Thailand", "Turkey": "Turkey",
            "Ukraine": "Ukraine", "United Arab Emirates": "United Arab Emirates",
            "united kingdom": "United Kingdom", "United Kingdom": "United Kingdom",
            "United States": "United States", "Uruguay": "Uruguay", "US": "United States",
            "USA": "United States", "Uzbekistan": "Uzbekistan",
            "Venezuela, Bolivarian Republic of": "Venezuela, Bolivarian Republic of", "Viet Nam": "Vietnam",
            "Vietnam": "Vietnam", "Virgin Islands, British": "Virgin Islands, British"
        }
        map_keys = array_construct(*[lit(k) for k in COUNTRY_STANDARDIZATION_MAP.keys()])
        map_values = array_construct(*[lit(v) for v in COUNTRY_STANDARDIZATION_MAP.values()])
        country_map = map_from_arrays(map_keys, map_values)
        df_standardized = snow_df.with_column(
            "STANDARDIZED_COUNTRY",
            coalesce(get(country_map, col("HQ_COUNTRY_GC")), col("HQ_COUNTRY_GC"))
        ).drop("HQ_COUNTRY_GC")
        
        # Industry grouping function
        BROADER_INDUSTRY_CATEGORIES = {
            # (same dictionary as provided by user)
            "Technology & Software": [
                "Lending & Brokerage", "Consumer Tech", "Business Intelligence (BI) Software",
                "Security Software", "Networking Software", "Financial Software", "Engineering Software",
                "Mobile App Development", "Social Networks", "Content & Collaboration Software",
                "Internet Service Providers, Website Hosting & Internet-related Services",
                "Customer Relationship Management (CRM) Software", "Storage & System Management Software",
                "Telephony & Wireless", "Custom Software & IT Services", "Internet",
                "Database & File Management Software", "Multimedia, Games & Graphics Software",
                "E-Commerce and Marketplaces", "Data Collection & Internet Portals",
                "Information & Document Management", "Computer Equipment & Peripherals",
                "Technology Hardware, Storage & Periph...", "Technology", "Security Products & Services",
                "Legal Software", "Data Processing & Outsourced Services", "Graphic Design",
                "Biotechnology", "Photography Studio", "Multimedia & Graphic Design"
            ],
            "Financial Services & Insurance": [
                "Credit Cards & Transaction Processing", "Financial Services", "Banking",
                "Investment Banking", "Banking & Mortgages", "Professional Services", "Insurance",
                "Diversified Capital Markets", "Payments", "Accounting Services", "Diversified Financial Services",
                "Capital Markets", "Accounting", "Asset Management & Custody Banks", "Banks",
                "Venture Capital & Private Equity"
            ],
            # ... (include all other categories as per user code)
            "Healthcare & Medical": [
                "Healthcare Software",
                "Medical & Surgical Hospitals",
                "Medical Devices & Equipment",
                "Medical Specialists",
                "Mental Health & Rehabilitation Facilities",
                "Pharmaceuticals",
                "Healthcare",
                "Health Care Providers & Services",
                "Dental Offices",
                "Health Care Services",
                "Physicians Clinics",
                "Medical Laboratories & Imaging Centers",
                "Vitamins, Supplements & Health Stores",
                "Elderly Care Services",
                "Health & Nutrition Products",
                "Health Care",
                "Hospital & Health Care",
                "Health & Wellness"
            ],
            "Retail & Consumer Goods": [
                "Grocery Retail",
                "Drug Stores & Pharmacies",
                "Apparel & Accessories Retail",
                "Plastic, Packaging & Containers", # For consumer goods packaging
                "Toys & Games",
                "Textiles & Apparel",
                "Electronics",
                "Apparel, Accessories & Luxury Goods",
                "Flowers, Gifts & Specialty Stores",
                "Consumer Electronics & Computers Retail",
                "Department Stores, Shopping Centers & Superstores",
                "Furniture",
                "Jewelry & Watch Retail",
                "Retailing",
                "Specialty Retail",
                "Watches & Jewelry",
                "Consumer Staples",
                "Consumer Discretionary",
                "Household Goods",
                "Cleaning Products",
                "Packaged Foods & Meats", # If sold via retail
                "Pet Products",
                "Sporting Goods",
                "Sporting & Recreational Equipment Retail",
                "Home Improvement & Hardware Retail",
                "Record, Video & Book Stores",
                "Office Products Retail & Distribution",
                "Convenience Stores, Gas Stations & Liquor Stores",
                "Appliances",
                "Automobile Parts Stores",
                "Household Durables",
                "Eyewear",
                "Home Improvement Retail",
                "Consumer Goods" # Broad category
            ],
            "Media, Entertainment & Hospitality": [
                "Publishing",
                "Social Networks", # Content/entertainment focus
                "Restaurants",
                "Fitness & Dance Facilities",
                "Music Production & Services",
                "Broadcasting",
                "Data Collection & Internet Portals", # If content aggregation/news portals
                "Gambling & Gaming",
                "Media",
                "Newspapers & News Services",
                "Photography Studio", # If primarily for entertainment/events
                "Sports Teams & Leagues",
                "Movies & Entertainment",
                "Lodging & Resorts",
                "Amusement Parks, Arcades & Attractions",
                "Hotels, Restaurants & Leisure",
                "Performing Arts Theaters",
                "Casinos & Gaming",
                "Leisure Products"
            ],
            "Professional & Business Services": [
                "Other Rental Stores (Furniture, A/V, Construction & Industrial Equipment)",
                "Training",
                "Human Resources Software", # If referring to HR services platforms
                "HR & Staffing",
                "Repair Services", # General repair services
                "Management Consulting",
                "Commercial Printing",
                "Professional Services",
                "Legal Services",
                "Automotive Service & Collision Repair",
                "Advertising & Marketing",
                "Human Resource & Employment Services",
                "B2B", # Broad B2B services
                "Consulting",
                "Research & Development",
                "Research & Consulting Services",
                "Diversified Consumer Services", # If consumer-facing services beyond retail/healthcare
                "Call Centers & Business Centers",
                "Family Services",
                "Translation & Linguistic Services",
                "Cleaning Services",
                "Landscape Services",
                "Photographic & Optical Equipment", # If service based
                "Travel and Hospitality", # If related to professional event planning etc.
                "Consumer Services" # Broad consumer services
            ],
            "Education & Non-Profit": [
                "Non-Profit & Charitable Organizations",
                "Education Services",
                "Colleges & Universities",
                "Religious Organizations",
                "K-12 Schools",
                "Education",
                "Cultural & Informational Centers",
                "Fundraising",
                "Childcare"
            ],
            "Industrial, Manufacturing & Utilities": [
                "Industrials & Manufacturing",
                "Plastic, Packaging & Containers", # Manufacturing of containers
                "Industrial Machinery & Equipment",
                "Electricity, Oil & Gas",
                "Telecommunication Equipment",
                "Automotive", # Manufacturing focus
                "Motor Vehicles",
                "Electrical Equipment",
                "Chemicals & Related Products",
                "Aerospace & Defense",
                "Boats & Submarines",
                "Test & Measurement Equipment",
                "Distributors", # Industrial distributors
                "Household Durables", # Manufacturing focus
                "Manufacturing" # Assuming this is meant to be a general manufacturing category
            ],
            "Transportation & Logistics": [
                "Trucking, Moving & Storage",
                "Freight & Logistics Services",
                "Travel Agencies & Services",
                "Rail, Bus & Taxi",
                "Airlines, Airports & Air Services",
                "Transportation",
                "Car & Truck Rental",
                "Wireless Telecommunication Services", # Communication infrastructure for transport/logistics
                "Diversified Telecommunication Services", # Could involve logistics network
                "Shipping & Logistics",
                "Ground Transportation"
            ],
            "Agriculture & Food Production": [
                "Food & Beverage", # Production focus
                "Crops",
                "Animals & Livestock",
                "Food Service", # Production/wholesale
                "Food Products",
                "Forestry",
                "Agricultural Products",
                "al Products" # Assuming this is a typo and should be Agricultural Products
            ],
            "Real Estate & Construction": [
                "Architecture",
                "Commercial & Residential Construction",
                "Construction & Engineering",
                "Building Materials",
                "Real Estate",
                "Architecture, Engineering & Design",
                "Civil Engineering Construction"
            ],
            "Government & Public Sector": [
                "Federal" # Specific government category
            ],
            "Environmental Services": [
                "Waste Treatment, Environmental Services & Recycling"
            ],
            "Other/Miscellaneous": [ # For any remaining or ambiguous items
                "Other" # Explicitly listed as 'Other'
            ]
        }
        

        df_capped = apply_industry_grouping(df_standardized, 'INDUSTRY', BROADER_INDUSTRY_CATEGORIES, 'Other/Unspecified')
        
        # Cast decimal columns to float in train and test dfs (assuming snow_train_df and snow_test_df exist)
        snow_train_df = df_capped
        
        for col_name in numerical_features:
            col_dtype = dict(snow_train_df.dtypes).get(col_name, '').lower()
            if col_dtype.startswith('decimal'):
                snow_train_df = snow_train_df.with_column(col_name, F.col(col_name).cast(FloatType()))
        
        # Remove some columns from categorical_features
        for col_to_remove in ['ACCOUNT_ID', 'ACCOUNT_NAME','CHURN_TYPE', 'BLADES_PURCHASED', 'STANDARDIZED_COUNTRY']:
            if col_to_remove in categorical_features:
                categorical_features.remove(col_to_remove)
        if 'CHURN_TYPE_ENCODED' in numerical_features:
            numerical_features.remove('CHURN_TYPE_ENCODED')
        
        
        # Cap outliers for numerical features
        df_capped_outliers = snow_train_df
        for col_name in numerical_features:
            df_capped_outliers = cap_outliers_snowpark(df_capped_outliers, col_name)
        
        
        
        MODELS_STAGE = '@MODELS_STAGE'  # Replace with your actual stage name
        VERSION_FOLDER = 'ml_model_artifacts_v1'
        preprocessor_fitted = load_object_from_stage(
            session=session,
            stage_name=MODELS_STAGE,
            filename_on_stage='preprocessor.pkl',
            folder_name=VERSION_FOLDER
        )
        
        # Add ROW_NUM for ordering and join back CLOSE_DATE_MONTH after transformation
        window_spec = Window.order_by(col("ACCOUNT_ID"), col('CLOSE_DATE_MONTH'))
        df_with_rownum = df_capped_outliers.with_column("ROW_NUM", row_number().over(window_spec))
        test_df = df_with_rownum.drop('CLOSE_DATE_MONTH')
        
        # Transform test data with preprocessor
        snow_test_df_transformed = preprocessor_fitted.transform(test_df)
        
        test_date_df = df_with_rownum.select('ROW_NUM', 'CLOSE_DATE_MONTH')
        test_final_df = snow_test_df_transformed.join(test_date_df, snow_test_df_transformed['ROW_NUM'] == test_date_df['ROW_NUM'], how='left')
        
        # Drop ROW_NUM columns
        cols_to_drop = [c for c in test_final_df.columns if 'ROW_NUM' in c.replace('"', '')]
        test_final_df_clean = test_final_df.drop(*cols_to_drop)
        
        # Save processed data to table
        output_table = 'ACCNT_CHURN_FUTURE_DATASET_PROCESSED'
        test_final_df_clean.write.save_as_table(table_name=output_table, mode='overwrite')
        
        return "Preprocessing done"
        
    except Exception as e:
        return {"error": str(e)}

$$;

CALL ml_preprocessing('DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.FUTURE_CHURN_PREDICTION_DATASET_FEAT_ENGG');


CREATE OR REPLACE PROCEDURE ml_modelling(TABLENAME VARCHAR, model_name varchar,created_on date)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-ml-python', 'numpy', 'pandas')
HANDLER = 'ml_modelling_handler'
EXECUTE AS OWNER
AS
$$
import os
import pickle
import numpy as np
import pandas as pd
import re
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import StringType, FloatType
from snowflake.ml.registry import Registry
from snowflake.snowpark.functions import current_date
from snowflake.snowpark.functions import lit

try:
    session = get_active_session()
    print("Active session found.")
except Exception as e:
    print(f"No active session found: {e}")


def clean_column_names(df):
    new_names = []
    for col in df.columns:
        col.replace(" ", "_").replace("-", "_").replace("/", "_").replace('"', "").upper()
        new_col = col.split('.')[0]
        #Replace spaces, dashes, slashes with underscores
        new_col = re.sub(r"[ \-\/]", "_", col)
        # Remove decimal points and following digits
        new_col = re.sub(r"\.\d+$", "", new_col)
        # Remove any remaining non-alphanumeric/underscore
        new_col = re.sub(r"[^A-Za-z0-9_]", "", new_col)
        # Collapse multiple underscores
        new_col = re.sub(r"_+", "_", new_col)
        # Remove trailing underscores
        new_col = new_col.rstrip("_")
        # Ensure it starts with a letter or underscore
        if not re.match(r"^[A-Za-z_]", new_col):
            new_col = f"F_{new_col}"
        # Uppercase for consistency
        #new_col = new_col.upper()
        new_names.append(new_col)
    for old, new in zip(df.columns, new_names):
        if old != new:
            df = df.with_column_renamed(old, new)
    return df

    
def ml_modelling_handler(session, tablename,model_name,created_on):
   
    snow_df = session.table(tablename)
    snow_df = clean_column_names(snow_df)
    snow_df= snow_df.withColumn('PREDICTION_DATE',current_date())
    

    # Create a registry to log the model to
    method_options={
      "predict": {
        "case_sensitive": True,
        "enable_monitoring": True,
        "enable_explainability": True
        },
      "predict_proba": {
        "case_sensitive": True,
        "enable_monitoring": True,
        "enable_explainability": True
  }
}
    
    # Create a registry to log the model to
    model_registry = Registry(session=session, 
                              database_name='DEV_DATA_SCIENCE', 
                              schema_name='ACCT_LVL_CHURN_PRED',
                              options={"method_options": method_options})
    m = model_registry.get_model(model_name)
    mv = m.version("TUNED")
    creation_date = created_on
    
    predict_proba_df = mv.run(snow_df, function_name="predict_proba")
    predict_df = mv.run(snow_df, function_name="predict")
    pred1_df =predict_proba_df.select(["ACCOUNT_ID","PREDICTION_DATE","PREDICT_PROBA_0","PREDICT_PROBA_1" ,"PREDICT_PROBA_2"] )
    pred2_df =predict_df.select(["ACCOUNT_ID","PREDICTION"] )
    
    # Join on ACCOUNT_ID
    pred1_pd = pred1_df.to_pandas()
    pred2_pd = pred2_df.to_pandas()
    class_mapping = {
        0: "no_churn",
        1: "partial_churn",
        2: "full_churn"
    }
    
    # Create a new column with the mapped labels
    pred2_pd["PREDICTION_LABEL"] = pred2_pd["PREDICTION"].map(class_mapping)
    # Merge using pandas
    final_df = pred1_pd.merge(pred2_pd, on="ACCOUNT_ID", how="inner")
    df =session.table('FUTURE_CHURN_PREDICTION_DATASET_FEAT_ENGG').to_pandas()
    df= df.drop('CHURN_TYPE', axis=1)
    new_df = df.merge(final_df, on="ACCOUNT_ID", how="inner")
    df1 =session.sql('select distinct ACCOUNT_ID,BEGINNING_ARR,ENDING_ARR,CHURN_ARR ,RENEWAL_ARR from  FUTURE_CHURN_PREDICTION_DATASET where ACCOUNT_ID in (select ACCOUNT_ID from ACCNT_CHURN_FUTURE_DATASET_PROCESSED ) ').to_pandas()
    new_df2 = new_df.merge(df1, on="ACCOUNT_ID", how="inner")
    snowpark_df  = session.create_dataframe(new_df2)
    model_info = f"{model_name}_{creation_date}"

    # Add the new column to your Snowpark DataFrame
    snowpark_df = snowpark_df.with_column("MODEL_NAME_CREATION_DATE", lit(model_info))
    snowpark_df.write.save_as_table(table_name='ACCNT_CHURN_FUTURE_DATASET_SCORED', mode='overwrite')
    return "Success" 

$$
;

call ml_modelling('ACCNT_CHURN_FUTURE_DATASET_PROCESSED', 'TUNED_XGBOOST_MULTICLASS_MODEL_2','2025-07-09');
select * from ACCNT_CHURN_FUTURE_DATASET_SCORED;
-- drop table ACCNT_CHURN_FUTURE_DATASET_SCORED;

CREATE OR REPLACE PROCEDURE local_explainability(TABLENAME VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-ml-python', 'numpy', 'pandas')
HANDLER = 'local_explain'
EXECUTE AS OWNER
AS
$$
import shap
import numpy as np
import pandas as pd
import json
from snowflake.ml.registry import Registry
from datetime import date
today = date.today()
import re
import os
import pickle

try:
    session = get_active_session()
    print("Active session found.")
except Exception as e:
    print(f"No active session found: {e}")

def clean_column_names(df):
    new_names = []
    for col in df.columns:
        col.replace(" ", "_").replace("-", "_").replace("/", "_").replace('"', "").upper()
        new_col = col.split('.')[0]
        #Replace spaces, dashes, slashes with underscores
        new_col = re.sub(r"[ \-\/]", "_", col)
        # Remove decimal points and following digits
        new_col = re.sub(r"\.\d+$", "", new_col)
        # Remove any remaining non-alphanumeric/underscore
        new_col = re.sub(r"[^A-Za-z0-9_]", "", new_col)
        # Collapse multiple underscores
        new_col = re.sub(r"_+", "_", new_col)
        # Remove trailing underscores
        new_col = new_col.rstrip("_")
        # Ensure it starts with a letter or underscore
        if not re.match(r"^[A-Za-z_]", new_col):
            new_col = f"F_{new_col}"
        # Uppercase for consistency
        #new_col = new_col.upper()
        new_names.append(new_col)
    for old, new in zip(df.columns, new_names):
        if old != new:
            df = df.with_column_renamed(old, new)
    return df
    
# Load preprocessor object from stage
def load_object_from_stage(session, stage_name, filename_on_stage, folder_name):
    local_temp_path = '/tmp/'
    full_stage_path = f"{stage_name.strip('/')}/{folder_name}/{filename_on_stage}"
    session.file.get(full_stage_path, local_temp_path)
    downloaded_file_path = os.path.join(local_temp_path, filename_on_stage)
    with open(downloaded_file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    os.remove(downloaded_file_path)
    return loaded_object
    
def local_explain(session, tablename):
   
    snow_df = session.table(tablename)
    snow_df = clean_column_names(snow_df)
    
    today_str = today.strftime("%Y-%m-%d")
    score_df = session.table('ACCNT_CHURN_FUTURE_DATASET_SCORED').to_pandas()
    preds = score_df["PREDICTION"].values
    
    features =[]
    for cols in snow_df.columns:
        if cols not in ["ACCOUNT_ID","ACCOUNT_NAME","CHURN_TYPE","BLADES_PURCHASED","CLOSE_DATE_MONTH", "STANDARDIZED_COUNTRY","CHURN_TYPE_ENCODED"]:
            features.append(cols)
    # Get mean absolute SHAP values for each feature
    X_train_pd = snow_df.to_pandas()[features]

    MODELS_STAGE = '@MODELS_STAGE'  # Replace with your actual stage name
    VERSION_FOLDER = 'ml_model_artifacts_v1'
    model = load_object_from_stage(
        session=session,
        stage_name=MODELS_STAGE,
        filename_on_stage='best_xgb_for_local_exp.pkl',
        folder_name=VERSION_FOLDER
    )
    
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_train_pd.to_numpy())

    top10_shap_json = []
    
    n_samples = X_train_pd.shape[0]
    n_classes = len(shap_values)
    n_features = X_train_pd.shape[1]
    
    for i in range(n_samples):
        pred_class = int(preds[i])  # 0, 1, or 2
        shap_row =  shap_values[i, :, pred_class]   # shape: (n_features,)
        # Get indices of top 10 absolute SHAP values
        top_idx = np.argsort(np.abs(shap_row))[::-1][:10]
        # Build dict of feature: shap_value
        top_features = {features[j]: float(shap_row[j]) for j in top_idx}
        # Store as JSON string
        top10_shap_json.append(json.dumps(top_features))
    score_df["TOP10_SHAP_JSON"] = top10_shap_json
    df =session.create_dataframe(score_df)
    df.write.save_as_table(table_name='ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY', mode='append')

    return "Local explainability saved"
$$
;

call local_explainability('ACCNT_CHURN_FUTURE_DATASET_PROCESSED');

select * from ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY;
-- drop table ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY;

CREATE OR REPLACE PROCEDURE what_if_analysis_SP(TABLENAME VARCHAR, model_name varchar,created_on date)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('snowflake-ml-python', 'numpy', 'pandas')
HANDLER = 'local_explain'
EXECUTE AS OWNER
AS
$$
import shap
import numpy as np
import pandas as pd
import json
from snowflake.ml.registry import Registry
from datetime import date
today = date.today()
import re
import os
import pickle
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import lit, map_from_arrays, get, col, array_construct, coalesce, when, row_number
from snowflake.snowpark.window import Window
from snowflake.snowpark.types import StringType, FloatType
from snowflake.snowpark.window import Window
from snowflake.ml.registry import Registry
from snowflake.snowpark.functions import current_date

MODELS_STAGE = '@MODELS_STAGE'  # Replace with your actual stage name
VERSION_FOLDER = 'ml_model_artifacts_v1'
    
def apply_industry_grouping(df, industry_column='INDUSTRY', mapping_dict=None, unmapped_category='Other/Unspecified'):
            if mapping_dict is None:
                raise ValueError("mapping_dict must be provided.")
            case_expr = None
            for broader_cat, specific_cats in mapping_dict.items():
                for specific_cat in specific_cats:
                    condition = F.col(industry_column) == F.lit(specific_cat)
                    if case_expr is None:
                        case_expr = F.when(condition, F.lit(broader_cat))
                    else:
                        case_expr = case_expr.when(condition, F.lit(broader_cat))
            case_expr = case_expr.otherwise(F.lit(unmapped_category))
            df = df.with_column('BROADER_INDUSTRY', case_expr)
            df = df.drop(industry_column)
            return df
            
# Helper function to cap outliers
def cap_outliers_snowpark(df, col_name):
    Q1 = df.select(F.expr(f"percentile_cont(0.25) within group (order by {col_name})")).collect()[0][0]
    Q3 = df.select(F.expr(f"percentile_cont(0.75) within group (order by {col_name})")).collect()[0][0]
    IQR = float(Q3) - float(Q1)
    lower_bound = float(Q1) - 1.5 * IQR
    upper_bound = float(Q3) + 1.5 * IQR
    capped_col = F.when(F.col(col_name) > upper_bound, upper_bound)\
                  .when(F.col(col_name) < lower_bound, lower_bound)\
                  .otherwise(F.col(col_name))
    return df.with_column(col_name, capped_col)

# Load preprocessor object from stage
def load_object_from_stage(session, stage_name, filename_on_stage, folder_name):
    local_temp_path = '/tmp/'
    full_stage_path = f"{stage_name.strip('/')}/{folder_name}/{filename_on_stage}"
    session.file.get(full_stage_path, local_temp_path)
    downloaded_file_path = os.path.join(local_temp_path, filename_on_stage)
    with open(downloaded_file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    os.remove(downloaded_file_path)
    return loaded_object
    
def clean_column_names(df):
    new_names = []
    for col in df.columns:
        col.replace(" ", "_").replace("-", "_").replace("/", "_").replace('"', "").upper()
        new_col = col.split('.')[0]
        #Replace spaces, dashes, slashes with underscores
        new_col = re.sub(r"[ \-\/]", "_", col)
        # Remove decimal points and following digits
        new_col = re.sub(r"\.\d+$", "", new_col)
        # Remove any remaining non-alphanumeric/underscore
        new_col = re.sub(r"[^A-Za-z0-9_]", "", new_col)
        # Collapse multiple underscores
        new_col = re.sub(r"_+", "_", new_col)
        # Remove trailing underscores
        new_col = new_col.rstrip("_")
        # Ensure it starts with a letter or underscore
        if not re.match(r"^[A-Za-z_]", new_col):
            new_col = f"F_{new_col}"
        # Uppercase for consistency
        #new_col = new_col.upper()
        new_names.append(new_col)
    for old, new in zip(df.columns, new_names):
        if old != new:
            df = df.with_column_renamed(old, new)
    return df
    
def ml_preprocessing_handler(session, table_name, return_table_name):

    #try:
    # Load data from table 
    snow_df = session.table(table_name)
    
    # Cast boolean columns to string (if exist)
    boolean_columns = [
        'FINANCE_SEGMENT', 'HAS_CSM', 'ENT_', 'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_1_MONTHS',
        'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_3_MONTHS', 'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_6_MONTHS',
        'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS', 'HAS_PSP', 'HAD_EBR',
        'PRIORITY_ACCOUNT_GC', 'PLG_SOURCED', 'FORMERLY_ON_PLUS', 'FORMERLY_ON_SCHOLARSHIP'
    ]
    for col_name in boolean_columns:
        if col_name in snow_df.columns:
            snow_df = snow_df.with_column(col_name, snow_df[col_name].cast(StringType()))
    
    # Define target variable and drop rows with missing target
    TARGET_VARIABLE = 'CHURN_TYPE'
    snow_df = snow_df.dropna(subset=[TARGET_VARIABLE])
    
    # Encode target variable
    target_mapping = {'no_churn': 0, 'partial_churn': 1, 'full_churn': 2}
    ENCODED_TARGET_VARIABLE = 'CHURN_TYPE_ENCODED'
    case_expr = None
    for category, encoded_value in target_mapping.items():
        if case_expr is None:
            case_expr = when(col(TARGET_VARIABLE) == category, lit(encoded_value))
        else:
            case_expr = case_expr.when(col(TARGET_VARIABLE) == category, lit(encoded_value))
    snow_df = snow_df.with_column(ENCODED_TARGET_VARIABLE, case_expr)
    
    # Drop unwanted columns
    cols_to_drop = [
        'AVG_NPSSCORE_PAST_1_MONTHS', 'AVG_NPSSCORE_PAST_3_MONTHS', 'AVG_NPSSCORE_PAST_6_MONTHS',
        'AVG_NPSSCORE_PAST_9_MONTHS', 'AVG_CSATSCORE_PAST_1_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_1_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_6_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_CLICK_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_1_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_6_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_UNSUBSCRIBE_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_1_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_6_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_1_MONTHS',
        'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_6_MONTHS',
        'AVG_MONTHLY_TIME_TO_CLICK_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_1_MONTHS',
        'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_3_MONTHS', 'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_6_MONTHS',
        'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_9_MONTHS', 'AVG_MONTHLY_LESSONS_COUNT_PAST_1_MONTHS',
        'AVG_MONTHLY_LESSONS_COUNT_PAST_3_MONTHS', 'AVG_MONTHLY_LESSON_COMPLETION_RATIO_PAST_1_MONTHS', 'AVG_MONTHLY_LESSON_COMPLETION_RATIO_PAST_3_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_LAST_ACTIVITY_PAST_1_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_LAST_ACTIVITY_PAST_3_MONTHS', 'AVG_MONTHLY_DAYS_SINCE_ENROLLED_PAST_1_MONTHS',
        'AVG_MONTHLY_DAYS_SINCE_ENROLLED_PAST_3_MONTHS', 'AVG_MONTHLY_TICKETS_OPEN_SINCE_90DAYS_PAST_1_MONTHS', 'AVG_MONTHLY_TICKETS_OPEN_SINCE_90DAYS_PAST_3_MONTHS',
        'HAS_PSP', 'FORMERLY_ON_SCHOLARSHIP'
    ]
    snow_df = snow_df.drop(*cols_to_drop)
    
    # Prepare feature columns and data types
    feature_cols = [c for c in snow_df.columns]
    train_dtypes = dict(snow_df.dtypes)
    numerical_features = []
    categorical_features = []
    for c in feature_cols:
        dtype_str = train_dtypes.get(c, '').lower()
        if any(x in dtype_str for x in ['int', 'float', 'double', 'decimal', 'numeric']):
            numerical_features.append(c)
        elif any(x in dtype_str for x in ['string', 'varchar', 'char', 'boolean', 'bool']):
            categorical_features.append(c)
    
    # Standardize country names using mapping
    COUNTRY_STANDARDIZATION_MAP = {
        # (same mapping as provided in user code)
        "Antigua and Barbuda": "Antigua and Barbuda", "Argentina": "Argentina", "Armenia": "Armenia",
        "Australia": "Australia", "Austria": "Austria", "Azerbaijan": "Azerbaijan", "Bahrain": "Bahrain",
        "Belarus": "Belarus", "Belgium": "Belgium", "Brazil": "Brazil", "Bulgaria": "Bulgaria",
        "Canada": "Canada", "Cayman Islands": "Cayman Islands", "Chile": "Chile", "China": "China",
        "Colombia": "Colombia", "Croatia": "Croatia", "Cyprus": "Cyprus", "Czech Republic": "Czech Republic",
        "Denmark": "Denmark", "Dominican Republic": "Dominican Republic", "Ecuador": "Ecuador",
        "Egypt": "Egypt", "El Salvador": "El Salvador", "Estonia": "Estonia", "Finland": "Finland",
        "France": "France", "Georgia": "Georgia", "Germany": "Germany", "Gibraltar": "Gibraltar",
        "Greece": "Greece", "Guatemala": "Guatemala", "Guernsey": "Guernsey", "Hong Kong": "Hong Kong",
        "Hungary": "Hungary", "Iceland": "Iceland", "India": "India", "Indonesia": "Indonesia",
        "Iraq": "Iraq", "Ireland": "Ireland", "Israel": "Israel", "Italy": "Italy",
        "Jamaica": "Jamaica", "Japan": "Japan", "Kazakhstan": "Kazakhstan", "Kenya": "Kenya",
        "Korea": "Korea, Republic of", "Korea, Democratic People's Republic of": "Korea, Democratic People's Republic of",
        "Korea, Republic of": "Korea, Republic of", "Kuwait": "Kuwait", "Kyrgyzstan": "Kyrgyzstan",
        "Latvia": "Latvia", "Lebanon": "Lebanon", "Lithuania": "Lithuania", "Luxembourg": "Luxembourg",
        "Malaysia": "Malaysia", "Malta": "Malta", "Mauritius": "Mauritius", "Mexico": "Mexico",
        "Moldova, Republic of": "Moldova, Republic of", "Morocco": "Morocco", "Netherlands": "Netherlands",
        "New Zealand": "New Zealand", "Nigeria": "Nigeria", "Norway": "Norway", "Pakistan": "Pakistan",
        "Peru": "Peru", "Philippines": "Philippines", "Poland": "Poland", "Portugal": "Portugal",
        "Qatar": "Qatar", "Romania": "Romania", "Russia": "Russian Federation",
        "Russian Federation": "Russian Federation", "Saudi Arabia": "Saudi Arabia", "Senegal": "Senegal",
        "Serbia": "Serbia", "Seychelles": "Seychelles", "Singapore": "Singapore", "Slovenia": "Slovenia",
        "South Africa": "South Africa", "South Korea": "Korea, Republic of", "Spain": "Spain",
        "Sweden": "Sweden", "Switzerland": "Switzerland", "Taiwan": "Taiwan",
        "Taiwan, Province of China": "Taiwan", "Thailand": "Thailand", "Turkey": "Turkey",
        "Ukraine": "Ukraine", "United Arab Emirates": "United Arab Emirates",
        "united kingdom": "United Kingdom", "United Kingdom": "United Kingdom",
        "United States": "United States", "Uruguay": "Uruguay", "US": "United States",
        "USA": "United States", "Uzbekistan": "Uzbekistan",
        "Venezuela, Bolivarian Republic of": "Venezuela, Bolivarian Republic of", "Viet Nam": "Vietnam",
        "Vietnam": "Vietnam", "Virgin Islands, British": "Virgin Islands, British"
    }
    map_keys = array_construct(*[lit(k) for k in COUNTRY_STANDARDIZATION_MAP.keys()])
    map_values = array_construct(*[lit(v) for v in COUNTRY_STANDARDIZATION_MAP.values()])
    country_map = map_from_arrays(map_keys, map_values)
    df_standardized = snow_df.with_column(
        "STANDARDIZED_COUNTRY",
        coalesce(get(country_map, col("HQ_COUNTRY_GC")), col("HQ_COUNTRY_GC"))
    ).drop("HQ_COUNTRY_GC")
    
    # Industry grouping function
    BROADER_INDUSTRY_CATEGORIES = {
        # (same dictionary as provided by user)
        "Technology & Software": [
            "Lending & Brokerage", "Consumer Tech", "Business Intelligence (BI) Software",
            "Security Software", "Networking Software", "Financial Software", "Engineering Software",
            "Mobile App Development", "Social Networks", "Content & Collaboration Software",
            "Internet Service Providers, Website Hosting & Internet-related Services",
            "Customer Relationship Management (CRM) Software", "Storage & System Management Software",
            "Telephony & Wireless", "Custom Software & IT Services", "Internet",
            "Database & File Management Software", "Multimedia, Games & Graphics Software",
            "E-Commerce and Marketplaces", "Data Collection & Internet Portals",
            "Information & Document Management", "Computer Equipment & Peripherals",
            "Technology Hardware, Storage & Periph...", "Technology", "Security Products & Services",
            "Legal Software", "Data Processing & Outsourced Services", "Graphic Design",
            "Biotechnology", "Photography Studio", "Multimedia & Graphic Design"
        ],
        "Financial Services & Insurance": [
            "Credit Cards & Transaction Processing", "Financial Services", "Banking",
            "Investment Banking", "Banking & Mortgages", "Professional Services", "Insurance",
            "Diversified Capital Markets", "Payments", "Accounting Services", "Diversified Financial Services",
            "Capital Markets", "Accounting", "Asset Management & Custody Banks", "Banks",
            "Venture Capital & Private Equity"
        ],
        # ... (include all other categories as per user code)
        "Healthcare & Medical": [
            "Healthcare Software",
            "Medical & Surgical Hospitals",
            "Medical Devices & Equipment",
            "Medical Specialists",
            "Mental Health & Rehabilitation Facilities",
            "Pharmaceuticals",
            "Healthcare",
            "Health Care Providers & Services",
            "Dental Offices",
            "Health Care Services",
            "Physicians Clinics",
            "Medical Laboratories & Imaging Centers",
            "Vitamins, Supplements & Health Stores",
            "Elderly Care Services",
            "Health & Nutrition Products",
            "Health Care",
            "Hospital & Health Care",
            "Health & Wellness"
        ],
        "Retail & Consumer Goods": [
            "Grocery Retail",
            "Drug Stores & Pharmacies",
            "Apparel & Accessories Retail",
            "Plastic, Packaging & Containers", # For consumer goods packaging
            "Toys & Games",
            "Textiles & Apparel",
            "Electronics",
            "Apparel, Accessories & Luxury Goods",
            "Flowers, Gifts & Specialty Stores",
            "Consumer Electronics & Computers Retail",
            "Department Stores, Shopping Centers & Superstores",
            "Furniture",
            "Jewelry & Watch Retail",
            "Retailing",
            "Specialty Retail",
            "Watches & Jewelry",
            "Consumer Staples",
            "Consumer Discretionary",
            "Household Goods",
            "Cleaning Products",
            "Packaged Foods & Meats", # If sold via retail
            "Pet Products",
            "Sporting Goods",
            "Sporting & Recreational Equipment Retail",
            "Home Improvement & Hardware Retail",
            "Record, Video & Book Stores",
            "Office Products Retail & Distribution",
            "Convenience Stores, Gas Stations & Liquor Stores",
            "Appliances",
            "Automobile Parts Stores",
            "Household Durables",
            "Eyewear",
            "Home Improvement Retail",
            "Consumer Goods" # Broad category
        ],
        "Media, Entertainment & Hospitality": [
            "Publishing",
            "Social Networks", # Content/entertainment focus
            "Restaurants",
            "Fitness & Dance Facilities",
            "Music Production & Services",
            "Broadcasting",
            "Data Collection & Internet Portals", # If content aggregation/news portals
            "Gambling & Gaming",
            "Media",
            "Newspapers & News Services",
            "Photography Studio", # If primarily for entertainment/events
            "Sports Teams & Leagues",
            "Movies & Entertainment",
            "Lodging & Resorts",
            "Amusement Parks, Arcades & Attractions",
            "Hotels, Restaurants & Leisure",
            "Performing Arts Theaters",
            "Casinos & Gaming",
            "Leisure Products"
        ],
        "Professional & Business Services": [
            "Other Rental Stores (Furniture, A/V, Construction & Industrial Equipment)",
            "Training",
            "Human Resources Software", # If referring to HR services platforms
            "HR & Staffing",
            "Repair Services", # General repair services
            "Management Consulting",
            "Commercial Printing",
            "Professional Services",
            "Legal Services",
            "Automotive Service & Collision Repair",
            "Advertising & Marketing",
            "Human Resource & Employment Services",
            "B2B", # Broad B2B services
            "Consulting",
            "Research & Development",
            "Research & Consulting Services",
            "Diversified Consumer Services", # If consumer-facing services beyond retail/healthcare
            "Call Centers & Business Centers",
            "Family Services",
            "Translation & Linguistic Services",
            "Cleaning Services",
            "Landscape Services",
            "Photographic & Optical Equipment", # If service based
            "Travel and Hospitality", # If related to professional event planning etc.
            "Consumer Services" # Broad consumer services
        ],
        "Education & Non-Profit": [
            "Non-Profit & Charitable Organizations",
            "Education Services",
            "Colleges & Universities",
            "Religious Organizations",
            "K-12 Schools",
            "Education",
            "Cultural & Informational Centers",
            "Fundraising",
            "Childcare"
        ],
        "Industrial, Manufacturing & Utilities": [
            "Industrials & Manufacturing",
            "Plastic, Packaging & Containers", # Manufacturing of containers
            "Industrial Machinery & Equipment",
            "Electricity, Oil & Gas",
            "Telecommunication Equipment",
            "Automotive", # Manufacturing focus
            "Motor Vehicles",
            "Electrical Equipment",
            "Chemicals & Related Products",
            "Aerospace & Defense",
            "Boats & Submarines",
            "Test & Measurement Equipment",
            "Distributors", # Industrial distributors
            "Household Durables", # Manufacturing focus
            "Manufacturing" # Assuming this is meant to be a general manufacturing category
        ],
        "Transportation & Logistics": [
            "Trucking, Moving & Storage",
            "Freight & Logistics Services",
            "Travel Agencies & Services",
            "Rail, Bus & Taxi",
            "Airlines, Airports & Air Services",
            "Transportation",
            "Car & Truck Rental",
            "Wireless Telecommunication Services", # Communication infrastructure for transport/logistics
            "Diversified Telecommunication Services", # Could involve logistics network
            "Shipping & Logistics",
            "Ground Transportation"
        ],
        "Agriculture & Food Production": [
            "Food & Beverage", # Production focus
            "Crops",
            "Animals & Livestock",
            "Food Service", # Production/wholesale
            "Food Products",
            "Forestry",
            "Agricultural Products",
            "al Products" # Assuming this is a typo and should be Agricultural Products
        ],
        "Real Estate & Construction": [
            "Architecture",
            "Commercial & Residential Construction",
            "Construction & Engineering",
            "Building Materials",
            "Real Estate",
            "Architecture, Engineering & Design",
            "Civil Engineering Construction"
        ],
        "Government & Public Sector": [
            "Federal" # Specific government category
        ],
        "Environmental Services": [
            "Waste Treatment, Environmental Services & Recycling"
        ],
        "Other/Miscellaneous": [ # For any remaining or ambiguous items
            "Other" # Explicitly listed as 'Other'
        ]
    }
    

    df_capped = apply_industry_grouping(df_standardized, 'INDUSTRY', BROADER_INDUSTRY_CATEGORIES, 'Other/Unspecified')
    
    # Cast decimal columns to float in train and test dfs (assuming snow_train_df and snow_test_df exist)
    snow_train_df = df_capped
    
    for col_name in numerical_features:
        col_dtype = dict(snow_train_df.dtypes).get(col_name, '').lower()
        if col_dtype.startswith('decimal'):
            snow_train_df = snow_train_df.with_column(col_name, F.col(col_name).cast(FloatType()))
    
    # Remove some columns from categorical_features
    for col_to_remove in ['ACCOUNT_ID', 'ACCOUNT_NAME','CHURN_TYPE', 'BLADES_PURCHASED', 'STANDARDIZED_COUNTRY']:
        if col_to_remove in categorical_features:
            categorical_features.remove(col_to_remove)
    if 'CHURN_TYPE_ENCODED' in numerical_features:
        numerical_features.remove('CHURN_TYPE_ENCODED')
    
    
    # Cap outliers for numerical features
    df_capped_outliers = snow_train_df
    #for col_name in numerical_features:
    #    df_capped_outliers = cap_outliers_snowpark(df_capped_outliers, col_name)
    
    preprocessor_fitted = load_object_from_stage(
        session=session,
        stage_name=MODELS_STAGE,
        filename_on_stage='preprocessor.pkl',
        folder_name=VERSION_FOLDER
    )
    
    # Add ROW_NUM for ordering and join back CLOSE_DATE_MONTH after transformation
    window_spec = Window.order_by(col("ACCOUNT_ID"), col('CLOSE_DATE_MONTH'))
    df_with_rownum = df_capped_outliers.with_column("ROW_NUM", row_number().over(window_spec))
    test_df = df_with_rownum.drop('CLOSE_DATE_MONTH')
    
    # Transform test data with preprocessor
    snow_test_df_transformed = preprocessor_fitted.transform(test_df)
    
    test_date_df = df_with_rownum.select('ROW_NUM', 'CLOSE_DATE_MONTH')
    test_final_df = snow_test_df_transformed.join(test_date_df, snow_test_df_transformed['ROW_NUM'] == test_date_df['ROW_NUM'], how='left')
    
    # Drop ROW_NUM columns
    cols_to_drop = [c for c in test_final_df.columns if 'ROW_NUM' in c.replace('"', '')]
    test_final_df_clean = test_final_df.drop(*cols_to_drop)
    
    test_final_df_clean.write.save_as_table( return_table_name , mode='overwrite')
    return return_table_name
        
    #except Exception as e:
    #return "Error while doing preprocessing for the selected features" str(e)
        
def ml_modelling_handler(session, return_table_name, return_model_table_name, model_name,created_on):
   
    #snow_df = pd.DataFrame([clean_df])
    df = session.table(return_table_name)
    snow_df = clean_column_names(df)
    
    
    #try: 

    # Create a registry to log the model to
    method_options={
      "predict": {
        "case_sensitive": True,
        "enable_monitoring": True,
        "enable_explainability": True
        },
      "predict_proba": {
        "case_sensitive": True,
        "enable_monitoring": True,
        "enable_explainability": True
  }
}
    
    # Create a registry to log the model to
    model_registry = Registry(session=session, 
                              database_name='DEV_DATA_SCIENCE', 
                              schema_name='ACCT_LVL_CHURN_PRED',
                              options={"method_options": method_options})
    m = model_registry.get_model(model_name)
    mv = m.version("TUNED")
    creation_date = created_on
    
    predict_proba_df = mv.run(snow_df, function_name="predict_proba")
    predict_df = mv.run(snow_df, function_name="predict")
    pred1_df =predict_proba_df.select(["ACCOUNT_ID","PREDICT_PROBA_0","PREDICT_PROBA_1" ,"PREDICT_PROBA_2"] )
    pred2_df =predict_df.select(["ACCOUNT_ID","PREDICTION"] )
    
    # Join on ACCOUNT_ID
    pred1_pd = pred1_df.to_pandas()
    pred2_pd = pred2_df.to_pandas()
    class_mapping = {
        0: "no_churn",
        1: "partial_churn",
        2: "full_churn"
    }
    
    # Create a new column with the mapped labels
    pred2_pd["PREDICTION_LABEL"] = pred2_pd["PREDICTION"].map(class_mapping)
    # Merge using pandas
    final_df = pred1_pd.merge(pred2_pd, on="ACCOUNT_ID", how="inner")
    
    snowpark_df  = session.create_dataframe(final_df)
    
    snowpark_df.write.save_as_table(return_model_table_name , mode='overwrite')
    return "Modelling success"
        
    #except Exception as e:
        #return "Error while doing modelling for the selected features": str(e)
        
def local_explain(session, table_name , model_name,created_on):
    return_table_name = 'DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.WHAT_IF_TABLE'
    return_model_table_name = 'DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.WHAT_IF_TABLE_PREDICTION'
    
    ml_preprocessing_handler(session, table_name,return_table_name)
    
    ml_modelling_handler(session, return_table_name,return_model_table_name, model_name,created_on)
    
    try:
        #snow_df = pd.DataFrame([clean_df])
        clean_df= session.table(return_table_name)
        snow_df = clean_column_names(clean_df)
        
        today_str = today.strftime("%Y-%m-%d")
        model_df =session.table(return_model_table_name)
        #score_df = pd.DataFrame([model_df])
        score_df = model_df.to_pandas()
        preds = score_df["PREDICTION"].values
        
        features =[]
        for cols in snow_df.columns:
            if cols not in ["ACCOUNT_ID","ACCOUNT_NAME","CHURN_TYPE","BLADES_PURCHASED","CLOSE_DATE_MONTH", "STANDARDIZED_COUNTRY","CHURN_TYPE_ENCODED"]:
                features.append(cols)
        # Get mean absolute SHAP values for each feature
        X_train_pd = snow_df.to_pandas()[features]
    
        
        model = load_object_from_stage(
            session=session,
            stage_name=MODELS_STAGE,
            filename_on_stage='best_xgb_for_local_exp.pkl',
            folder_name=VERSION_FOLDER
        )
        
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_train_pd.to_numpy())
    
        top10_shap_json = []
        
        n_samples = X_train_pd.shape[0]
        n_classes = len(shap_values)
        n_features = X_train_pd.shape[1]
        
        for i in range(n_samples):
            pred_class = int(preds[i])  # 0, 1, or 2
            shap_row =  shap_values[i, :, pred_class]   # shape: (n_features,)
            # Get indices of top 10 absolute SHAP values
            top_idx = np.argsort(np.abs(shap_row))[::-1][:10]
            # Build dict of feature: shap_value
            top_features = {features[j]: float(shap_row[j]) for j in top_idx}
            # Store as JSON string
            top10_shap_json.append(json.dumps(top_features))
        score_df["TOP10_SHAP_JSON"] = top10_shap_json
        df =session.create_dataframe(score_df)
        df.write.save_as_table(return_model_table_name , mode='overwrite')
        return "Success"
    except Exception as e:
        return "Error while doing prediction for the selected features"
$$
;

create or replace view test as select * from FUTURE_CHURN_PREDICTION_DATASET_FEAT_ENGG limit 1;
call  what_if_analysis_SP('test' ,'TUNED_XGBOOST_MULTICLASS_MODEL_2','2025-07-09');
select * from DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.WHAT_IF_TABLE_PREDICTION;


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import json

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Churn Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- Custom CSS Styling ---
st.markdown("""
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.main .block-container, .block-container { padding: 2rem 5rem 2rem 5rem; }
.main-title { font-size: 2.2rem; font-weight: 700; color: #343a40; padding-top: 0.5rem; }
.info-card, .metric-card, .compact-metric-card {
    background: #fff; border: 1px solid #E0E0E0; border-radius: 8px;
    text-align: center;
}
.info-card { padding: 1.25rem; margin-bottom: 2rem; }
.info-card-title { color: #007bff; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem; text-transform: uppercase; }
.info-card-value { font-size: 2rem; font-weight: 600; color: #343a40; margin: 0; }
.metric-card { padding: 1.5rem; text-align: left; margin-bottom: 1rem; }
.metric-card h3 { color: #6c757d; font-size: 1rem; font-weight: 400; margin-bottom: 0.5rem; }
.metric-card p { color: #343a40; font-size: 1.25rem; font-weight: 600; margin: 0; }
.detail-row { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EFF2F6; padding: 0.75rem 0.25rem; }
.compact-metric-card { padding: 1rem; text-align: left; margin-bottom: 1rem; }
.compact-metric-title { color: #6c757d; font-size: 0.85rem; font-weight: 400; margin-bottom: 0.25rem; }
.compact-metric-value { color: #343a40; font-size: 1.1rem; font-weight: 600; margin: 0; }
.stPlotlyChart { border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Snowflake Loader ---
@st.cache_data
def load_snowflake_data():
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    # session.use_role('DEV_DS_ACCT_LVL_CHURN_PRED_RW_ROLE')
    snow_df = session.table('DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY')
    df = snow_df.to_pandas()
    return df

df = load_snowflake_data()

# --- UI (preserve all card/column structure) ---
title_card, filter_card1 = st.columns([2, 1])#, 2])

with title_card:
    st.subheader("Churn Insights - Single Account")
    
with filter_card1:
    st.markdown('<p class="info-card-title">Select Account</p>', unsafe_allow_html=True)
    selected_account = st.selectbox(
        "Account Name",
        df['ACCOUNT_NAME'].unique(),
        index=0,
        label_visibility="collapsed"
    )
# with filter_card2:
#     st.markdown('<p class="info-card-title">Date</p>', unsafe_allow_html=True)
#     st.date_input("Date of Inference", datetime.now(), label_visibility="collapsed")

st.markdown("----")


# # Trend chart for predicted churn type (categorical) for the selected account
# account_trend_df = df[df['ACCOUNT_NAME'] == selected_account].copy()
# if 'CLOSE_DATE_MONTH' in account_trend_df.columns and 'PREDICTION_LABEL' in account_trend_df.columns:
#     account_trend_df = account_trend_df.sort_values('CLOSE_DATE_MONTH')
#     label_map = {
#         "no churn": "No Churn",
#         "partial churn": "Partial Churn",
#         "full churn": "Full Churn"
#     }
#     color_map = {
#         "No Churn": "#34D399",      # Green
#         "Partial Churn": "#FBBF24", # Amber
#         "Full Churn": "#F87171"     # Red
#     }
#     account_trend_df['Churn Type'] = account_trend_df['PREDICTION_LABEL'].str.lower().map(label_map).fillna(account_trend_df['PREDICTION_LABEL'])
#     st.markdown("### Predicted Churn Type Trend")
#     fig_type = px.scatter(
#         account_trend_df,
#         x='CLOSE_DATE_MONTH',
#         y='Churn Type',
#         color='Churn Type',
#         color_discrete_map=color_map,
#         size_max=18,
#         title="Predicted Churn Type Over Time",
#         labels={'CLOSE_DATE_MONTH': 'Month', 'Churn Type': 'Predicted Churn Type'},
#     )
#     fig_type.update_traces(marker=dict(size=18, symbol='circle'))
#     fig_type.update_layout(yaxis=dict(categoryorder='array', categoryarray=["No Churn", "Partial Churn", "Full Churn"]))
#     st.plotly_chart(fig_type, use_container_width=True)
# else:
#     st.info("No monthly churn type prediction data available for this account.")

account_data = df[df['ACCOUNT_NAME'] == selected_account].iloc[0]

# Churn type label mapping
churn_label_map = {
    "no_churn": "No Churn",
    "partial_churn": "Partial Churn",
    "full_churn": "Full Churn"
}
# Get churn type value and map it
churn_type_raw = str(account_data['PREDICTION_LABEL']).strip().lower()
churn_type_display = churn_label_map.get(churn_type_raw, str(account_data['PREDICTION_LABEL']))

main_card_col1, main_card_col2 = st.columns(2)
with main_card_col1:
    st.markdown(f"""
    <div class="info-card">
        <p class="info-card-title">Account Name</p>
        <p class="info-card-value">{account_data['ACCOUNT_NAME']}</p>
    </div>
    """, unsafe_allow_html=True)
    
with main_card_col2:
    st.markdown(f"""
    <div class="info-card">
        <p class="info-card-title">Predicted Churn Probability</p>
        <p class="info-card-value">{account_data['PREDICT_PROBA_2']*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # Format ARR values with $ and two decimals
    def format_currency(val):
        try:
            return f"$ {float(val):,.2f}"
        except Exception:
            return val

    metrics = [
        ("Account Type", account_data['GTM_SEGMENT_C']),
        # Calculate "Account Since" using TENURE_MONTHS
        ("Account Since",
            (
                (datetime.strptime(str(account_data['PREDICTION_DATE']), "%Y-%m-%d") 
                 if not pd.isnull(account_data['PREDICTION_DATE']) else datetime.now()
                ) - pd.DateOffset(months=int(account_data['TENURE_MONTHS']))
            ).strftime("%Y-%m-%d") if not pd.isnull(account_data['TENURE_MONTHS']) else ""
        ),
        ("Number of Blades", len(account_data['BLADES_PURCHASED'].split(','))),
        ("Number of Users", account_data['ACCOUNT_TOTAL_EMPLOYEE_COUNT']),
        ("Churn Type", churn_type_display),
        ("Last Renewal Date", account_data['CLOSE_DATE_MONTH']),
        ("Current ARR", format_currency(account_data['BEGINNING_ARR'])),
        ("Predicted Churn ARR", format_currency(abs(account_data['BEGINNING_ARR'] * account_data['PREDICT_PROBA_2']))),
        # ("Prediction Date", account_data['PREDICTION_DATE']),
    ]
    for title, value in metrics:
        st.markdown(f"""
        <div class="detail-row">
            <span class="detail-title" style='text-align: center; color: blue;'>{title}</span>
            <span class="detail-value" style='text-align: center; color: blue;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown("<h5>Top 10 Most Influential Churn Drivers</h5>", unsafe_allow_html=True)
    # Use the 'TOP10_SHAP_JSON' column if present
    if 'TOP10_SHAP_JSON' in account_data and pd.notnull(account_data['TOP10_SHAP_JSON']):
        try:
            feature_importance_dict = json.loads(account_data['TOP10_SHAP_JSON'])
            # Sort by absolute importance descending and take top 10
            sorted_items = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            feature_names = [k for k, v in sorted_items]
            importances = [v for k, v in sorted_items]
        except Exception:
            feature_names = []
            importances = []
    else:
        # Simulate for demo
        st.write('Simulated Feature Importance')
        candidate_cols = [
            'CPM', 'ACNT_RENEW_TIMES', 'AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_1_MONTHS',
            'TENURE_MONTHS', 'AVG_MONTHLY_OVERALL_SCORE_1_MONTH', 'AVG_PROJECTED_VOLUME_PAST_1_MONTHS',
            'AVG_NPSSCORE_PAST_1_MONTHS', 'AVG_CSATSCORE_PAST_1_MONTHS', 'AVG_TICKETS_PAST_1_MONTHS',
            'AVG_MAU_PAST_1_MONTH', 'AVG_POWER_USERS_PAST_1_MONTH'
        ]
        np.random.seed(hash(selected_account) % 2**32)
        chosen = np.random.choice(candidate_cols, size=min(10, len(candidate_cols)), replace=False)
        importances = np.random.uniform(10, 100, len(chosen))
        feature_names = list(chosen)
        importances = list(importances)
    drivers_df = pd.DataFrame({'Driver': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    fig = px.bar(drivers_df, x="Importance", y="Driver", orientation="h", color="Importance", color_continuous_scale="Viridis")
    fig.update_layout(
        yaxis={'categoryorder':'total ascending', 'title': None},  # Remove y axis label
        xaxis=dict(showticklabels=True, showgrid=False, zeroline=False),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True) 
	
	multiple page --
	import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(
    page_title="Churn Insights - Multiple Accounts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Callback function to reset all filters ---
def reset_all_filters():
    """Sets all filter values in the session state back to 'All'."""
    keys_to_reset = [
        'country_filter', 'region_filter', 'multiproduct_filter',
        'gtm_filter', 'plan_type_filter', 'churn_type_filter', 'arr_group_filter'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = "All"

@st.cache_data
def load_snowflake_data():
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    
    snow_df = session.table('DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY')
    df = snow_df.to_pandas()
    
    # df2 = session.sql('select distinct AMPLITUDE_PLAN_TYPE_C ,ID as ACCOUNT_ID from PROD_DBT_COMMONS.SALESFORCE.VW_DATALAKE_SALESFORCE_ACCOUNT where ID in (select  distinct ACCOUNT_ID from FUTURE_CHURN_PREDICTION_DATASET_FEAT_ENGG)').to_pandas()
    # df = df.merge(df2, on="ACCOUNT_ID", how="inner")
    
    # temp_snow_df = session.table('PROD_DBT_COMMONS.SALESFORCE.VW_DATALAKE_SALESFORCE_ACCOUNT')
    # temp_df = temp_snow_df.to_pandas()
    
    df['CHURN_ARR'] = df['CHURN_ARR'].abs()
    df['PREDICTION_LABEL'] = df['PREDICTION_LABEL'].replace({
        "no_churn": "No Churn",
        "partial_churn": "Partial Churn",
        "full_churn": "Full Churn"
    })
    df['AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS'] = df['AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS'].replace({
        0: False,
        1: True
    })
    # df = pd.merge(df, temp_df, left_on='ACCOUNT_ID', right_on='ID', how='left')
    return df

df = load_snowflake_data()

# --- Filters + Title Card ---
card_style = "background-color:#fff;border:1.5px solid #d3d3d3;border-radius:8px;padding:0.7rem 0.5rem 0.7rem 0.5rem;margin-bottom:0.5rem;text-align:center;min-width:110px;min-height:60px;"
filter_label_style = "font-size:0.95rem;font-weight:500;color:#6c757d;margin-bottom:0.2rem;"

# Added an extra column for the reset button
top_cols = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 1])

with top_cols[0]:
    st.subheader("Churn Insights by Cohort")
	st.write("Code Review Testing by LLM")
with top_cols[1]:
    st.markdown(f"<span style='{filter_label_style}'>Country</span>", unsafe_allow_html=True)
    country = st.selectbox(" ", options=["All"] + sorted(df['HQ_COUNTRY_GC'].fillna('Undefined').unique().tolist()), label_visibility="collapsed", key='country_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[2]:
    st.markdown(f"<span style='{filter_label_style}'>Region</span>", unsafe_allow_html=True)
    region = st.selectbox("  ", options=["All"] + sorted(df['GEO_C'].fillna('Undefined').unique().tolist()), label_visibility="collapsed", key='region_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[3]:
    st.markdown(f"<span style='{filter_label_style}'>Multiproduct</span>", unsafe_allow_html=True)
    multiproduct_attach = st.selectbox("   ", options=["All"] + sorted(df['AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS'].unique().tolist()), label_visibility="collapsed", key='multiproduct_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[4]:
    st.markdown(f"<span style='{filter_label_style}'>GTM Segment</span>", unsafe_allow_html=True)
    gtm_segment = st.selectbox("    ", options=["All"] + sorted(df['GTM_SEGMENT_C'].unique().tolist()), label_visibility="collapsed", key='gtm_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[5]:
    st.markdown(f"<span style='{filter_label_style}'>Plan Type</span>", unsafe_allow_html=True)
    plan_type = st.selectbox("Plan Type", options= ["All"] + sorted(df['PLG_UPGRADE_SOURCE'].unique().tolist()), label_visibility="collapsed", key='plan_type_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[6]:
    st.markdown(f"<span style='{filter_label_style}'>Churn Type</span>", unsafe_allow_html=True)
    churn_type = st.selectbox("     ", options=["All"] + sorted(df['PREDICTION_LABEL'].unique().tolist()), label_visibility="collapsed", key='churn_type_filter')
    st.markdown("</div>", unsafe_allow_html=True)
with top_cols[7]:
    st.markdown(f"<span style='{filter_label_style}'>Beginning ARR</span>", unsafe_allow_html=True)
    arr_bins = [-float('inf'), 50000, 100000, 250000, 500000, float('inf')]
    arr_labels = ["<50k", "50-100k", "100k-250k", "250k-500k", "500k+"]
    df['BEGINNING_ARR_GROUP'] = pd.cut(df['BEGINNING_ARR'], bins=arr_bins, labels=arr_labels, right=False)
    arr_group_options = ["All"] + arr_labels
    beginning_arr_group = st.selectbox("      ", options=arr_group_options, label_visibility="collapsed", key='arr_group_filter')
    st.markdown("</div>", unsafe_allow_html=True)

# --- Add the reset button in the last column ---
with top_cols[8]:
    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True) # Align button vertically
    st.button("Reset", on_click=reset_all_filters, use_container_width=True)


# --- Filter DataFrame ---
filtered_df = df.copy()
if country != "All":
    filtered_df = filtered_df[filtered_df['HQ_COUNTRY_GC'] == country].copy()
if region != "All":
    filtered_df = filtered_df[filtered_df['GEO_C'] == region].copy()
if multiproduct_attach != "All":
    filtered_df = filtered_df[filtered_df['AVG_MULTIPRODUCT_ATTACH_BINARY_PAST_9_MONTHS'] == multiproduct_attach].copy()
if gtm_segment != "All":
    filtered_df = filtered_df[filtered_df['GTM_SEGMENT_C'] == gtm_segment].copy()
if plan_type != "All":
    filtered_df = filtered_df[filtered_df['PLG_UPGRADE_SOURCE'] == plan_type].copy()
if churn_type != "All":
    filtered_df = filtered_df[filtered_df['PREDICTION_LABEL'] == churn_type].copy()
if beginning_arr_group != "All":
    filtered_df = filtered_df[filtered_df['BEGINNING_ARR_GROUP'] == beginning_arr_group].copy()

# (The rest of your code remains unchanged)
num_accounts = len(filtered_df)
total_churn_prob = filtered_df['PREDICT_PROBA_1'].sum()
full_churn = (filtered_df['PREDICTION_LABEL'] == 'Full Churn').sum() if 'PREDICTION_LABEL' in filtered_df.columns else 0
partial_churn = (filtered_df['PREDICTION_LABEL'] == 'Partial Churn').sum() if 'PREDICTION_LABEL' in filtered_df.columns else 0
full_churn_prob = filtered_df[filtered_df['PREDICTION_LABEL'] == 'Full Churn']['PREDICT_PROBA_2'].mean() if 'PREDICTION_LABEL' in filtered_df.columns else 0
partial_churn_prob = filtered_df[filtered_df['PREDICTION_LABEL'] == 'Partial Churn']['PREDICT_PROBA_1'].mean() if 'PREDICTION_LABEL' in filtered_df.columns else 0
total_renewal = filtered_df['RENEWAL_ARR'].sum()
total_churn_arr = (filtered_df['BEGINNING_ARR'] * filtered_df['PREDICT_PROBA_2']).sum()
percent_at_risk = (total_churn_arr / total_renewal * 100) if total_renewal > 0 else 0


# --- Summary Cards ---
card_style = """
    background-color: #fff;
    border: 1.5px solid #d3d3d3;
    border-radius: 8px;
    padding: 0.7rem 0.5rem 0.7rem 0.5rem;
    margin-bottom: 0.5rem;
    text-align: center;
    min-width: 110px;
    min-height: 60px;
"""

card_value_style = "font-size:1.5rem;font-weight:600;color:#343a40;margin:0;"
card_label_style = "font-size:0.95rem;font-weight:500;color:#6c757d;margin-bottom:0.2rem;"

summary_cols = st.columns(8)
with summary_cols[0]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Number of Accounts</div><div style='{card_value_style}'>{num_accounts}</div></div>", unsafe_allow_html=True)
with summary_cols[1]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Full Churn</div><div style='{card_value_style}'>{full_churn}</div></div>", unsafe_allow_html=True)
with summary_cols[2]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Partial Churn</div><div style='{card_value_style}'>{partial_churn}</div></div>", unsafe_allow_html=True)
with summary_cols[3]:
    val = f"{full_churn_prob * 100:.2f} %" if not np.isnan(full_churn_prob) else "N/A"
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Full Churn Avg. Probability</div><div style='{card_value_style}'>{val}</div></div>", unsafe_allow_html=True)
with summary_cols[4]:
    val = f"{partial_churn_prob:.0f} %" if not np.isnan(partial_churn_prob) else "N/A"
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Partial Churn Avg. Probability</div><div style='{card_value_style}'>{val}</div></div>", unsafe_allow_html=True)
with summary_cols[5]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Total Renewal Amount</div><div style='{card_value_style}'>$ {total_renewal:,.0f}</div></div>", unsafe_allow_html=True)
with summary_cols[6]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>Total Churn ARR</div><div style='{card_value_style}'>$ {total_churn_arr:,.0f}</div></div>", unsafe_allow_html=True)
with summary_cols[7]:
    st.markdown(f"<div style='{card_style}'><div style='{card_label_style}'>% Amount at Churn Risk</div><div style='{card_value_style}'>{percent_at_risk:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# --- Load actual churn ARR for previous months (May, June) ---
def load_actual_churn_arr():
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    snow_df = session.table('DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.ACCOUNT__ACTUAL_CHURN_ARR_VALUES')
    df = snow_df.to_pandas()
    df = df[pd.to_datetime(df['CLOSE_DATE_MONTH']).dt.month <= 6]
    df['CHURN_ARR'] = df['CHURN_ARR'].abs()
    df['CHURN_TYPE'] = df['CHURN_TYPE'].replace({
        "no_churn": "No Churn",
        "partial_churn": "Partial Churn",
        "full_churn": "Full Churn"
    })
    return df

# Load actual churn ARR for previous months (May, June)
actual_churn_df = load_actual_churn_arr()

arr_bins = [-float('inf'), 50000, 100000, 250000, 500000, float('inf')]
arr_labels = [
    "<50k",
    "50-100k",
    "100k-250k",
    "250k-500k",
    "500k+"
]
# Bin the BEGINNING_ARR values
actual_churn_df['BEGINNING_ARR_GROUP'] = pd.cut(actual_churn_df['BEGINNING_ARR'], bins=arr_bins, labels=arr_labels, right=False)

# --- Filter DataFrame ---
if country != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['HQ_COUNTRY_GC'] == country].copy()
if region != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['GEO_C'] == region].copy()
if multiproduct_attach != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['MULTIPRODUCT_ATTACH_BINARY'] == multiproduct_attach].copy()
if gtm_segment != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['GTM_SEGMENT_C'] == gtm_segment].copy()
if plan_type != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['PLG_UPGRADE_SOURCE'] == plan_type].copy()
if churn_type != "All":
    actual_churn_df = actual_churn_df[actual_churn_df['CHURN_TYPE'] == churn_type].copy()
if beginning_arr_group!= "All":
    actual_churn_df = actual_churn_df[actual_churn_df['BEGINNING_ARR_GROUP'] == beginning_arr_group].copy()
    

row1_col1, row1_col2 = st.columns([2,2])
with row1_col1:
    with st.container(border=True):
        # Group by month
        if not actual_churn_df.empty:
            actual_monthly = actual_churn_df.groupby('CLOSE_DATE_MONTH').agg({
                'CHURN_ARR': 'sum',
                'ACCOUNT_ID': 'count'
            }).reset_index()
        else:
            actual_monthly = pd.DataFrame(columns=['CLOSE_DATE_MONTH', 'CHURN_ARR', 'ACCOUNT_ID'])
    
        
        # --- Predicted churn ARR for future months (July, Aug, Sep, etc.) ---
        if 'CLOSE_DATE_MONTH' in filtered_df.columns and 'BEGINNING_ARR' in filtered_df.columns and 'PREDICT_PROBA_2' in filtered_df.columns:
            filtered_df['Predicted_CHURN_ARR'] = filtered_df['BEGINNING_ARR'] * filtered_df['PREDICT_PROBA_2']
            pred_monthly = filtered_df.groupby('CLOSE_DATE_MONTH').agg({
                'Predicted_CHURN_ARR': 'sum',
                'ACCOUNT_ID': 'count'
            }).reset_index()
        else:
            pred_monthly = pd.DataFrame(columns=['CLOSE_DATE_MONTH', 'Predicted_CHURN_ARR', 'ACCOUNT_ID'])
    
        # --- Combine actual and predicted for plotting ---
        all_months = pd.concat([
            actual_monthly[['CLOSE_DATE_MONTH']],
            pred_monthly[['CLOSE_DATE_MONTH']]
        ]).drop_duplicates().sort_values('CLOSE_DATE_MONTH')
    
        # Merge for plotting
        plot_df = all_months.copy()
        plot_df['CLOSE_DATE_MONTH'] = pd.to_datetime(plot_df['CLOSE_DATE_MONTH'])
        plot_df['Actual Churn ARR'] = plot_df['CLOSE_DATE_MONTH'].map(
            dict(zip(pd.to_datetime(actual_monthly['CLOSE_DATE_MONTH']), actual_monthly['CHURN_ARR']))
        )
        plot_df['Predicted Churn ARR'] = plot_df['CLOSE_DATE_MONTH'].map(
            dict(zip(pd.to_datetime(pred_monthly['CLOSE_DATE_MONTH']), pred_monthly['Predicted_CHURN_ARR']))
        )
        plot_df['Actual Accounts'] = plot_df['CLOSE_DATE_MONTH'].map(
            dict(zip(pd.to_datetime(actual_monthly['CLOSE_DATE_MONTH']), actual_monthly['ACCOUNT_ID']))
        )
        plot_df['Predicted Accounts'] = plot_df['CLOSE_DATE_MONTH'].map(
            dict(zip(pd.to_datetime(pred_monthly['CLOSE_DATE_MONTH']), pred_monthly['ACCOUNT_ID']))
        )
    
    
        fig = go.Figure()
    
        # Bar for actuals (May, June)
        if plot_df['Actual Churn ARR'].notnull().any():
            fig.add_trace(go.Bar(
                x=plot_df['CLOSE_DATE_MONTH'].dt.strftime('%b %Y'),
                y=plot_df['Actual Churn ARR'],
                text=plot_df['Actual Accounts'],
                textposition='inside',
                marker_color='#A3E635',  # Greenish for actuals
                name='Actual Churn ARR',
            ))
    
        # Bar for predicted (July+)
        if plot_df['Predicted Churn ARR'].notnull().any():
            fig.add_trace(go.Bar(
                x=plot_df['CLOSE_DATE_MONTH'].dt.strftime('%b %Y'),
                y=plot_df['Predicted Churn ARR'],
                text=plot_df['Predicted Accounts'],
                textposition='inside',
                marker_color='#36A2EB',  # Blue for predicted
                name='Predicted Churn ARR',
            ))
    
        # Add annotation for each bar (actual and predicted)
        for i, row in plot_df.iterrows():
            month_str = row['CLOSE_DATE_MONTH'].strftime('%b %Y')
            def format_arr(val):
                if pd.isnull(val):
                    return ""
                if abs(val) >= 1e6:
                    return f"${val/1e6:.2f}M"
                elif abs(val) >= 1e3:
                    return f"${val/1e3:.1f}K"
                else:
                    return f"${val:.0f}"
            if not pd.isnull(row['Actual Churn ARR']):
                fig.add_annotation(
                    x=month_str,
                    y=row['Actual Churn ARR'],
                    text=format_arr(row['Actual Churn ARR']),
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12, color='black')
                )
            if not pd.isnull(row['Predicted Churn ARR']):
                fig.add_annotation(
                    x=month_str,
                    y=row['Predicted Churn ARR'],
                    text=format_arr(row['Predicted Churn ARR']),
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12, color='black')
                )
    
        fig.update_layout(
            title="Churn ARR Over Time (Actuals for Past Months, Predicted for Future)",
            yaxis_title="Churn ARR (sum)",
            xaxis_title="Month",
            bargap=0.2,
            barmode='group',
            showlegend=True,
            height=350,
            yaxis=dict(tickprefix="$")
        )
        st.plotly_chart(fig, use_container_width=True)
    

with row1_col2:
    with st.container(border=True):
        st.markdown("<h5>Top 10 Most Influential Churn Drivers (Cohort)</h5>", unsafe_allow_html=True)
        feature_importance_agg = {}
        feature_freq = {}
        for _, row in filtered_df.iterrows():
            if 'TOP10_SHAP_JSON' in row and pd.notnull(row['TOP10_SHAP_JSON']):
                try:
                    shap_dict = json.loads(row['TOP10_SHAP_JSON'])
                    for feat, imp in shap_dict.items():
                        if feat not in feature_importance_agg:
                            feature_importance_agg[feat] = []
                            feature_freq[feat] = 0
                        feature_importance_agg[feat].append(abs(imp))
                        feature_freq[feat] += 1
                except Exception:
                    continue
                
        if feature_importance_agg:
            agg_df = pd.DataFrame([
                {"Driver": feat, "MeanAbsImportance": np.mean(imps), "Frequency": feature_freq[feat]}
                for feat, imps in feature_importance_agg.items()
            ])
            agg_df = agg_df.sort_values("MeanAbsImportance", ascending=False).head(10)
            fig4 = px.bar(
                agg_df, x="MeanAbsImportance", y="Driver", orientation="h",
                color="MeanAbsImportance", color_continuous_scale="Viridis"
            )
            fig4.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis=dict(showticklabels=True, showgrid=False, zeroline=False), margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)


st.markdown("**Summary**")

# Define columns needed for calculation and display
required_cols = [
    'CLOSE_DATE_MONTH', 'ACCOUNT_NAME', 'BLADES_PURCHASED', 'PREDICTION_LABEL', 'PREDICT_PROBA_2', 'BEGINNING_ARR'
]

if all(col in filtered_df.columns for col in required_cols):
    summary_df = filtered_df[required_cols].copy()
    # Calculate Predicted Churn ARR
    summary_df['Predicted Churn ARR'] = (summary_df['BEGINNING_ARR'].abs() * summary_df['PREDICT_PROBA_2'])
    # Reorder columns for display
    summary_df = summary_df[
        ['CLOSE_DATE_MONTH', 'ACCOUNT_NAME', 'BLADES_PURCHASED', 'PREDICTION_LABEL', 'Predicted Churn ARR', 'PREDICT_PROBA_2']
    ]
    # Rename columns for display
    summary_df.columns = [
        'Renewal Date', 'Account Name', 'Products Purchased', 'Predicted Churn Type', 'Predicted Churn ARR', 'Churn Probability'
    ]
    # Add total row for Predicted Churn ARR
    total_row = {
        'Renewal Date': '',
        'Account Name': 'Total',
        'Products Purchased': '',
        'Predicted Churn Type': '',
        'Predicted Churn ARR': summary_df['Predicted Churn ARR'].sum(),
        'Churn Probability': ''
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
else:
    summary_df = filtered_df.head(0)

st.dataframe(summary_df, use_container_width=True,
             column_config={
                 "Predicted Churn ARR": st.column_config.NumberColumn(format="$%.2f"),
                 "Churn Probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)
             })
			 
What -if -
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import joblib
import os
import pickle
from utils import apply_industry_grouping, BROADER_INDUSTRY_CATEGORIES
import shap
import plotly.graph_objects as go

from snowflake.snowpark.context import get_active_session
session = get_active_session()

st.set_page_config(
    page_title="Churn Insights – WHAT-IF-ANALYSIS",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .scrollable-feature-panel {
            max-height: 500px;
            overflow-y: auto;
            padding-right: 8px;
        }
        .section-title { font-size: 1.2rem; font-weight: 700; color: #6c3fcf; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)


COUNTRY_STANDARDIZATION_MAP = {
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


# --- Load DataFrame from Snowflake ---
@st.cache_data
def load_feature_data():
    snow_df = session.table('DEV_DATA_SCIENCE.ACCT_LVL_CHURN_PRED.ACCNT_CHURN_FUTURE_DATASET_SCORED_WITH_LOCAL_EXPLAINABILITY_SKLEARN')
    df = snow_df.to_pandas()
    return df

df = load_feature_data()

# --- Preprocess columns for display/model ---
df['INDUSTRY'] = apply_industry_grouping(df, 'INDUSTRY', BROADER_INDUSTRY_CATEGORIES, 'Other/Unspecified')
df['HQ_COUNTRY_GC'] = df['HQ_COUNTRY_GC'].map(COUNTRY_STANDARDIZATION_MAP).fillna(df['HQ_COUNTRY_GC'])

# --- Load Preprocessor and Model from Snowflake Stage ---
def load_object_from_stage(session, stage_name, filename_on_stage, folder_name):
    local_temp_path = '/tmp/'
    full_stage_path = f'{stage_name.strip("/")}/{folder_name}/{filename_on_stage}'
    session.file.get(full_stage_path, local_temp_path)
    downloaded_file_path = os.path.join(local_temp_path, filename_on_stage)
    with open(downloaded_file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    os.remove(downloaded_file_path)
    return loaded_object

@st.cache_resource
def load_preprocessor_and_model():
    MODELS_STAGE = '@MODELS_STAGE'
    VERSION_FOLDER = 'sklearn_model_artifacts_v1'
    preprocessor = load_object_from_stage(session, MODELS_STAGE, 'preprocessor.pkl', VERSION_FOLDER)
    model = load_object_from_stage(session, MODELS_STAGE, 'model.pkl', VERSION_FOLDER)
    return preprocessor, model

preprocessor, model = load_preprocessor_and_model()

# --- Callback function to reset filters ---
def reset_filters():
    """Resets all user inputs to their default states."""
    st.session_state.label_filter = 'All'
    
    # Delete other keys to force them to their initial state on rerun
    for key in ['account_select', 'date_select', 'feature_state', 'last_account']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state['trigger_infer'] = True


# --- UI: Top Row Filters ---
prediction_labels = ['All', 'Full Churn', 'Partial Churn', 'No Churn']
label_display = {'All': 'All', 'Full Churn': 'Full Churn', 'Partial Churn': 'Partial Churn', 'No Churn': 'No Churn'}

top_row = st.columns([2, 1.5, 1.5, 1.5, 1])
with top_row[0]:
    st.subheader("Churn Insights - WHAT-IF-ANALYSIS")
with top_row[1]:
    selected_label = st.selectbox("Filter by Churn Type", [label_display[l] for l in prediction_labels], key="label_filter", index=0)
    label_map_inv = {v: k for k, v in label_display.items()}
    selected_label_internal = label_map_inv[selected_label]

# Filter accounts by label
if selected_label_internal == 'All':
    filtered_df = df.copy()
else:
    filtered_df = df[df['PREDICTION_LABEL'] == selected_label_internal]
account_names = filtered_df['ACCOUNT_NAME'].unique()

with top_row[2]:
    selected_account = st.selectbox("Select Account", account_names, key="account_select")
with top_row[3]:
    selected_date = st.date_input("Date", datetime.now(), key="date_select")
with top_row[4]:
    st.markdown("<div><br></div>", unsafe_allow_html=True) # Vertical alignment
    st.button("Reset", on_click=reset_filters, use_container_width=True)


# --- Feature Editing Panel ---
left, center, right = st.columns([1.2, 2, 2])

# Handle case where selected_account might not be in the filtered list after reset
if selected_account not in filtered_df['ACCOUNT_NAME'].values:
    # This can happen briefly after reset. We default to the first available account.
    if len(filtered_df) > 0:
        selected_account = filtered_df['ACCOUNT_NAME'].iloc[0]
        st.session_state.account_select = selected_account # Update state
    else:
        st.error("No accounts found for the selected Churn Type.")
        st.stop() # Stop execution if no accounts are available

account_row = filtered_df[filtered_df['ACCOUNT_NAME'] == selected_account].iloc[0]

# These columns are required for the model but should not be shown in the UI
fixed_feature_cols = ['HQ_COUNTRY_GC', 'GEO_C', 'BLADES_PURCHASED', 'INDUSTRY']
columns_to_exclude = [
    'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_1_MONTHS',
    'AVG_MONTHLY_DAYS_SINCE_BOUNCE_DIGITAL_EMAILS_PAST_1_MONTHS',
    'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_3_MONTHS',
    'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_6_MONTHS',
    'AVG_MONTHLY_UNSUBS_FOLLOWING_CLICK_EVENTS_FOR_DIGITAL_EMAILS_PAST_9_MONTHS',
    'ACCOUNT_ID', 'ACCOUNT_NAME', 'CLOSE_DATE_MONTH', 'PREDICTION_LABEL'
] + fixed_feature_cols

# Only show editable features (not fixed) in the UI
editable_feature_cols = [col for col in df.columns if col not in columns_to_exclude]

# Compute min/max for sliders
num_feature_ranges = {}
for col in editable_feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        col_min = float(df[col].min()) if not pd.isnull(df[col].min()) else 0.0
        col_max = float(df[col].max()) if not pd.isnull(df[col].max()) else 100.0
        if col_min == col_max:
            col_min, col_max = 0.0, col_max if col_max > 0 else 100.0
        num_feature_ranges[col] = (col_min, col_max)

# Initialize session state for features
if 'feature_state' not in st.session_state or st.session_state.get('last_account') != selected_account:
    st.session_state['feature_state'] = {col: account_row[col] for col in editable_feature_cols}
    # Always keep fixed features in state for model input
    for col in fixed_feature_cols:
        st.session_state['feature_state'][col] = account_row[col]
    st.session_state['last_account'] = selected_account
    st.session_state['trigger_infer'] = True

# --- Feature Panel with Scroll ---
def feature_panel():
    changed = False
    with st.container(border=True, height=800):
        st.markdown("<div style='text-align:center'><span class='section-title'>Features</span></div>", unsafe_allow_html=True)
        for col in editable_feature_cols:
            val = st.session_state['feature_state'][col]
            # Boolean detection
            is_bool_col = pd.api.types.is_bool_dtype(df[col])
            if not is_bool_col:
                unique_vals = pd.Series(df[col].dropna().unique())
                if set(unique_vals).issubset({0, 1, True, False}):
                    is_bool_col = True
            # Dropdown for low cardinality categorical features
            unique_vals = df[col].dropna().unique()
            if is_bool_col:
                new_val = st.checkbox(col, value=bool(val), key=col)
                if new_val != val:
                    st.session_state['feature_state'][col] = bool(new_val) if pd.api.types.is_bool_dtype(df[col]) else int(new_val)
                    changed = True
            elif pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = num_feature_ranges[col]
                step = 1.0 if df[col].dtype.kind in 'i' else 0.01
                new_val = st.slider(col, min_value=min_val, max_value=max_val, value=float(val) if not pd.isnull(val) else min_val, step=step, key=col)
                if new_val != val:
                    st.session_state['feature_state'][col] = new_val
                    changed = True
            elif len(unique_vals) <= 10:
                options = list(sorted(unique_vals))
                if val not in options:
                    options = [val] + options
                new_val = st.selectbox(col, options, index=options.index(val) if val in options else 0, key=col)
                if new_val != val:
                    st.session_state['feature_state'][col] = new_val
                    changed = True
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                new_val = st.date_input(col, value=pd.to_datetime(val) if not pd.isnull(val) else datetime.now(), key=col)
                if new_val != val:
                    st.session_state['feature_state'][col] = new_val
                    changed = True
            else:
                new_val = st.text_input(col, value=str(val) if not pd.isnull(val) else '', key=col)
                if new_val != val:
                    st.session_state['feature_state'][col] = new_val
                    changed = True
    return changed

with left:
    # Defensive: ensure all editable and fixed features are present in feature_state
    for col in editable_feature_cols + fixed_feature_cols:
        if col not in st.session_state['feature_state']:
            st.session_state['feature_state'][col] = account_row[col]
    changed = feature_panel()
    if changed:
        st.session_state['trigger_infer'] = True

# --- Helper: Get feature names after one-hot encoding ---
def get_feature_names(column_transformer, X_df):
    output_features = []
    for name, preprocessor, features in column_transformer.transformers_:
        if name == 'num':
            output_features.extend(features)
        elif name == 'cat':
            if hasattr(preprocessor, 'named_steps') and 'onehot' in preprocessor.named_steps:
                onehot_encoder = preprocessor.named_steps['onehot']
                output_features.extend(onehot_encoder.get_feature_names_out(features))
            else:
                output_features.extend(preprocessor.get_feature_names_out(features))
    return output_features

# --- Center: Churn Probability ---
with center:
    with st.container(border=True, height=500):
        st.markdown("<div style='text-align:center'><span class='section-title'>Churn Probability</span></div>", unsafe_allow_html=True)
        # Always include fixed features in model input
        model_input = {**{col: st.session_state['feature_state'][col] for col in editable_feature_cols}, **{col: st.session_state['feature_state'][col] for col in fixed_feature_cols}}
        feature_values = [model_input[col] for col in editable_feature_cols + fixed_feature_cols]
        if st.session_state.get('trigger_infer', False):
            with st.spinner("Running What-If Analysis..."):
                X_df = pd.DataFrame([model_input])
                X_proc = preprocessor.transform(X_df)
                feature_names = get_feature_names(preprocessor, X_df)
                proba = model.predict_proba(X_proc)[0]
                class_labels = getattr(model, "classes_", [0, 1, 2])
                class_display = {0: "No Churn", 1: "Partial Churn", 2: "Full Churn"}
                pie_labels = [class_display.get(c, str(c)) for c in class_labels]
                pie_colors = ['#36A2EB', '#FBBF24', '#FF6384']
                # Pie chart improvements
                fig = go.Figure(go.Pie(
                    labels=pie_labels,
                    values=proba,
                    hole=0.55,
                    marker_colors=pie_colors[:len(proba)],
                    textinfo='percent',
                    textfont_size=20,
                    insidetextorientation='auto',
                    sort=False
                ))
                fig.update_layout(
                    margin=dict(t=50, b=10, l=10, r=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=16)),
                )
                st.plotly_chart(fig, use_container_width=True)
                explainer = shap.Explainer(model)
                shap_values = explainer.shap_values(X_proc)
                pred_class = int(model.predict(X_proc)[0])
                if isinstance(shap_values, list):
                    shap_row = shap_values[pred_class][0]
                else:
                    shap_row = shap_values[0, :, pred_class]
                top_idx = np.argsort(np.abs(shap_row))[::-1][:10]
                local_importance = {feature_names[j]: float(shap_row[j]) for j in top_idx}
                st.session_state['last_local_importance'] = local_importance
            st.session_state['trigger_infer'] = False

# --- Right: Local Explainability ---
with right:
    with st.container(border=True):
        st.markdown("<div style='text-align:center'><span class='section-title'>Top 10 Most Influential Churn Drivers</span></div>", unsafe_allow_html=True)
        local_importance = st.session_state.get('last_local_importance', {})
        if local_importance:
            drivers_df = pd.DataFrame.from_records(list(local_importance.items()), columns=['Feature', 'Importance'])
            drivers_df = drivers_df.sort_values('Importance', ascending=False).head(10)
            fig = px.bar(drivers_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis=dict(showticklabels=True, showgrid=False, zeroline=False), margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No local feature importance returned.")
			


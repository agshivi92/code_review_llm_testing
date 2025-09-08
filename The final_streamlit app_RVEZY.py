import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
from snowflake.snowpark.functions import col,expr,count
import plotly.graph_objects as go
#import calendar
#import altair as alt


# Write directly to the app
#st.set_page_config(layout="wide", page_title="**RVEZY BOOKING APP**")


# Get the current credentials
def pie_plot(df, names):
    fig = px.pie(df, names=names, hole = 0.3)
    fig.update_layout(title={'text':f"{names} distribution", 'x': 0.5, 'y':0.95})
    st.plotly_chart(fig, use_container_width=True)

def pie_plot2(df, column_name):
    fig = px.pie(df, names=column_name, hole=0.3)
    fig.update_layout(title={'text': f"{column_name} distribution", 'x': 0.5, 'y': 0.95})
    st.plotly_chart(fig, use_container_width=True)

def bar_plot(self,df, X, Color):
    fig = px.bar(df, x=X, color=Color)
    fig.update_layout(title={'text':f"{X} vs {Color}", 'x': 0.5, 'y':0.95}, margin= dict(l=0,r=10,b=10,t=30), yaxis_title=Color, xaxis_title=X)
    st.plotly_chart(fig, use_container_width=True)
    
def bar_plot_XY(df, X, Y, Color):
    fig = px.bar(df, x=X, y=Y, color=Color)
        
    # Calculate the percentage values
    total_counts = df.groupby(X)[Y].sum().reset_index()
    total_counts['percentage'] = (total_counts[Y] / total_counts[Y].sum()) * 100

    # Add percentage labels to the bars
    for i, row in total_counts.iterrows():
        percentage_label = f"{row['percentage']:.2f}%"
        fig.add_annotation(
            x=row[X],
            y=row[Y],
            text=percentage_label,
            showarrow=False,
            font=dict(size=10),
            yshift=10
        )
        
        fig.update_layout(
            title={'text': f"{X} by {Y}", 'x': 0.5, 'y': 0.95},
            margin=dict(l=0, r=10, b=10, t=30),
            yaxis_title=Color,
            xaxis_title=X,
        )
    
        st.plotly_chart(fig, use_container_width=True)

    
      
def main(session):
    st.set_page_config(layout="wide")
    st.header("**BOOKING PERFORMANCE DASHBOARD**")
    ############### creating 4 sheets for the app#########################
    tabs = st.tabs(['Booking Details', 'GMV','Payment Rate','Cancellation Rate'])     
    with tabs[0]:
        df = session.sql(f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW").to_pandas()
        #st.write(total_review_count_df.dtypes)
        df['DATE'] = pd.to_datetime(df['DATE'])
        MIN_DATE = df['DATE'].min()
        MAX_DATE = df['DATE'].max()
        #st.write(MIN_DATE,MAX_DATE) 
        st.subheader("**EXECUTIVE REPORT**")
        ######################for the below graphs and report , we are checking the max date of the whole dataset and report and graph is generated for the max date###################
        GMV_REPORT= session.sql(f" select DATE, COUNTRY,GMV , PD_GMV,  LWSD_GMV, LYSD_GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_TYPE = 'Country' ").to_pandas()
        st.subheader("**Executive Report by GMV**")
        st.write(GMV_REPORT)
        
        ################## GMV comparison of Canada and USA ##############################################
        f1,f2 = st.columns(2)
        with f1:
            st.write("**GMV as per country Canada**")
            Canada_REPORT= session.sql(f" select DATE, COUNTRY,GMV , PD_GMV,  LWSD_GMV, LYSD_GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'Canada' ").to_pandas()       
            df = pd.DataFrame(Canada_REPORT, columns=['DATE','GMV' , 'PD_GMV',  'LWSD_GMV', 'LYSD_GMV']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)
        with f2:
            st.write("**GMV as per country USA**")
            USA_REPORT= session.sql(f" select DATE, COUNTRY,GMV , PD_GMV,  LWSD_GMV, LYSD_GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'USA' ").to_pandas()       
            df = pd.DataFrame(USA_REPORT, columns=['DATE','GMV' , 'PD_GMV',  'LWSD_GMV', 'LYSD_GMV']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)

        ######################for the below graphs and report , we are checking the max date of the whole dataset and report and graph is generated for the max date###################33
        Payment_REPORT= session.sql(f" select DATE, COUNTRY, PAYMENT_RATE, PD_PAYMENT_RATE, LWSD_PAYMENT_RATE, LYSD_PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_TYPE = 'Country' ")
        st.subheader("**Executive Report by Payment Rate**")
        st.write(Payment_REPORT)
        ################## Payment Rate comparison of Canada and USA ##############################################
        f1,f2 = st.columns(2)
        with f1:
            st.write("**Payment Rate as per country Canada**")
            Canada_REPORT= session.sql(f" select DATE, COUNTRY,PAYMENT_RATE , PD_PAYMENT_RATE,  LWSD_PAYMENT_RATE, LYSD_PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'Canada' ").to_pandas()       
            df = pd.DataFrame(Canada_REPORT, columns=['DATE','PAYMENT_RATE' , 'PD_PAYMENT_RATE',  'LWSD_PAYMENT_RATE', 'LYSD_PAYMENT_RATE']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)
        with f2:
            st.write("**Payment Rate as per country USA**")
            USA_REPORT= session.sql(f" select DATE, COUNTRY,PAYMENT_RATE , PD_PAYMENT_RATE,  LWSD_PAYMENT_RATE, LYSD_PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'USA' ").to_pandas()       
            df = pd.DataFrame(USA_REPORT, columns=['DATE','PAYMENT_RATE' , 'PD_PAYMENT_RATE',  'LWSD_PAYMENT_RATE', 'LYSD_PAYMENT_RATE']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)

        
        ######################for the below graphs and report , we are checking the max date of the whole dataset and report and graph is generated for the max date###################33    
        Cancellation_REPORT= session.sql(f" select DATE, COUNTRY, CANCELLATION_RATE, PD_CANCELLATION_RATE, LWSD_CANCELLATION_RATE, LYSD_CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_TYPE = 'Country' ")
        st.subheader("**Executive Report by Cancellation Rate**")
        st.write(Cancellation_REPORT)
        ################## Cancellation Rate comparison of Canada and USA ##############################################
        f1,f2 = st.columns(2)
        with f1:
            st.write("**Cancellation Rate as per country Canada**")
            Canada_REPORT= session.sql(f" select DATE, COUNTRY,CANCELLATION_RATE , PD_CANCELLATION_RATE,  LWSD_CANCELLATION_RATE, LYSD_CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'Canada' ").to_pandas()       
            df = pd.DataFrame(Canada_REPORT, columns=['DATE','CANCELLATION_RATE' , 'PD_CANCELLATION_RATE',  'LWSD_CANCELLATION_RATE', 'LYSD_CANCELLATION_RATE']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)
        with f2:
            st.write("**Cancellation Rate as per country USA**")
            USA_REPORT= session.sql(f" select DATE, COUNTRY,CANCELLATION_RATE , PD_CANCELLATION_RATE,  LWSD_CANCELLATION_RATE, LYSD_CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE = '{MAX_DATE}' and SEGMENT_VALUE = 'USA' ").to_pandas()       
            df = pd.DataFrame(USA_REPORT, columns=['DATE','CANCELLATION_RATE' , 'PD_CANCELLATION_RATE',  'LWSD_CANCELLATION_RATE', 'LYSD_CANCELLATION_RATE']) 
            cols = df.columns[1:]
            df2 = df.melt(id_vars='DATE', value_vars = cols)
            fig = px.scatter(df2, x="variable", y="value", color='variable')  
            fig.update_traces(marker_size=10)
            st.plotly_chart(fig, use_container_width=True)

        ######################for the below graphs and report , we have calender asking for start and end date for which you want to see the report and graphs of & by default the start date is the min date of the dataset and end date is max date of the dataset the max date of the whole dataset ###################33
        st.subheader("**BOOKING REPORT**")
        start_date = st.date_input("Select Start Date:", min_value=MIN_DATE, max_value=MAX_DATE, value=MIN_DATE)
        #date=pd.DataFrame(start_date).pd.to_datetime
        end_date = st.date_input("Select End Date", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE)
        #start_date_year = start_date.year
        report_df =pd.DataFrame()
        country_filter = ""
        if start_date:
            #######Adding the country filter , if country_filter = " ", then the report and graphs has data for both (Canada and USA) and country_filter = Canada then only Canada records will be displayed and similarily for USA############################# 
            country_filter = st.selectbox("Please select Country :", [" ",'Canada','USA'])
            #st.write(country_filter)
            if country_filter == " ":
                query= f"""
                    SELECT 
                        DATE,
                        COUNTRY,
                        SEGMENT_TYPE,
                        SEGMENT_VALUE,
                        GMV,
                        PD_GMV,
                        LWSD_GMV,
                        LYSD_GMV,
                        PAYMENT_RATE,
                        PD_PAYMENT_RATE,
                        LWSD_PAYMENT_RATE,
                        LYSD_PAYMENT_RATE,
                        CANCELLATION_RATE,
                        PD_CANCELLATION_RATE,
                        LWSD_CANCELLATION_RATE,
                        LYSD_CANCELLATION_RATE 
                    FROM 
                        RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW 
                    WHERE 
                        TO_DATE(DATE) BETWEEN TO_DATE('{start_date}') AND TO_DATE('{end_date}') 
                        
                    """
                report_df = session.sql(query).to_pandas()
                st.write("**Report calculated for countries Canada & USA for selected date**")
                st.write(report_df)
            elif country_filter == "Canada":
                 query= f"""
                    SELECT 
                        DATE,
                        COUNTRY,
                        SEGMENT_TYPE,
                        SEGMENT_VALUE,
                        GMV,
                        PD_GMV,
                        LWSD_GMV,
                        LYSD_GMV,
                        PAYMENT_RATE,
                        PD_PAYMENT_RATE,
                        LWSD_PAYMENT_RATE,
                        LYSD_PAYMENT_RATE,
                        CANCELLATION_RATE,
                        PD_CANCELLATION_RATE,
                        LWSD_CANCELLATION_RATE,
                        LYSD_CANCELLATION_RATE 
                    FROM 
                        RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW 
                    WHERE 
                        TO_DATE(DATE) BETWEEN TO_DATE('{start_date}') AND TO_DATE('{end_date}') 
                        AND COUNTRY = 'Canada'     
                    """
                 report_df = session.sql(query).to_pandas()
                 st.write("**Report calculated for country Canada for selected date**")
                 st.write(report_df)
            elif country_filter == "USA":
                 query= f"""
                    SELECT 
                        DATE,
                        COUNTRY,
                        SEGMENT_TYPE,
                        SEGMENT_VALUE,
                        GMV,
                        PD_GMV,
                        LWSD_GMV,
                        LYSD_GMV,
                        PAYMENT_RATE,
                        PD_PAYMENT_RATE,
                        LWSD_PAYMENT_RATE,
                        LYSD_PAYMENT_RATE,
                        CANCELLATION_RATE,
                        PD_CANCELLATION_RATE,
                        LWSD_CANCELLATION_RATE,
                        LYSD_CANCELLATION_RATE 
                    FROM 
                        RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW 
                    WHERE 
                        TO_DATE(DATE) BETWEEN TO_DATE('{start_date}') AND TO_DATE('{end_date}') 
                        AND COUNTRY = 'USA'     
                    """
                 report_df = session.sql(query).to_pandas()
                 st.write("**Report calculated for country USA for selected date**")
                 st.write(report_df)
                # RV_df= session.sql(query).to_pandas()
                #st.write(RV_df)
                # query_1= f" select distinct SEGMENT_VALUE  as USERAGE from RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW where SEGMENT_TYPE= 'User age'  and COUNTRY = '{country_filter}' and (DATE BETWEEN '{start_date}' AND '{end_date}') "
                # UserAge_df= session.sql(query_1).to_pandas()
                # #st.write(UserAge_df)
                # selected_RV = RV_df['RV'].tolist()
                # #st.write(selected_RV)
                # selected_age = UserAge_df['USERAGE'].tolist()
                # rv_filter= st.selectbox("Please select RV type",selected_RV)
                # # st.write(rv_filter)
                # userage_filter = st.selectbox("Please select User Age Group :",selected_age)
                # st.write(userage_filter)
                # rv_filter = st.multiselect("Please select RV Type :", selected_RV, disabled=False)
                
                # #st.write(rv_filter)
                # Userage_filter = st.multiselect("Please select User Age Group :",selected_age, disabled=False)
                # #age ="'" + Userage_filter +"'"
                # #placeholders=', '.join(['%s' for _ in selected_RV]) 
                # rv_filter_str = "','".join(rv_filter)
                # Userage_filter_str = "','".join(Userage_filter)
                # st.write("rv_filter_str:", rv_filter_str)
                # st.write("Userage_filter_str:", Userage_filter_str)
                #st.write(rv_filter_str,Userage_filter_str)
                
               # query_2 = f""" SELECT COUNTRY,GMV , PD_GMV,  LWSD_GMV, LYSD_GMV, PAYMENT_RATE, PD_PAYMENT_RATE, LWSD_PAYMENT_RATE, LYSD_PAYMENT_RATE, CANCELLATION_RATE, PD_CANCELLATION_RATE, LWSD_CANCELLATION_RATE, LYSD_CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE (TO_DATE(DATE) BETWEEN TO_DATE('{start_date}') AND TO_DATE('{end_date}')) and (COUNTRY = '{country_filter}')  and (SEGMENT_TYPE IN {rv_filter} OR SEGMENT_TYPE IN {Userage_filter}) """
                # query_2 = f"""
                #     SELECT 
                #         DATE,
                #         COUNTRY,
                #         SEGMENT_TYPE,
                #         SEGMENT_VALUE,
                #         GMV,
                #         PD_GMV,
                #         LWSD_GMV,
                #         LYSD_GMV,
                #         PAYMENT_RATE,
                #         PD_PAYMENT_RATE,
                #         LWSD_PAYMENT_RATE,
                #         LYSD_PAYMENT_RATE,
                #         CANCELLATION_RATE,
                #         PD_CANCELLATION_RATE,
                #         LWSD_CANCELLATION_RATE,
                #         LYSD_CANCELLATION_RATE 
                #     FROM 
                #         RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW 
                #     WHERE 
                #         TO_DATE(DATE) BETWEEN TO_DATE(?) AND TO_DATE(?) 
                #         AND COUNTRY = ? 
                #         AND (SEGMENT_VALUE = (?) OR SEGMENT_VALUE = (?))
                #     """
                # #Execute the SQL query
                
                # report_df = session.sql(query_2, (start_date, end_date, country_filter, rv_filter,userage_filter)).to_pandas()
                # #st.write(query_2)
                # # Display the result
                # st.write(report_df)
                #selected_year_df =  session.sql(f" select  distinct extract(YEAR from DATE) AS year from RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' ").to_pandas()
                #st.write(selected_year_df) 
                #selected_month_df =  session.sql(f" select  distinct extract(MONTH from DATE) AS month from RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' ").to_pandas()
                #st.write(selected_month_df)
          #################################################  Booking Percentage pie charts  ###########################################################
            if country_filter == " ":
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' AND SEGMENT_VALUE IN ('Canada', 'USA','Others') "
                report_df2 = session.sql(query2).to_pandas()
                #st.write(report_df2)
                formatted_text = f"Booking Percentage for all countries for duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per country**")
                
                conditions = [
                     (report_df2['SEGMENT_VALUE']=='Canada') ,
                     (report_df2['SEGMENT_VALUE']=='USA'),
                    (report_df2['SEGMENT_VALUE']=='Others') ] 
                values = ["CANADA","USA","Others"]
                report_df2['COUNTRY'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1], 
                                                 "Others"))
               
                pie_plot(df=report_df2, names = 'COUNTRY')
            elif country_filter=="Canada":
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}'  and  SEGMENT_TYPE = 'Country'"
                report_df2 = session.sql(query2).to_pandas()
                #st.write(report_df2)
                formatted_text = f"Booking Percentage as per country: Canada for duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per country**")
                
                conditions = [
                      (report_df2['SEGMENT_VALUE']=='Canada') ,
                      (report_df2['SEGMENT_VALUE']!='Canada')]
                values = ["CANADA","Others"]
                report_df2['COUNTRY'] = np.select(conditions, values, default ="Others" )
               
                pie_plot2(df=report_df2, column_name = 'COUNTRY')
            elif country_filter=="USA":
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}'  and  SEGMENT_TYPE = 'Country'"
                report_df2 = session.sql(query2).to_pandas()
                #st.write(report_df2)
                formatted_text = f"Booking Percentage as per country: USA for duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per country**")
                
                conditions = [
                      (report_df2['SEGMENT_VALUE'] =='USA'),
                     (report_df2['SEGMENT_VALUE'].isin(['Canada', 'Others']))] 
                values = ["USA","Others"]
                report_df2['COUNTRY'] = np.select(conditions, values, default="Others")
               
                pie_plot2(df=report_df2, column_name = 'COUNTRY')
         ####################################################################  RV Type Percentage pie charts #################################################################### 
            if country_filter == " ":
                formatted_text = f"Overall Booking Percentage as per all RVType for countries Canada & USA for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN  '{start_date}' AND '{end_date}' and SEGMENT_TYPE = 'RV Type'"
                report_df3 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df3['SEGMENT_VALUE']=='Fifth Wheel') ,
                     (report_df3['SEGMENT_VALUE']=='Tent Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Vintage Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Hybrid') ,
                     (report_df3['SEGMENT_VALUE']=='Toy Hauler') ,
                     (report_df3['SEGMENT_VALUE']=='Class A') ,
                     (report_df3['SEGMENT_VALUE']=='Class B') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Vintage Motorhome') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Micro Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Truck Camper') ,
                     (report_df3['SEGMENT_VALUE']=='Campervan') ,
                     (report_df3['SEGMENT_VALUE']=='RV Cottage') ,
                      
                     ] 
                values = ["Fifth Wheel","Tent Trailer","Travel Trailer","Vintage Trailer","Hybrid","Toy Hauler","Class A","Class B","Class C","Vintage Motorhome","Travel Trailer","Class C","Micro Trailer","Truck Camper","Campervan","RV Cottage"]
                
                report_df3['RVTYPE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 np.where(conditions[3], values[3],
                                                 np.where(conditions[4], values[4],
                                                 np.where(conditions[5], values[5],
                                                 np.where(conditions[6], values[6],
                                                 np.where(conditions[7], values[7],
                                                 np.where(conditions[8], values[8],
                                                 np.where(conditions[9], values[9],
                                                 np.where(conditions[10], values[10],
                                                 np.where(conditions[11], values[11],
                                                 np.where(conditions[12], values[12],
                                                 np.where(conditions[13], values[13],
                                                 np.where(conditions[14], values[14],
                                                 np.where(conditions[15], values[15],
                                                 "Unknown"))))))))))))))))
                pie_plot(df=report_df3, names = 'RVTYPE')
            elif country_filter == "Canada":
                formatted_text = f"Overall Booking Percentage as per all RVType for country: {country_filter} for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN  '{start_date}' AND '{end_date}' and COUNTRY = '{country_filter}' and SEGMENT_TYPE = 'RV Type'"
                report_df3 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df3['SEGMENT_VALUE']=='Fifth Wheel') ,
                     (report_df3['SEGMENT_VALUE']=='Tent Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Vintage Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Hybrid') ,
                     (report_df3['SEGMENT_VALUE']=='Toy Hauler') ,
                     (report_df3['SEGMENT_VALUE']=='Class A') ,
                     (report_df3['SEGMENT_VALUE']=='Class B') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Vintage Motorhome') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Micro Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Truck Camper') ,
                     (report_df3['SEGMENT_VALUE']=='Campervan') ,
                     (report_df3['SEGMENT_VALUE']=='RV Cottage') ,
                      
                     ] 
                values = ["Fifth Wheel","Tent Trailer","Travel Trailer","Vintage Trailer","Hybrid","Toy Hauler","Class A","Class B","Class C","Vintage Motorhome","Travel Trailer","Class C","Micro Trailer","Truck Camper","Campervan","RV Cottage"]
                
                report_df3['RVTYPE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 np.where(conditions[3], values[3],
                                                 np.where(conditions[4], values[4],
                                                 np.where(conditions[5], values[5],
                                                 np.where(conditions[6], values[6],
                                                 np.where(conditions[7], values[7],
                                                 np.where(conditions[8], values[8],
                                                 np.where(conditions[9], values[9],
                                                 np.where(conditions[10], values[10],
                                                 np.where(conditions[11], values[11],
                                                 np.where(conditions[12], values[12],
                                                 np.where(conditions[13], values[13],
                                                 np.where(conditions[14], values[14],
                                                 np.where(conditions[15], values[15],
                                                 "Unknown"))))))))))))))))
                pie_plot(df=report_df3, names = 'RVTYPE')
            elif country_filter == "USA":
                formatted_text = f"Overall Booking Percentage as per all RVType for country: {country_filter} for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN  '{start_date}' AND '{end_date}' and COUNTRY = '{country_filter}' and SEGMENT_TYPE = 'RV Type'"
                report_df3 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df3['SEGMENT_VALUE']=='Fifth Wheel') ,
                     (report_df3['SEGMENT_VALUE']=='Tent Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Vintage Trailer'),
                     (report_df3['SEGMENT_VALUE']=='Hybrid') ,
                     (report_df3['SEGMENT_VALUE']=='Toy Hauler') ,
                     (report_df3['SEGMENT_VALUE']=='Class A') ,
                     (report_df3['SEGMENT_VALUE']=='Class B') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Vintage Motorhome') ,
                     (report_df3['SEGMENT_VALUE']=='Travel Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Class C') ,
                     (report_df3['SEGMENT_VALUE']=='Micro Trailer') ,
                     (report_df3['SEGMENT_VALUE']=='Truck Camper') ,
                     (report_df3['SEGMENT_VALUE']=='Campervan') ,
                     (report_df3['SEGMENT_VALUE']=='RV Cottage') ,
                      
                     ] 
                values = ["Fifth Wheel","Tent Trailer","Travel Trailer","Vintage Trailer","Hybrid","Toy Hauler","Class A","Class B","Class C","Vintage Motorhome","Travel Trailer","Class C","Micro Trailer","Truck Camper","Campervan","RV Cottage"]
                
                report_df3['RVTYPE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 np.where(conditions[3], values[3],
                                                 np.where(conditions[4], values[4],
                                                 np.where(conditions[5], values[5],
                                                 np.where(conditions[6], values[6],
                                                 np.where(conditions[7], values[7],
                                                 np.where(conditions[8], values[8],
                                                 np.where(conditions[9], values[9],
                                                 np.where(conditions[10], values[10],
                                                 np.where(conditions[11], values[11],
                                                 np.where(conditions[12], values[12],
                                                 np.where(conditions[13], values[13],
                                                 np.where(conditions[14], values[14],
                                                 np.where(conditions[15], values[15],
                                                 "Unknown"))))))))))))))))
                pie_plot(df=report_df3, names = 'RVTYPE')
    ############################################################ User Age Percentage pie charts ##################################################################################################
            if country_filter ==" ":
                formatted_text = f"Overall Booking Percentage as per all UserAge Group for countries Canada & USA for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per User Age Group**")
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}'  and SEGMENT_TYPE = 'User age'"
                report_df4 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df4['SEGMENT_VALUE']=='Below_25') ,
                     (report_df4['SEGMENT_VALUE']=='Between_25-40') ,
                    (report_df4['SEGMENT_VALUE']=='Above_40') ,
                     (report_df4['SEGMENT_VALUE']=='Unknown') 
                     ] 
                values = ["BELOW_25","Between_25_40","ABOVE_40","Unknown"]
                report_df4['USER AGE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 "Unknown")))
               
                pie_plot(df=report_df4, names = 'USER AGE')
            if country_filter =="Canada":
                formatted_text = f"Overall Booking Percentage as per all UserAge Group for country: {country_filter} for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per User Age Group**")
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = '{country_filter}' and SEGMENT_TYPE = 'User age'"
                report_df4 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df4['SEGMENT_VALUE']=='Below_25') ,
                     (report_df4['SEGMENT_VALUE']=='Between_25-40') ,
                    (report_df4['SEGMENT_VALUE']=='Above_40') ,
                     (report_df4['SEGMENT_VALUE']=='Unknown') 
                     ] 
                values = ["BELOW_25","Between_25_40","ABOVE_40","Unknown"]
                report_df4['USER AGE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 "Unknown")))
               
                pie_plot(df=report_df4, names = 'USER AGE')
            if country_filter =="USA":
                formatted_text = f"Overall Booking Percentage as per all UserAge Group for country: {country_filter} for the selected duration {start_date} - {end_date} "
                st.subheader(formatted_text)
                #st.subheader("**Overall Booking Percentage as per User Age Group**")
                query2= f" SELECT * FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = '{country_filter}' and SEGMENT_TYPE = 'User age'"
                report_df4 = session.sql(query2).to_pandas()
                conditions = [
                     (report_df4['SEGMENT_VALUE']=='Below_25') ,
                     (report_df4['SEGMENT_VALUE']=='Between_25-40') ,
                    (report_df4['SEGMENT_VALUE']=='Above_40') ,
                     (report_df4['SEGMENT_VALUE']=='Unknown') 
                     ] 
                values = ["BELOW_25","Between_25_40","ABOVE_40","Unknown"]
                report_df4['USER AGE'] = np.where(conditions[0], values[0], 
                                                 np.where(conditions[1], values[1],
                                                 np.where(conditions[2], values[2],
                                                 "Unknown")))
               
                pie_plot(df=report_df4, names = 'USER AGE')
###### Tab 1: GMV visualization as per selected date, date_filter and Country, which is further segmented into RV type and User age group #####################################################################
    with tabs[1]:
        
        selected_end_date = st.date_input("Please select the GMV date:", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE)
        ###### this selected date will be by default the max date of the dataset ######################################
        date_filter = st.selectbox("Select GMV Date Filter", ["Last 7 Days","Last 28 Days", "Last 3 Months","Last 6 Months" ,"Last 1 Year", "All Time"])
        ###### by default the date filter is set to be 7 days filter and you change it as per your requirement ######################################
        if date_filter == "Last 7 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=7)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 28 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=28)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 3 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=3)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 6 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=6)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 1 Year":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(years=1)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "All Time":
            start_date = pd.to_datetime('2017-01-01')  # or any other appropriate start date
            end_date = pd.to_datetime(selected_end_date)
        else:
            st.error("Invalid date filter selected")
        country_filter = st.selectbox("Please select the country for GMV:",[' ','Canada','USA'])
        ####################### country filter as mentioned above ##############################################
        if country_filter == ' ':
            ### if country_filter = ' ', we are displaying visualization corresponding to both countries####################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True) 

            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                 
#########################GMV as per country  'Canada' ##############################################################################
        elif country_filter == 'Canada':  
            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
############################################# Canada segregartion on RV Type ########################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
##############################################Canada user Age segregartion###############################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,GMV,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
#########################GMV as per country  'USA' ##############################################################################
        elif country_filter == 'USA':
            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='GMV', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)  

            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)       
###################################GMV as per User Age #######################################
            USA_REPORT_PY= session.sql(f" SELECT DATE,GMV ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,GMV , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**GMV of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**GMV of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='GMV', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='GMV',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

    
################### Tab 2: Payment Rate visualization as per selected date, date_filter and Country, which is further segmented into RV type and User age group ########################################################

    with tabs[2]:
        
        selected_end_date = st.date_input("Please select the payment date:", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE)
        #selected_end_date = pd.to_datetime(f'{max(selected_years)}-{max(selected_months)}-01') + pd.DateOffset(months=1, days=-1)
        date_filter = st.selectbox("Select Payment Date Filter", ["Last 7 Days","Last 28 Days", "Last 3 Months","Last 6 Months" ,"Last 1 Year", "All Time"])
        
        if date_filter == "Last 7 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=7)
            end_date = pd.to_datetime(selected_end_date)
            #last_year_start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=372) 
            #last_year_end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 28 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=28)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 3 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=3)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 6 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=6)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 1 Year":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(years=1)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "All Time":
            start_date = pd.to_datetime('2017-01-01')  # or any other appropriate start date
            end_date = pd.to_datetime(selected_end_date)
        else:
            st.error("Invalid date filter selected")
            
        country_filter = st.selectbox("Please select the country for Payment Rate:",[' ','Canada','USA'])
        if country_filter == ' ':
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='DATE',barmode='group' )
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='DATE',barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            #########################Payment Rate as per RV Type ##############################################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            ###################################Payment Rate as per User age #######################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
###################################################################tab 2 : filter on country= Canada############################################################
        elif country_filter == 'Canada':
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='DATE',barmode='group' )
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                       #########################Payment Rate as per RV Type ##############################################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            ###################################Payment Rate as per User age for country canada#######################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
 
 #########################################################TAB 2 COUNTRY FILTER = USA##########################################################################               
        elif country_filter== 'USA':
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='DATE',barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            #################################################RV Type filter ############################################
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            #################################################user age filter ########################################
            USA_REPORT_PY= session.sql(f" SELECT DATE,PAYMENT_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,PAYMENT_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Payment Rate of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='PAYMENT_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Payment Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
#################################### Tab 3: Cancellation Rate visualization as per selected date, date_filter and Country, which is further segmented into RV type and User age group ################################################# ##################################

    with tabs[3]:
        
        selected_end_date = st.date_input("Please select the cancellation date:", min_value=MIN_DATE, max_value=MAX_DATE, value=MAX_DATE)
        #selected_end_date = pd.to_datetime(f'{max(selected_years)}-{max(selected_months)}-01') + pd.DateOffset(months=1, days=-1)
        date_filter = st.selectbox("Select Cancellation Date Filter", ["Last 7 Days","Last 28 Days", "Last 3 Months","Last 6 Months" ,"Last 1 Year", "All Time"])
        
        if date_filter == "Last 7 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=7)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 28 Days":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(days=28)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 3 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=3)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 6 Months":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(months=6)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "Last 1 Year":
            start_date = pd.to_datetime(selected_end_date) - pd.DateOffset(years=1)
            end_date = pd.to_datetime(selected_end_date)
        elif date_filter == "All Time":
            start_date = pd.to_datetime('2017-01-01')  # or any other appropriate start date
            end_date = pd.to_datetime(selected_end_date)
        else:
            st.error("Invalid date filter selected")
        country_filter = st.selectbox("Please select the country for Cancellation Rate:",[' ','Canada','USA'])
        if country_filter == ' ':
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
     #########################Cancellation Rate as per RV Type ##############################################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    ###################################Cancellation Rate as per User age #######################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
    
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
        ################################################################ filter on country Canada####################################################
        elif country_filter == 'Canada':
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            #########################Cancellation Rate as per RV Type ##############################################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            ##########################################User age##################################################
            Canada_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            Canada_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'Canada' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of Canada for selected date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of Canada for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(Canada_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
        #####################################################Filter based on Country USA#######################################
        elif country_filter == 'USA':
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE  FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of USA for selected date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='DATE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='DATE'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'RV Type' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Payment Rate of USA for selected date as per RV Type**")
                
                fig = px.line(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date as per RV Type**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.line(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE',  markers=True)
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='RV Type'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            ######################################################################################################################
            USA_REPORT_PY= session.sql(f" SELECT DATE,CANCELLATION_RATE ,SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN '{start_date}' AND '{end_date}' and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            USA_REPORT_LY= session.sql(f" SELECT DATE,CANCELLATION_RATE , SEGMENT_VALUE FROM RVEZY_CONSUMPTION.EXECUTIVE.BOOKING_REPORT_VIEW WHERE DATE BETWEEN DATEADD(YEAR,-1,TO_DATE('{start_date}')) AND DATEADD(YEAR,-1,TO_DATE('{end_date}')) and COUNTRY = 'USA' AND SEGMENT_TYPE = 'User age' ").to_pandas()
            f1,f2 = st.columns(2)
            with f1:
                st.subheader("**Cancellation Rate of USA for selected date as per User Age**")
                
                fig = px.bar(USA_REPORT_PY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Selected Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
            with f2:
                st.subheader("**Cancellation Rate of USA for last year same date as per User Age**")
                #df = px.data.gapminder().query("country in ['CANADA','USA']")
                
                fig = px.bar(USA_REPORT_LY, x='DATE', y='CANCELLATION_RATE', color='SEGMENT_VALUE', barmode='group')
        
                # Update chart layout
                fig.update_layout(
                    xaxis_title='Last Year Date',
                    yaxis_title='Cancellation Rate',
                    legend_title='User Age'
                )
                
                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
if __name__ == '__main__':
    session = get_active_session()
    main(session)
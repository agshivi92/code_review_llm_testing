# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd 
import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt 
import plotly.express as px
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import statsmodels.api as sm 
import os
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import *
import snowflake.snowpark.types as T
from snowflake.snowpark.types import *
from pycaret.classification import *
import shap
from shap import *

matplotlib.use("Agg")

st.set_option('deprecation.showPyplotGlobalUse', False)

class EDA_Dataframe_Analysis():
    def __init__(self):
        print("General_EDA object created")
    
    def show_dtypes(self,x):
        return x.dtypes
        
    def show_columns(self,x):
        return x.columns
        
    def Show_Missing(self,x):
        return x.isna().sum()
        
    def Show_Missing1(self,x):
        return x.isna().sum()
            
    def Show_Missing2(self,x):
        return x.isna().sum()
            
    def show_hist(self,x):
        fig = plt.figure(figsize = (15,20))
        ax = fig.gca()
        return x.hist(ax=ax)
            
    def Tabulation(self,x):
        table = pd.DataFrame(x.dtypes,columns=['dtypes'])
        table1 =pd.DataFrame(x.columns,columns=['Names'])
        table = table.reset_index()
        table= table.rename(columns={'index':'Name'})
        table['No of Missing'] = x.isnull().sum().values
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
        table['First Observation'] = x.loc[0].values
        table['Second Observation'] = x.loc[1].values
        table['Third Observation'] = x.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
        return table
            
    def Numerical_variables(self,x):
        Num_var = [var for var in x.columns if x[var].dtypes!="object"]
        Num_var = x[Num_var]
        return Num_var
            
    def categorical_variables(self,x):
        cat_var = [var for var in x.columns if x[var].dtypes=="object"]
        cat_var = x[cat_var]
        return cat_var
            
    def plotly(self,a,x,y):
        fig = px.scatter(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
										line=dict(width=2,
											color='DarkSlateGrey')),
							selector=dict(mode='markers'))
        fig.show()
            
    def show_displot(self,x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)
        plt.subplot(122)
        x.plot.box(figsize=(16,5))
        plt.show()
            
    def Show_DisPlot(self,x):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12,7))
        return sns.distplot(x, bins = 25)
            
    def Show_CountPlot(self,x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x,ax=ax)
        
    def plotly_histogram(self,a,x,y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
										line=dict(width=2,
												color='DarkSlateGrey')),
							selector=dict(mode='markers'))
        fig.show()
            
    def plotly_violin(self,a,x,y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
										line=dict(width=2,
												color='DarkSlateGrey')),
							selector=dict(mode='markers'))
        fig.show()
            
    def Show_PairPlot(self,x):
        return sns.pairplot(x)
        
    def Show_HeatMap(self,x):
        f,ax = plt.subplots(figsize=(15, 15))
        x = self.Numerical_variables(x)
        return sns.heatmap(x.corr(),annot=True,ax=ax);
            
    def label(self,x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x=le.fit_transform(x)
        return x
            
    def label1(self,x):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x=le.fit_transform(x)
        return x
            
    def concat(self,x,y,z,axis):
        return pd.concat([x,y,z],axis)
        
    def dummy(self,x):
        return pd.get_dummies(x)
        
    def qqplot(self,x):
        return sm.qqplot(x, line ='45')
        
    def Anderson_test(self,a):
        return anderson(a)
        
    def PCA(self,x):
        pca =PCA(n_components=8)
        principlecomponents = pca.fit_transform(x)
        principledf = pd.DataFrame(data = principlecomponents)
        return principledf
            
    def outlier(self,x):
        high=0
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3-q1
        low = q1-1.5*iqr
        high += q3+1.5*iqr
        outlier = (x.loc[(x < low) | (x > high)])
        return(outlier)


class Attribute_Information():

    def __init__(self):
        
        print("Attribute Information object created")
        
    def Column_information(self,data):
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of Records'] = data.count()
        data_info.loc[0,'No of Columns'] = len(data.columns)
        # data_info.loc[0,'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = len(data.Numerical_variables(data))
        # data_info.loc[0,'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        # data_info.loc[0,'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        # data_info.loc[0,'No of zero variance variables'] = data.loc[:,data.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info
    
    def __get_missing_values(self,data):
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

        
    def __iqr(self,x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self,x):
        upper_out = x.quantile(q=0.75) + (1.5 * self.__iqr(x))
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self,df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
            i = i+1
        return data_info_num
    
    def statistical_summary(self,df):
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num



class Charts():

    def __init__(self):
        print("Charts object created")
    
    def scatter_plot(self,df,X,Y, Color=None):
        fig = px.scatter(df, y = Y, x=X,orientation='h', color=Color, render_mode='svg')
        fig.update_layout(title={'text':f"{X} vs {Y}", 'x': 0.5, 'y':0.95}, margin= dict(l=0,r=10,b=10,t=30), yaxis_title=Y, xaxis_title=X)
        st.plotly_chart(fig, use_container_width=True)

    def box_plot(self,df,X,Y):
        fig = px.box(df, y = Y, x=X)
        fig.update_layout(title={'text':f"{X} vs {Y}", 'x': 0.5, 'y':0.95}, margin= dict(l=0,r=10,b=10,t=30), yaxis_title=Y, xaxis_title=X)
        st.plotly_chart(fig, use_container_width=True)


    def bar_plot(self,df, X, Color):
        fig = px.bar(df, x=X, color=Color)
        fig.update_layout(title={'text':f"{X} vs {Color}", 'x': 0.5, 'y':0.95}, margin= dict(l=0,r=10,b=10,t=30), yaxis_title=Color, xaxis_title=X)
        st.plotly_chart(fig, use_container_width=True)


    def plotly_violin(self,df,Y):
        fig = px.violin(df, y=Y)
        fig.update_traces(marker=dict(size=10,
                          line=dict(width=2,
                          color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        st.plotly_chart(fig, use_container_width=True)


 

def main():
    st.title("Classmates - Likelyhood to Click On Email")
    tabs = st.tabs(['EDA:chart_with_upwards_trend:', 'Model Inferences:gear:','Explainability:white_check_mark:'])

    st.info(" Streamlit Web Application ")
    
    def create_table_config(df):
        table_config = {}
        for column in df.columns:
            table_config[column] = str(column)
        return table_config
			

    with tabs[0]:
        st.subheader("Exploratory Data Analysis")
        data = session.sql('select * from KIPI.ADMIN.MASTER_MARKETING_DATA2_KIPI_CLEANED')
        data_p = data.limit(5).toPandas()


        num_cols = ['EMAIL_CLICK_COUNT_LAST_30_DAYS','EMAIL_CLICK_COUNT_LAST_60_DAYS','EMAIL_CLICK_COUNT_LAST_90_DAYS','EMAIL_CLICK_COUNT_LAST_120_DAYS',
             'EMAIL_CLICK_COUNT_LAST_365_DAYS','EMAIL_CLICK_COUNT_LAST_730_DAYS','LIFETIME_EMAIL_CLICK_COUNT',
             'EMAIL_OPEN_COUNT_LAST_30_DAYS','EMAIL_OPEN_COUNT_LAST_60_DAYS','EMAIL_OPEN_COUNT_LAST_90_DAYS','EMAIL_OPEN_COUNT_LAST_120_DAYS',
             'EMAIL_OPEN_COUNT_LAST_365_DAYS','EMAIL_OPEN_COUNT_LAST_730_DAYS','LIFETIME_EMAIL_OPEN_COUNT',
             'EMAIL_SEND_COUNT_LAST_30_DAYS','EMAIL_SEND_COUNT_LAST_60_DAYS','EMAIL_SEND_COUNT_LAST_90_DAYS','EMAIL_SEND_COUNT_LAST_120_DAYS',
             'EMAIL_SEND_COUNT_LAST_365_DAYS','EMAIL_SEND_COUNT_LAST_730_DAYS','LIFETIME_EMAIL_SEND_COUNT',
             'PHOTO_COUNT', 'SCHOOL_SIZE_RATIO','MAX_GRAD_YEAR','AGE','CLASS_SIZE', 'DAYS_SINCE_LAST_LOGIN',
             'LOGIN_COUNT', 'LOGIN_COUNT_LAST_30_DAYS', 'LOGIN_COUNT_LAST_60_DAYS', 'LOGIN_COUNT_LAST_90_DAYS',
             'GB_COUNT_LAST_30_DAYS','GB_COUNT_LAST_60_DAYS','GB_COUNT_LAST_90_DAYS','GB_COUNT_LAST_120_DAYS',
             'GB_COUNT_LAST_365_DAYS','GB_COUNT_LAST_730_DAYS','LIFETIME_COUNT_OF_GBS',
            'IRU_COUNT_LAST_120_DAYS','IRU_COUNT_LAST_365_DAYS','IRU_COUNT_LAST_730_DAYS','LIFETIME_IRU_TAGS', 'LIFETIME_HINOTE_COUNT',
                                            'GENERATED_EMAIL_MSG_COUNT','CUSTOMER_TRANSALL_COUNT','LIFETIME_SLOTS_COUNT','DAYS_BW_LAST_TWO_LOGINS']
         

        if data is not None:
            st.success("Data Frame Loaded successfully")
            st.subheader("Data Information")
            if st.checkbox("Show Datatypes of columns "):
                st.write(data_p.dtypes)
                
            if st.checkbox("Show Column names"):
                st.write(data_p.columns)
            
            if st.checkbox("Show Numerical Variables"):
                num_df = dataframe.Numerical_variables(data_p)
                numer_df=pd.DataFrame(num_df).head()
                st.dataframe(numer_df)
            
            if st.checkbox("Show Categorical Variables"):
                new_df = dataframe.categorical_variables(data_p)
                catego_df=pd.DataFrame(new_df).head()
                st.dataframe(catego_df)

            st.subheader("Data Analysis")

            
#------------------------- MAIN ANALYSIS CODE STARTS HERE ----------------------------------------------------------------------------------------            

#------------------------- 1. UNIVARIATE ANALYSIS ----------------------------------------------------------------------------------------            
        
            if st.checkbox("Univariate Analysis"):
                if st.checkbox("**Numerical Features** (Univariate)"):
                    st.subheader('Numerical Features - EDA')
    
                    
                    x = st.selectbox("Select Category to view distribution", [ 'AGE', 'EMAIL_CLICK_COUNT_LAST_90_DAYS','EMAIL_CLICK_COUNT_LAST_120_DAYS',
                 'EMAIL_CLICK_COUNT_LAST_365_DAYS','EMAIL_CLICK_COUNT_LAST_730_DAYS','LIFETIME_EMAIL_CLICK_COUNT',
                 'EMAIL_OPEN_COUNT_LAST_120_DAYS',
                 'EMAIL_SEND_COUNT_LAST_90_DAYS','EMAIL_SEND_COUNT_LAST_120_DAYS',
                 'LOGIN_COUNT', 'LOGIN_COUNT_LAST_60_DAYS', 'LOGIN_COUNT_LAST_90_DAYS' ])
                  
                    if x in ('AGE'):
                        df = data.groupBy(x).count().toPandas()
                        bins=[0,10,20,30,40,50,60,70,80,90,100]
                        fg = sns.displot(data=df, x=x, bins=bins, weights='COUNT', height=5, aspect=2, stat='percent')
                        for ax in fg.axes.ravel():                    
                            for c in ax.containers:                
                                labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]               
                                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=0, padding=2)                 
                            ax.margins(y=0.2)
                        st.pyplot(plt)
    
                    if x in ('EMAIL_CLICK_COUNT_LAST_90_DAYS','EMAIL_CLICK_COUNT_LAST_120_DAYS',
                 'EMAIL_CLICK_COUNT_LAST_365_DAYS','EMAIL_CLICK_COUNT_LAST_730_DAYS','LIFETIME_EMAIL_CLICK_COUNT',
                            'LOGIN_COUNT_LAST_60_DAYS', 'LOGIN_COUNT_LAST_90_DAYS'):
                        df = data.groupBy(x).count().toPandas()
                        bins=[0,2,4,6,8,10,12,14,16,18,20,22]
                        fg = sns.displot(data=df, x=x,bins=bins, weights='COUNT', height=5, aspect=2, stat='percent')
                        for ax in fg.axes.ravel():                    
                            for c in ax.containers:                
                                labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]               
                                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=0, padding=2)                 
                            ax.margins(y=0.2)
                        st.pyplot(plt)
    
                    if x in ('EMAIL_OPEN_COUNT_LAST_120_DAYS',
                 'EMAIL_SEND_COUNT_LAST_90_DAYS','EMAIL_SEND_COUNT_LAST_120_DAYS'):
                        df = data.groupBy(x).count().toPandas()
                        bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
                        fg = sns.displot(data=df, x=x,bins=bins, weights='COUNT', height=5, aspect=2, stat='percent')
                        for ax in fg.axes.ravel():                    
                            for c in ax.containers:                
                                labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]               
                                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=0, padding=2)                 
                            ax.margins(y=0.2)
                        st.pyplot(plt)
    
                    if x in ( 'LOGIN_COUNT'):
                        df = data.groupBy(x).count().toPandas()
                        bins=[0,50,100,200,300,400,500,600,700,800,900,1000,1500,2000]
                        fg = sns.displot(data=df, x=x,bins=bins, weights='COUNT', height=5, aspect=2, stat='percent')
                        for ax in fg.axes.ravel():                    
                            for c in ax.containers:                
                                labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]               
                                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=0, padding=2)                 
                            ax.margins(y=0.2)
                        st.pyplot(plt)

                
#----------------------------------------------------------------------------------------------------------------            
                
                if st.checkbox("**Categorical Features** (Univariate)"):     
                    st.subheader('Categorical Features - EDA')

                    x = st.selectbox("Select Category to view distribution", ['EMAIL_DOMAIN', 'GENDER', 'REUNION_YEAR', 'DNER_FLAG', 
                                                                               'COMMERCIAL_IND', 'ACQUISITION_SOURCE', 'MEMBERSHIP_STATUS', 'MEMBERSHIP_STATUS_HISTORY',
                                                                               'GB_MOMENTUM', 'EMAIL_DOMAIN_GROUP', 'WEEKLY_DIGEST_IND', 'MY_PROFILE_VISITS_IND', 
                                                                               'MY_REMINDER_IND', 'SCHOOL_COMMUNITY_IND', 'MY_REMEMBERS_IND', 'MY_INBOX_IND', 
                                                                               'MY_PROFILE_NOTES_IND', 'MY_PRIVATE_MESSAGES_IND', 'SCHOOL_PROFILE_IND', 
                                                                               'SCHOOL_REMINDER_IND', 'SCHOOL_YEARBOOK_IND', 'NEW_CLASSMATES_FEATURES_IND', 
                                                                               'DO_NOT_EMAIL_IND', 'AGE_GROUP', 'SCHOOL_NAME', 'SCHOOL_CITY', 'SCHOOL_STATE',
                                                                               'PUBLISHER_OWNER_NAME', 'PUBLISHER_NAME', 'LOGIN_COUNT_BUCKETS', 
                                                                               'UPLOADED_PHOTOS_OR_NOT', 'TAGGED_IN_YEARBOOK_OR_NOT', 'REUNION_INVITEE', 
                                                                               'FACEBOOK_TOKEN_EXPIRED', 'FACEBOOK_TOKEN_AVAILABLE'] )

                    if(x):

                        df = data.groupBy(x).count().toPandas()
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top10 = df.sort_values(by='COUNT', ascending=False).head(10)            
                        ax.pie(top10['COUNT'], labels=top10[x], autopct='%1.1f%%')
                        ax.set_title(f'Distribution of {x}')
                        ax.axis('equal') 
                        st.pyplot(plt)


#------------------------- 2. BIVARIATE ANALYSIS ----------------------------------------------------------------------------------------            
            
            if st.checkbox("Bivariate Analysis"):
                # st.write(":gray[*Choose the feature type for analysis -*]")
                if st.checkbox("**Numerical Features** (Bivariate)"):
                    st.subheader('BIVARIATE ANALYSIS - Numerical Features')
        
                    CLASS_SIZE_col=[x for x in num_cols if x  in ('CLASS_SIZE')]
                    AGE_col=[x for x in num_cols if x  in ('AGE')]
                    thirty_days_col_list = [x for x in num_cols if x in ('EMAIL_SEND_COUNT_LAST_30_DAYS','EMAIL_OPEN_COUNT_LAST_30_DAYS') ]
                    thirty_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_30_DAYS','LOGIN_COUNT_LAST_30_DAYS')]
                    thirty_days_col_list_2 = [x for x in num_cols if x  in ('GB_COUNT_LAST_30_DAYS')]
                    sixty_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_60_DAYS','LOGIN_COUNT_LAST_60_DAYS','GB_COUNT_LAST_60_DAYS') ]
                    sixty_days_col_list_2 = [x for x in num_cols if x  in ('EMAIL_SEND_COUNT_LAST_60_DAYS','EMAIL_OPEN_COUNT_LAST_60_DAYS') ]
                    ninety_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_90_DAYS','LOGIN_COUNT_LAST_90_DAYS','GB_COUNT_LAST_90_DAYS') ]
                    ninety_days_col_list_2 = [x for x in num_cols if x  in ('EMAIL_SEND_COUNT_LAST_90_DAYS','EMAIL_OPEN_COUNT_LAST_90_DAYS') ]
                    onetwenty_days_col_list = [x for x in num_cols if x  in ('IRU_COUNT_LAST_120_DAYS')]
                    onetwenty_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_120_DAYS','GB_COUNT_LAST_120_DAYS')]
                    onetwenty_days_col_list_2 = [x for x in num_cols if x  in ('EMAIL_SEND_COUNT_LAST_120_DAYS','EMAIL_OPEN_COUNT_LAST_120_DAYS') ]
                    threesixtyfive_days_col_list = [x for x in num_cols if x  in ('IRU_COUNT_LAST_365_DAYS')]
                    threesixtyfive_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_365_DAYS','GB_COUNT_LAST_365_DAYS')]
                    threesixtyfive_days_col_list_2 = [x for x in num_cols if x  in ('EMAIL_SEND_COUNT_LAST_365_DAYS','EMAIL_OPEN_COUNT_LAST_365_DAYS') ]
                    seventhirty_days_col_list = [x for x in num_cols if x  in ('IRU_COUNT_LAST_730_DAYS')]
                    seventhirty_days_col_list_1 = [x for x in num_cols if x  in ('EMAIL_CLICK_COUNT_LAST_730_DAYS','GB_COUNT_LAST_730_DAYS')]
                    seventhirty_days_col_list_2 = [x for x in num_cols if x  in ('EMAIL_SEND_COUNT_LAST_730_DAYS','EMAIL_OPEN_COUNT_LAST_730_DAYS') ]
                    Lifetime_days_col_list = [x for x in num_cols if x  in ('LIFETIME_IRU_TAGS','LIFETIME_HINOTE_COUNT')]
                    Lifetime_days_col_list_1 = [x for x in num_cols if x  in ('LIFETIME_EMAIL_CLICK_COUNT','LIFETIME_COUNT_OF_GBS')]
                    Lifetime_days_col_list_2 = [x for x in num_cols if x  in ('LIFETIME_EMAIL_SEND_COUNT','LIFETIME_EMAIL_OPEN_COUNT') ]
                    PHOTO_cols= [x for x in num_cols if x == 'PHOTO_COUNT' ]
                    School_size_cols= [x for x in num_cols if x == 'SCHOOL_SIZE_RATIO' ]
                    # MAX_GRAD_YEAR_col = [x for x in num_cols if x  in ('MAX_GRAD_YEAR')]
                    non_skewed_col_list=[x for x in num_cols if x  in ('DAYS_SINCE_LAST_LOGIN')]
        
                    features = [CLASS_SIZE_col,AGE_col, thirty_days_col_list, thirty_days_col_list_1, thirty_days_col_list_2,
                               sixty_days_col_list_1, sixty_days_col_list_2, ninety_days_col_list_1,ninety_days_col_list_2,
                               onetwenty_days_col_list, onetwenty_days_col_list_1, onetwenty_days_col_list_2,
                               threesixtyfive_days_col_list, threesixtyfive_days_col_list_1, threesixtyfive_days_col_list_2,
                               seventhirty_days_col_list, seventhirty_days_col_list_1, seventhirty_days_col_list_2,
                               Lifetime_days_col_list, Lifetime_days_col_list_1, Lifetime_days_col_list_2,
                               PHOTO_cols,School_size_cols, non_skewed_col_list]
                    bivariate_cols = [col for x in features for col in x]
    
                    x = st.selectbox("Select Category to view distribution", bivariate_cols)
                    
                    if(x):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        pdf = data.groupBy(x, "CLICKED_OR_NOT").count().toPandas()
                        # st.write(len(pdf))
                        bins=[0,100, 200, 300, 400, 500, 600, 700, 800]
                        if x in ('AGE'):
                            bins=[0,10,20,30,40,50,60,70,80,90,100]
                        if x in ('EMAIL_CLICK_COUNT_LAST_30_DAYS','EMAIL_CLICK_COUNT_LAST_60_DAYS','EMAIL_CLICK_COUNT_LAST_90_DAYS','EMAIL_CLICK_COUNT_LAST_120_DAYS',
                                    'EMAIL_CLICK_COUNT_LAST_365_DAYS','EMAIL_CLICK_COUNT_LAST_730_DAYS','PHOTO_COUNT',
                                    'LOGIN_COUNT_LAST_60_DAYS', 'LOGIN_COUNT_LAST_90_DAYS','EMAIL_OPEN_COUNT_LAST_30_DAYS','EMAIL_OPEN_COUNT_LAST_60_DAYS','LOGIN_COUNT_LAST_60_DAYS',
                                    'EMAIL_SEND_COUNT_LAST_30_DAYS','GB_COUNT_LAST_30_DAYS','GB_COUNT_LAST_60_DAYS','EMAIL_SEND_COUNT_LAST_60_DAYS',
                                'GB_COUNT_LAST_90_DAYS', 'IRU_COUNT_LAST_120_DAYS','IRU_COUNT_LAST_365_DAYS','IRU_COUNT_LAST_730_DAYS'):
                             bins = [0,2,4,6,8,10,12,14,16,18,20,22]
                        if x in ('EMAIL_OPEN_COUNT_LAST_90_DAYS','EMAIL_OPEN_COUNT_LAST_120_DAYS',
                                    'EMAIL_SEND_COUNT_LAST_90_DAYS', 'GB_COUNT_LAST_730_DAYS', 'LIFETIME_IRU_TAGS',
                                'LIFETIME_HINOTE_COUNT','LIFETIME_EMAIL_CLICK_COUNT'):
                            bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120, 150]
                        if x in ( 'LOGIN_COUNT','LIFETIME_EMAIL_SEND_COUNT'):
                            bins=[0,50,100,200,300,400,500,600,700,800,900,1000,1500,2000]
                        if x in ( 'LIFETIME_EMAIL_SEND_COUNT','DAYS_SINCE_LAST_LOGIN'):
                            bins=[0,250,500,750,1000,1250,1500, 1750,2000, 2250, 2500, 2750]
                            
                            
                        bars = sns.histplot(data=pdf, x=x, hue="CLICKED_OR_NOT", bins=bins, multiple='stack', weights='COUNT')
                        sns.move_legend(ax, "upper right")
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))                       
                        bars_height=0
                        for bar in bars.patches:
                            bars_height+=bar.get_height()                   
                        for bar in bars.patches:
                            yval = bar.get_height()
                            percentage = f'{yval / bars_height * 100:.1f}%' 
                            ax.text(bar.get_x() + bar.get_width() / 2, yval +(bars_height * 0.02), percentage, ha='center', va='center', fontsize=10)
                        
                        total_counts = pdf.groupby("CLICKED_OR_NOT")["COUNT"].sum()
                        percentages = (total_counts / total_counts.sum()) * 100            
                        legend_labels = bars.legend_.texts
                        for label, percentage in zip(legend_labels, percentages):
                            label.set_text(f"{label.get_text()} ({percentage:.1f}%)")
                                
                        ax.set_title(f'Bivariate Analysis of {x} with target variable Clicked_Or_Not_Clicked')
                        ax.set_xlabel(f'{x}')
                        ax.set_ylabel('Count')
                        st.pyplot(plt)

#----------------------------------------------------------------------------------------------------------------                           

                if st.checkbox("**Categorical Features** (Bivariate)"):
                    st.subheader('BIVARIATE ANALYSIS - Categorical Features')

                    x = st.selectbox("Select Categories to view distribution", ['EMAIL_DOMAIN', 'GENDER', 'REUNION_YEAR', 'DNER_FLAG', 
                                                                               'COMMERCIAL_IND', 'ACQUISITION_SOURCE', 'MEMBERSHIP_STATUS', 'MEMBERSHIP_STATUS_HISTORY',
                                                                               'GB_MOMENTUM', 'EMAIL_DOMAIN_GROUP', 'WEEKLY_DIGEST_IND', 'MY_PROFILE_VISITS_IND', 
                                                                               'MY_REMINDER_IND', 'SCHOOL_COMMUNITY_IND', 'MY_REMEMBERS_IND', 'MY_INBOX_IND', 
                                                                               'MY_PROFILE_NOTES_IND', 'MY_PRIVATE_MESSAGES_IND', 'SCHOOL_PROFILE_IND', 
                                                                               'SCHOOL_REMINDER_IND', 'SCHOOL_YEARBOOK_IND', 'NEW_CLASSMATES_FEATURES_IND', 
                                                                               'DO_NOT_EMAIL_IND', 'AGE_GROUP', 'SCHOOL_NAME', 'SCHOOL_CITY', 'SCHOOL_STATE',
                                                                               'PUBLISHER_OWNER_NAME', 'PUBLISHER_NAME', 'LOGIN_COUNT_BUCKETS', 
                                                                               'UPLOADED_PHOTOS_OR_NOT', 'TAGGED_IN_YEARBOOK_OR_NOT', 'REUNION_INVITEE', 
                                                                               'FACEBOOK_TOKEN_EXPIRED', 'FACEBOOK_TOKEN_AVAILABLE'])
    
                    data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))
                    
                # Assuming old_with_target is a Spark DataFrame and string_cols_v2 is a list of column names
                    if(x):
                        a = data.groupBy(x, "CLICKED_OR_NOT").count().toPandas()
                        pivot_data = a.pivot_table(index=x, columns='CLICKED_OR_NOT', values='COUNT', fill_value=0).reset_index()
                        pivot_data.columns = [x] + [f'CLICKED_{val}' for val in pivot_data.columns[1:]]
                    
                        # Select the top 10 rows
                        top10 = pivot_data.sort_values(by=[f'CLICKED_1', f'CLICKED_0'], ascending=[False, False]).head(10)
                    
                        # Melt the pivot table to long format for seaborn
                        top10_melted = top10.melt(id_vars=[x], value_vars=[f'CLICKED_0', f'CLICKED_1'], var_name='Clicked_Status', value_name='Count')
                    
                        fig, ax = plt.subplots(figsize=(10, 6))
                    
                        bars = sns.histplot(data=top10_melted, x=x, weights='Count', hue='Clicked_Status', multiple='stack', bins=len(top10[x].unique()))
                        sns.move_legend(ax, "upper right")
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                    
                        total_counts = top10[[f'CLICKED_1', f'CLICKED_0']].sum().sum()
                    
                        for bar in bars.patches:
                            yval = bar.get_height()
                            percentage = f'{yval / total_counts * 100:.1f}%'
                            ax.text(bar.get_x() + bar.get_width() / 2, yval, percentage, ha='center', va='bottom', fontsize=10)
                    
                        total_counts_per_status = top10[[f'CLICKED_1', f'CLICKED_0']].sum()
                        percentages = (total_counts_per_status / total_counts_per_status.sum()) * 100
                    
                        if bars.legend_:
                            legend_labels = bars.legend_.texts
                            for label, percentage in zip(legend_labels, percentages):
                                label.set_text(f"{label.get_text()} ({percentage:.1f}%)")
                    
                        ax.set_xticklabels(top10[x].values, rotation=45, ha='right')
                    
                        ax.set_title(f'Bivariate Analysis of {x} with target variable Clicked_Or_Not_Clicked')
                        ax.set_xlabel(f'{x}')
                        ax.set_ylabel('Count')
                        plt.tight_layout()
                        st.pyplot(plt)
                

#------------------------- 3. MULTIVARIATE ANALYSIS ----------------------------------------------------------------------------------------            

            if st.checkbox("Multivariate Analysis"):
                st.subheader("MULTIVARIATE ANALYSIS")

                option = st.selectbox("Select Category to view distribution", ['Emails Sent  vs  Click-through Rate  vs  Age Group',
                                         'Emails Sent  vs  Click-through Rate  vs  Gender',
                                                'Emails Sent  vs  Click-through Rate  vs  EMAIL_DOMAIN  vs  GENDER',
                                                'Emails Sent  vs  Click-through Rate  vs  AGE  vs  GENDER'
                                                ])
                
                
#------------------------------ Emails Sent  vs  Click-through Rate  vs  Age Group ------------------------------------------------------------
                
                if(option == 'Emails Sent  vs  Click-through Rate  vs  Age Group'):
                    # data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))
                    data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))
                    
                    df_plot = data.groupBy("AGE").agg({"EMAIL_SEND_COUNT_LAST_30_DAYS": "sum","CLICKED_OR_NOT": "sum"}).toPandas()
                    # Rename the columns for clarity
                    df_plot.columns = ['AGE', 'EMAIL_SEND_COUNT_LAST_30_DAYS', 'CLICKED_OR_NOT']
                    
                    # Ensure columns are numeric
                    df_plot['AGE'] = pd.to_numeric(df_plot['AGE'], errors='coerce')
                    df_plot['EMAIL_SEND_COUNT_LAST_30_DAYS'] = pd.to_numeric(df_plot['EMAIL_SEND_COUNT_LAST_30_DAYS'], errors='coerce')
                    df_plot['CLICKED_OR_NOT'] = pd.to_numeric(df_plot['CLICKED_OR_NOT'], errors='coerce')
                    
                    # Define bins and labels for age groups
                    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
                    df_plot['AGE_BIN'] = pd.cut(df_plot['AGE'], bins=bins, labels=labels, include_lowest=True)
                    
                    # Calculate total email counts and click counts per age group
                    total_emails = df_plot.groupby('AGE_BIN')['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum().reset_index()
                    total_clicks = df_plot.groupby('AGE_BIN')['CLICKED_OR_NOT'].sum().reset_index()
                    
                    # Calculate total emails and total clicks across all age groups
                    total_emails_sent = total_emails['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum()
                    
                    # Merge the DataFrames
                    combined_df = pd.merge(total_emails, total_clicks, on='AGE_BIN')
                    
                    # Calculate percentage of emails sent and CTR
                    combined_df['EMAILS_SENT_PERCENT'] = np.where(
                        total_emails_sent > 0, 
                        (combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] / total_emails_sent) * 100, 
                        0
                    )
                    combined_df['CTR'] = np.where(
                        combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] > 0, 
                        (combined_df['CLICKED_OR_NOT'] / combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS']) * 100, 
                        0 
                    )
                    
                    # Plot the data
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Plot email sent percentage
                    bars = ax1.bar(combined_df['AGE_BIN'], combined_df['EMAILS_SENT_PERCENT'], color='b', alpha=0.6, label='Emails Sent (%)')
                    ax1.set_xlabel('Age Group')
                    ax1.set_ylabel('Emails Sent (%)', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    # Add annotations for email sent percentage
                    for bar, percent in zip(bars, combined_df['EMAILS_SENT_PERCENT']):
                        ax1.annotate(f"{percent:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')
                    
                    # Create a second y-axis to plot CTR
                    ax2 = ax1.twinx()
                    ax2.plot(combined_df['AGE_BIN'], combined_df['CTR'], color='r', marker='o', linestyle='-', label='Click-through Rate (%)')
                    ax2.set_ylabel('Click-through Rate (%)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    # Add annotations for CTR
                    for x, y, ctr in zip(range(len(combined_df)), combined_df['CTR'], combined_df['CTR']):
                        ax2.annotate(f"{ctr:.1f}%", xy=(x, y), xytext=(7, 0), textcoords="offset points", va='center')
                    
                    # Title and legend
                    plt.title('Emails Sent  vs  Click-through Rate  vs  Age Group')
                    fig.tight_layout()
                    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

                    st.pyplot(plt)


#------------------------------ Emails Sent  vs  Click-through Rate  vs  Gender ------------------------------------------------------------
                
                if(option == 'Emails Sent  vs  Click-through Rate  vs  Gender'):
                    data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))

                    df = data.groupBy("GENDER").agg({"EMAIL_SEND_COUNT_LAST_30_DAYS": "sum","CLICKED_OR_NOT": "sum"}).toPandas()
                    # Rename the columns for clarity
                    df.columns = ['GENDER', 'EMAIL_SEND_COUNT_LAST_30_DAYS', 'CLICKED_OR_NOT']
                    
                    # Ensure columns are numeric
                    df['EMAIL_SEND_COUNT_LAST_30_DAYS'] = pd.to_numeric(df['EMAIL_SEND_COUNT_LAST_30_DAYS'], errors='coerce')
                    df['CLICKED_OR_NOT_INT'] = pd.to_numeric(df['CLICKED_OR_NOT'], errors='coerce')
                    
                    # Calculate total email counts and click counts per gender
                    total_emails = df.groupby('GENDER')['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum().reset_index()
                    total_clicks = df.groupby('GENDER')['CLICKED_OR_NOT'].sum().reset_index()
                    
                    # Calculate total emails sent across all genders
                    total_emails_sent = total_emails['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum()
                    
                    # Merge the DataFrames
                    combined_df = pd.merge(total_emails, total_clicks, on='GENDER')
                    #combined_df = pd.merge(combined_df, total_count, on='GENDER')
                    
                    # Calculate percentage of emails sent and CTR
                    combined_df['EMAILS_SENT_PERCENT'] = np.where(
                        total_emails_sent > 0, 
                        (combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] / total_emails_sent) * 100, 
                        0
                    )
                    combined_df['CTR'] = np.where(
                        combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] > 0, 
                        (combined_df['CLICKED_OR_NOT'] / combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS']) * 100, 
                        0
                    )
                    
                    # Plot the data
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Plot email sent percentage
                    bars = ax1.bar(combined_df['GENDER'], combined_df['EMAILS_SENT_PERCENT'], color='b', alpha=0.6, label='Emails Sent (%)')
                    ax1.set_xlabel('GENDER')
                    ax1.set_ylabel('Emails Sent (%)', color='b')
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    # Add annotations for email sent percentage
                    for bar, percent in zip(bars, combined_df['EMAILS_SENT_PERCENT']):
                        ax1.annotate(f"{percent:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')
                    
                    # Create a second y-axis to plot CTR
                    ax2 = ax1.twinx()
                    ax2.plot(combined_df['GENDER'], combined_df['CTR'], color='r', marker='o', linestyle='-', label='Click-through Rate (%)')
                    ax2.set_ylabel('Click-through Rate (%)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    # Add annotations for CTR
                    for x, y, ctr in zip(range(len(combined_df)), combined_df['CTR'], combined_df['CTR']):
                        ax2.annotate(f"{ctr:.1f}%", xy=(x, y), xytext=(7, 0), textcoords="offset points", va='center')
                    
                    # Title and legend
                    plt.title('Emails Sent  vs  Click-through Rate  vs  Gender')
                    fig.tight_layout()
                    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
                    st.pyplot(plt)
            

#------------------------ Emails Sent  vs  Click-through Rate  vs  EMAIL_DOMAIN  vs  GENDER ------------------------------------------------------------
                
                if(option == 'Emails Sent  vs  Click-through Rate  vs  EMAIL_DOMAIN  vs  GENDER'):               
                    data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))
                    
                    x = data.groupBy("EMAIL_DOMAIN", "GENDER").agg(
                        {"EMAIL_SEND_COUNT_LAST_30_DAYS": "sum", "CLICKED_OR_NOT": "sum"}
                    ).toPandas()
                    
                    # Rename the columns for clarity
                    x.columns = ['EMAIL_DOMAIN', 'GENDER', 'EMAIL_SEND_COUNT_LAST_30_DAYS', 'CLICKED_OR_NOT']
                    
                    # Ensure numeric columns are properly formatted
                    x['EMAIL_SEND_COUNT_LAST_30_DAYS'] = pd.to_numeric(x['EMAIL_SEND_COUNT_LAST_30_DAYS'], errors='coerce')
                    x['CLICKED_OR_NOT'] = pd.to_numeric(x['CLICKED_OR_NOT'], errors='coerce')
                    
                    total_emails_per_domain = x.groupby('EMAIL_DOMAIN')['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum().reset_index()
                    
                    # Sort and select the top 10 SCHOOL_IDs by total email send count
                    top_domains = total_emails_per_domain.sort_values(by='EMAIL_SEND_COUNT_LAST_30_DAYS', ascending=False).head(10)
                    top_domain_ids = top_domains['EMAIL_DOMAIN']
                    
                    # Filter the original DataFrame to include only the top SCHOOL_IDs
                    df = x[x['EMAIL_DOMAIN'].isin(top_domain_ids)]
                    
                    # Calculate total email counts and click counts per gender group
                    total_emails = df.groupby(['EMAIL_DOMAIN', 'GENDER'])['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum().reset_index()
                    total_clicks = df.groupby(['EMAIL_DOMAIN', 'GENDER'])['CLICKED_OR_NOT'].sum().reset_index()
                    
                    # Calculate total emails sent across all domains
                    total_emails_sent = total_emails['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum()
                    
                    # Merge the DataFrames on both EMAIL_DOMAIN and GENDER
                    combined_df = pd.merge(total_emails, total_clicks, on=['EMAIL_DOMAIN', 'GENDER'])
                    
                    # Calculate percentage of emails sent and CTR
                    combined_df['EMAILS_SENT_PERCENT'] = np.where(
                        total_emails_sent > 0,
                        (combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] / total_emails_sent) * 100,
                        0
                    )
                    combined_df['CTR'] = np.where(
                        combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] > 0,
                        (combined_df['CLICKED_OR_NOT'] / combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS']) * 100,
                        0
                    )
                    
                    # Plotting
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Use lowercase keys for gender_colors and gender_labels to avoid KeyErrors
                    gender_colors = {'m': 'b', 'f': 'g', 'others': 'r'}
                    gender_labels = {'m': 'Male', 'f': 'Female', 'others': 'Other'}
                    
                    # Convert gender values to lowercase for consistency
                    combined_df['GENDER'] = combined_df['GENDER'].str.lower()
                    
                    # Plot email sent percentage for each gender category within each email domain
                    bar_width = 0.3
                    domain_list = combined_df['EMAIL_DOMAIN'].unique()
                    gender_list = combined_df['GENDER'].unique()
                    index = np.arange(len(domain_list))
                    
                    # Prepare a dictionary to store the positions for line plotting
                    line_positions = {gender: [] for gender in gender_list}
                    line_ctrs = {gender: [] for gender in gender_list}
                    
                    for i, domain in enumerate(domain_list):
                        domain_df = combined_df[combined_df['EMAIL_DOMAIN'] == domain]
                        bar_positions = [i - bar_width + j * bar_width for j in range(len(gender_list))]
                        for j, gender in enumerate(gender_list):
                            gender_df = domain_df[domain_df['GENDER'] == gender]
                            if not gender_df.empty:
                                bars = ax1.bar(bar_positions[j], gender_df['EMAILS_SENT_PERCENT'], bar_width, 
                                               label=f"EMAIL_SENT_{gender_labels[gender]}_%", color=f"{gender_colors[gender]}")
                                ax1.bar_label(bars, labels=[f"{value:.1f}%" for value in gender_df['EMAILS_SENT_PERCENT']], padding=2, fontsize=8)
                                line_positions[gender].append(bar_positions[j])
                                line_ctrs[gender].append(gender_df['CTR'].values[0])
                    
                    ax1.set_xlabel('EMAIL_DOMAIN')
                    ax1.set_ylabel('Emails Sent (%)', color='b')
                    ax1.set_xticks(index)
                    ax1.set_xticklabels(domain_list, rotation=45, ha='right')
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    # Plot CTR for each gender category within each email domain
                    ax2 = ax1.twinx()
                    for gender in gender_list:
                        if line_positions[gender]:
                            line, = ax2.plot(line_positions[gender], line_ctrs[gender], 'o-', color=gender_colors[gender], markersize=5, label=f"CLICK_THRU_RATE_{gender_labels[gender]}_%")
                            for x, y, ctr in zip(line_positions[gender], line_ctrs[gender], line_ctrs[gender]):
                                ax2.annotate(f"{ctr:.1f}%", xy=(x, y), xytext=(7, 0), textcoords="offset points", ha='center', va='center', fontsize=8)
                    
                    ax2.set_ylabel('Click-through Rate (%)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    # Combine legends
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    # Create a dictionary to hold the correct label for each handle
                    label_dict = {f"EMAIL_SENT_{gender_labels[gender]}_%": handles1[j] for j, gender in enumerate(gender_list)}
                    label_dict.update({f"CLICK_THRU_RATE_{gender_labels[gender]}_%": handles2[j] for j, gender in enumerate(gender_list)})
                    
                    # Sort the labels and handles so they appear correctly in the legend
                    sorted_labels = sorted(label_dict.keys())
                    sorted_handles = [label_dict[label] for label in sorted_labels]
                    
                    # Update the legend
                    ax1.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
                    
                    
                    plt.title('Emails Sent  vs  Click-through Rate  vs  EMAIL_DOMAIN  vs  GENDER')
                    plt.tight_layout()
                    # plt.show()
                    st.pyplot(plt)

                
#------------------------ Emails Sent  vs  Click-through Rate  vs  AGE  vs  GENDER ------------------------------------------------------------

                if(option == 'Emails Sent  vs  Click-through Rate  vs  AGE  vs  GENDER'):               
                    data = data.withColumn('CLICKED_OR_NOT', when(data.CLICKED_OR_NOT == 'False', 0).otherwise(1))
                    df = data.groupBy("AGE", "GENDER").agg(
                        {"EMAIL_SEND_COUNT_LAST_30_DAYS": "sum", "CLICKED_OR_NOT": "sum"}
                    ).toPandas()
                    
                    # Rename the columns for clarity
                    df.columns = ['AGE', 'GENDER', 'EMAIL_SEND_COUNT_LAST_30_DAYS', 'CLICKED_OR_NOT']
                    
                    # Ensure numeric columns are properly formatted
                    df['EMAIL_SEND_COUNT_LAST_30_DAYS'] = pd.to_numeric(df['EMAIL_SEND_COUNT_LAST_30_DAYS'], errors='coerce')
                    df['CLICKED_OR_NOT'] = pd.to_numeric(df['CLICKED_OR_NOT'], errors='coerce')
                    
                    # Define bins and labels for age groups
                    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
                    df['AGE_BIN'] = pd.cut(df['AGE'], bins=bins, labels=labels, include_lowest=True)
                    
                    # Calculate total email counts and click counts per gender group
                    total_emails = df.groupby(['AGE_BIN', 'GENDER'])['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum().reset_index()
                    total_clicks = df.groupby(['AGE_BIN', 'GENDER'])['CLICKED_OR_NOT'].sum().reset_index()
                    
                    # Calculate total emails sent across all age groups
                    total_emails_sent = total_emails['EMAIL_SEND_COUNT_LAST_30_DAYS'].sum()
                    
                    # Merge the DataFrames on both AGE_BIN and GENDER
                    combined_df = pd.merge(total_emails, total_clicks, on=['AGE_BIN', 'GENDER'])
                    
                    # Calculate percentage of emails sent and CTR
                    combined_df['EMAILS_SENT_PERCENT'] = np.where(
                        total_emails_sent > 0,
                        (combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] / total_emails_sent) * 100,
                        0
                    )
                    combined_df['CTR'] = np.where(
                        combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS'] > 0,
                        (combined_df['CLICKED_OR_NOT'] / combined_df['EMAIL_SEND_COUNT_LAST_30_DAYS']) * 100,
                        0
                    )
                    
                    # Plotting
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Define colors for each gender category
                    gender_colors = {'m': 'b', 'f': 'g', 'others': 'r'}
                    gender_labels = {'m': 'Male', 'f': 'Female', 'others': 'Other'}
                    
                    # Convert gender values to lowercase for consistency
                    combined_df['GENDER'] = combined_df['GENDER'].str.lower()
                    
                    # Convert gender_list to a Python list for the index method
                    gender_list = combined_df['GENDER'].unique().tolist()
                    
                    # Plot email sent percentage for each gender category within each age group
                    bar_width = 0.3
                    age_bins = combined_df['AGE_BIN'].unique()
                    index = np.arange(len(age_bins))
                    
                    # Prepare a dictionary to store the positions for line plotting
                    line_positions = {gender: [] for gender in gender_list}
                    line_ctrs = {gender: [] for gender in gender_list}
                    
                    for i, age_bin in enumerate(age_bins):
                        age_bin_df = combined_df[combined_df['AGE_BIN'] == age_bin]
                        bar_positions = [i - bar_width + j * bar_width for j in range(len(gender_list))]
                        for j, gender in enumerate(gender_list):
                            gender_df = age_bin_df[age_bin_df['GENDER'] == gender]
                            if not gender_df.empty:
                                bars = ax1.bar(bar_positions[j], gender_df['EMAILS_SENT_PERCENT'], bar_width,
                                               label=f"EMAIL_SENT_{gender_labels[gender]}_%", color=gender_colors[gender])
                                ax1.bar_label(bars, labels=[f"{value:.1f}%" for value in gender_df['EMAILS_SENT_PERCENT']], padding=2, fontsize=8)
                                line_positions[gender].append(bar_positions[j])
                                line_ctrs[gender].append(gender_df['CTR'].values[0])
                    
                    ax1.set_xlabel('AGE')
                    ax1.set_ylabel('Emails Sent (%)', color='b')
                    ax1.set_xticks(index)
                    ax1.set_xticklabels(age_bins, rotation=45, ha='right')
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    # Plot CTR for each gender category within each age group
                    ax2 = ax1.twinx()
                    for gender in gender_list:
                        if line_positions[gender]:
                            line, = ax2.plot(line_positions[gender], line_ctrs[gender], 'o-', color=gender_colors[gender], markersize=5,
                                             label=f"CLICK_THRU_RATE_{gender_labels[gender]}_%")
                            for x, y, ctr in zip(line_positions[gender], line_ctrs[gender], line_ctrs[gender]):
                                ax2.annotate(f"{ctr:.1f}%", xy=(x, y), xytext=(0, 5),
                                             textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
                    ax2.set_ylabel('Click-through Rate (%)', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    # Combine legends
                    handles1, labels1 = ax1.get_legend_handles_labels()
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    
                    # Create a dictionary to hold the correct label for each handle
                    label_dict = {f"EMAIL_SENT_{gender_labels[gender]}_%": handles1[j] for j, gender in enumerate(gender_list)}
                    label_dict.update({f"CLICK_THRU_RATE_{gender_labels[gender]}_%": handles2[j] for j, gender in enumerate(gender_list)})
                    
                    # Sort the labels and handles so they appear correctly in the legend
                    sorted_labels = sorted(label_dict.keys())
                    sorted_handles = [label_dict[label] for label in sorted_labels]
                    
                    # Update the legend
                    ax1.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.05, 1))
                    
                    plt.title('Emails Sent  vs  Click-through Rate  vs  AGE  vs  GENDER')
                    plt.tight_layout()
                    # plt.show()
                    st.pyplot(plt)

                
    with tabs[1]: 
        ## CODE FOR DATABASE SELECTION ######
        result_df1=session.sql(f"""SHOW DATABASES""").collect()
        database_df=pd.DataFrame(result_df1)  
        DATABASE_NAME=database_df['name'].to_list()
        db=st.selectbox(options=[None] + DATABASE_NAME, index=0,label='**SELECT DATABASE**')
        
        ## CODE FOR SCHEMA SELECTION ######
        if db is None:
            schema=st.selectbox(options=[None] ,label='**SELECT SCHEMA NAME**')
        else:
            result_df2=session.sql(f"""show schemas in database {db}""").collect()
            schema_df=pd.DataFrame(result_df2)
            SCHEMA_NAME=schema_df['name'].to_list()
            schema=st.selectbox(options= SCHEMA_NAME,label='**SELECT SCHEMA NAME**')
            if "." in schema :
                st.write(':red[**Please enter a valid schema name**]')

        ## CODE FOR TABLE SELECTION ######
        if db is None:
            table=st.selectbox(options=[None] ,label='**SELECT TABLE NAME**')
        elif "." in schema:
                table=None
        else:
            result_df3=session.sql(f"""show tables in schema {db}.{schema}""").collect()  
            table_df=pd.DataFrame(result_df3)
            if table_df.empty:
                st.write(':red[**No tables inside schema**]')
                table=None   
            else:
                TABLE_NAME=table_df['name'].to_list()
                table=st.selectbox(options=TABLE_NAME ,label='**SELECT TABLE NAME**')
        
    
        session.file.get('@"KIPI"."ADMIN"."MODEL_STAGE"/Final_Model_WITHOUT_CAT_FEAT','/tmp')
        model = load_model('/tmp/Final_Model_WITHOUT_CAT_FEAT')
        if db == 'KIPI' and schema == 'ADMIN' and table == 'PRED_FINAL':
            # result_df2=session.sql(f"""select distinct PRED_PROB_BUCKET_KIPI from {db}.{schema}.{table};""").collect()
            # buckets=pd.DataFrame(result_df2)
            # #st.write(campaign_df)
            # BUCKET_LIST=buckets['PRED_PROB_BUCKET_KIPI'].to_list()
            PROBABILITY_BUCKET=st.selectbox(options= [None] +  ['01_PERCENT' ,'01_TO_05_PERCENT' , '06_TO_10_PERCENT','11_TO_30_PERCENT' ,'31_TO_50_PERCENT','51_TO_70_PERCENT','71_TO_90_PERCENT','91_TO_95_PERCENT','95_TO_100_PERCENT'], index=0,label='**PLEASE SELECT PREDICTION PROBABILITY BUCKET**')

            if PROBABILITY_BUCKET != None : 
                df = session.sql(f"""select * from {db}.{schema}.{table}""")
                df = df.withColumn('PRED_PROB_BUCKET',
                                when(((col('PREDICTION_SCORE')>=0)&(col('PREDICTION_SCORE')<=0.01)), '01_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.01)&(col('PREDICTION_SCORE')<=0.05)), '01_TO_05_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.05)&(col('PREDICTION_SCORE')<=0.1)), '06_TO_10_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.1)&(col('PREDICTION_SCORE')<=0.3)), '11_TO_30_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.3)&(col('PREDICTION_SCORE')<=0.5)), '31_TO_50_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.5)&(col('PREDICTION_SCORE')<=0.7)), '51_TO_70_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.7)&(col('PREDICTION_SCORE')<=0.9)), '71_TO_90_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.9)&(col('PREDICTION_SCORE')<=0.95)), '91_TO_95_PERCENT')\
                               .when(((col('PREDICTION_SCORE')>0.95)&(col('PREDICTION_SCORE')<=1)), '95_TO_100_PERCENT'))
                                
                bucket_data = df.filter(col('PRED_PROB_BUCKET')==PROBABILITY_BUCKET).orderBy('PREDICTION_SCORE').collect()
                #bucket_data = session.sql(f"""select * from df where PRED_PROB_BUCKET = '{PROBABILITY_BUCKET}'  order by prediction_score_1  desc; """ ).collect()
                bucket_data_df=pd.DataFrame(bucket_data)
                st.write('**Top 5 Users with High Prediction Value**')
                st.write(bucket_data_df.head())
                st.write('**Bottom 5 Users with Low Prediction Value**')
                st.write(bucket_data_df.tail())
            REGISTRATION_NUM = st.text_input("**PLEASE ENTER REGISTRATION ID YOU WANT TO CHECK THE PREDICTION FOR**")
            if REGISTRATION_NUM:
                data = session.sql(f"""select REGISTRATION_ID ,PREDICTION_SCORE from {db}.{schema}.{table} where REGISTRATION_ID = {REGISTRATION_NUM}  ; """ ).collect()
                data_df=pd.DataFrame(data)
                if data_df.empty:
                    st.write(':red[**No records for the selected REGSITRATION_ID**]')  
                else:
                    st.write(data_df)
                    final_data = session.sql(f"""select * from {db}.{schema}.{table} where REGISTRATION_ID = {REGISTRATION_NUM} ; """ ).collect()

                    final_data_df=pd.DataFrame(final_data)
                    #st.write(final_data_df)
                    
                    
                    one = final_data_df.drop(['REGISTRATION_ID','PREDICTION_LABEL','CLICKED_OR_NOT'],axis=1)
                    pipe = model[:-1].transform(one.drop('PREDICTION_SCORE',axis=1))
                    
                    try:
                        explainer = shap.Explainer(model.named_steps["trained_model"])
                        shap_values = explainer.shap_values(pipe)
                    except Exception as e:
                        st.error(f"Error creating SHAP explainer or computing SHAP values: {e}")
                        return
                    try:
                        st.subheader('Local Explainability')
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, pipe,  show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting SHAP values: {e}")
            else:
                st.write("Please enter valid REGISTRATION_ID")
        else:
            st.write("Please select the valid DATABASE, SCHEMA, TABLE where prediction table is saved")


    with tabs[2]:
    # Explanation tab    

        st.subheader('Global Explanability')
        session.file.get('@"KIPI"."ADMIN"."MODEL_STAGE"/Feature Importance (All).png','/tmp')
        st.image('/tmp/Feature Importance (All).png')
        

if __name__ == '__main__':
    dataframe = EDA_Dataframe_Analysis()
    info = Attribute_Information()
    session = get_active_session()
    plot_charts = Charts()
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 19:25:15 2025

@author: spinosaurus777
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import regex as re

#%%

def get_and_read_data() -> pd.DataFrame:
    file_url = "XXXXXXXXXXXXX/survey-data.csv"
    df = pd.read_csv(file_url)
    pd.set_option('display.max_columns', None) 
    print(df.head())
    return df

def compesation_distribution(df:pd.DataFrame) -> None:
    df = df.dropna(subset="ConvertedCompYearly")
    print(df["ConvertedCompYearly"].describe())
    sns.histplot(data=df,x="ConvertedCompYearly")
    plt.show()
    
def median_compesation_for_full_time(df:pd.DataFrame) -> None:
    df_employed=df[df["Employment"]=="Employed, full-time"] 
    print(df_employed.describe())
    print('The medan compesation for full-time employees is: ', 
          df_employed["ConvertedCompYearly"].median())
    
def compesation_range_and_dsitribution_by_count(df:pd.DataFrame) -> None:
    df = df.dropna(subset="ConvertedCompYearly")
    # Just top 10 countries
    df_top10 = df.groupby("Country")["ConvertedCompYearly"].mean().nlargest(10)
    df_top10 = df_top10.sort_values(ascending=False) 
    countries = df_top10.index
    df_top10_comp = df[(df["Country"].isin(countries)) & (df["ConvertedCompYearly"]<8000000)]
    print(df_top10_comp.describe())
    plt.figure(figsize = (15, 8))
    sns.boxplot(data=df_top10_comp, x="Country", y="ConvertedCompYearly")
    plt.xticks(rotation=90)
    plt.show()
    
def remove_outlier_ConvertedComp(df:pd.DataFrame)->pd.DataFrame:
    df=df.dropna(subset=["ConvertedCompYearly","WorkExp", "JobSatPoints_1"])
    stats=df.describe()
    IQR_value=(stats.loc["75%","ConvertedCompYearly"])-(stats.loc["25%","ConvertedCompYearly"]) 
    whisker_up=(IQR_value*1.5)+(stats.loc["75%","ConvertedCompYearly"])
    whisker_down=(stats.loc["75%","ConvertedCompYearly"])-(IQR_value*1.5)
    df_filtered=df[(df["ConvertedCompYearly"]<=whisker_up) & (df["ConvertedCompYearly"]>=whisker_down)]
    return df_filtered

def correlation_between_variables(df_filtered:pd.DataFrame) -> None:
    df_corr = df_filtered.corr(numeric_only=True)
    df_corr = df_corr.loc[["ConvertedCompYearly","WorkExp","JobSatPoints_1"], ["ConvertedCompYearly","WorkExp","JobSatPoints_1"]]
    sns.heatmap(data=df_corr)
    plt.show()
    
def programming_languages_trends(df:pd.DataFrame) -> pd.DataFrame:
    df=df.dropna(subset=["LanguageHaveWorkedWith","LanguageWantToWorkWith"])
    df=df.reset_index()
    list_lan=df["LanguageHaveWorkedWith"].unique().tolist()

    languages=[]
    for item in list_lan:
        words=re.split(";",item) 
        for word in words:
            if word not in languages: 
                languages.append(word) 
    print(languages) 

    count = np.zeros(len(languages))
    def count_languages(df_lan, languages):
        df_lan = re.split(";",df_lan) 
        for language in languages:
            count_by_language = 0
            for word in df_lan:
                if language == word:
                    count_by_language+=1
            count[languages.index(language)]=count_by_language 
        return count

    count_worked_lan=np.zeros(len(languages)) 
    count_want_to_work_lan=np.zeros(len(languages))
    # Overral count in the data frame
    for i in np.arange(0,df.shape[0]-1,1):
        count_each_worked=count_languages(df.loc[i,"LanguageHaveWorkedWith"],languages)
        count_worked_lan=np.add(count_worked_lan,count_each_worked) 
        count_each_want_to_work=count_languages(df.loc[i,"LanguageWantToWorkWith"],languages)
        count_want_to_work_lan=np.add(count_want_to_work_lan,count_each_want_to_work) 

    print(count_worked_lan)
    print(count_want_to_work_lan)

    df_languages_count = pd.DataFrame({"Languages":languages,"CountLanguagesWorkedWith":count_worked_lan,"CountLanguagesToWorkWith":count_want_to_work_lan})
    df_languages_count = df_languages_count.sort_values(by=["CountLanguagesWorkedWith"],ascending=False) 
    df_languages_count = df_languages_count.set_index("Languages") 

    # Graph
    df_languages_count.plot(kind='bar', stacked=True, figsize=(15,8))
    plt.xticks(rotation=90)
    
    return df_languages_count
    
def coefficient_progeamming_languages(df_languages_count:pd.DataFrame) -> pd.DataFrame:
    df_languages_count = pd.DataFrame({"Database":languages,"CountDatabaseWorkedWith":count_worked_lan,"CountDatabaseToWorkWith":count_want_to_work_lan})
    df_languages_count = df_languages_count.sort_values(by=["CountDatabaseWorkedWith"],ascending=False) 
    df_languages_count = df_languages_count.set_index("Database") 
    df_languages_count["Coefficient"] = df_languages_count["CountDatabaseToWorkWith"]/df_languages_count["CountDatabaseWorkedWith"] 
    df_languages_count = df_languages_count.sort_values(by=["Coefficient"],ascending=False) 

    

def remote_work_trends_by_country(df:pd.DataFrame) -> None:
    df = df.dropna(subset="RemoteWork")
    df.isna().sum()
    df_remotework_by_country=df[["RemoteWork", "Country"]]
    work_types = df_remotework_by_country["RemoteWork"].unique()

    dfs=[0,0,0]
    for i in range(0,3,1):
        dfs[i] = df_remotework_by_country[df_remotework_by_country["RemoteWork"]==work_types[i]]
        dfs[i] = dfs[i]["Country"].value_counts().to_frame().rename(columns={"count":work_types[i]})


    df_rw_by_c = pd.concat(dfs,axis=1)
    df_rw_by_c = df_rw_by_c.fillna(0)
    df_rw_by_c["Total"] = df_rw_by_c.agg("sum", axis="columns")
    df_top10 = df_rw_by_c.nlargest(10,"Total")


    df_top10.index.name = "RemoteWork"
    df_top10 = df_top10.transpose()
    
    # Graph
    # Just for the top 10 countries
    df_top10.plot(kind='bar', stacked=False, figsize=(15,8))
    plt.show()
    
def crosstab_edlevel_emplyment(df:pd.DataFrame) -> None:
    df=df.dropna(subset=["Employment","EdLevel"])
    cross=pd.crosstab(df.Employment,df.EdLevel)
    print(cross)


df_languages_count=pd.DataFrame({"Database":languages,"CountDatabaseWorkedWith":count_worked_lan,"CountDatabaseToWorkWith":count_want_to_work_lan})
df_languages_count=df_languages_count.sort_values(by=["CountDatabaseWorkedWith"],ascending=False) 
df_languages_count=df_languages_count.set_index("Database") 
df_languages_count["Coefficient"]=df_languages_count["CountDatabaseToWorkWith"]/df_languages_count["CountDatabaseWorkedWith"] 
df_languages_count=df_languages_count.sort_values(by=["Coefficient"],ascending=False) 

    
    

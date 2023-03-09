import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
import numpy as np
Data Cleaning
path = r'C:\Users\PutuAndika\OneDrive - Migo\Desktop\Data Analyst Project\Netflix'
df = pd.read_csv(os.path.join(path, 'netflix_titles.csv'))
df.head()
df.info()
for i in df.columns:
    null_values = round(df[i].isnull().sum()/len(df)*100,2)
    if null_values>0:
        print(f'Null Values in {i} column: {null_values}%')
df['date_added'] = pd.to_datetime(df['date_added'])
df.info()
#Handling Missing Values
df[['director', 'cast']] = df[['director', 'cast']].fillna("No Data")
df['country'] = df.country.fillna(df.country.mode()[0])
df = df.dropna()
df.isnull().sum()
df.type.unique()
#Movie Category Netflix
type = df.groupby(['type'])['title'].nunique().reset_index().rename(columns={'title':'Number of Movies'})
px.pie(type, values = 'Number of Movies', names ='type',title='Netflix Video Category')
Time Series Analysis of Movie Type
#Analyzing Month
df['Month'] = df['date_added'].dt.month
df['Year'] = df['date_added'].dt.year
df_month_movie = df.groupby(['Year'])['title'].count().reset_index().rename(columns={'title':'Number of Movie'})
px.line(df_month_movie, x = 'Year', y = 'Number of Movie', title = 'Number of Movies Added per Year', 
        labels={'date_added':'Date Added'})
#Number of Movie by Category
df_month_type_movie = df.groupby(['Year','type'])['title'].count().reset_index().rename(columns={'title':'Number of Movie'})
px.area(df_month_type_movie, x = 'Year', y = 'Number of Movie', color='type',line_group = 'type',title = 'Number of Movies Added per Year by Movue Type', 
        labels={'date_added':'Date Added'})
month_type = df.pivot_table(index = 'Month', values = 'title', columns='type', aggfunc='count').reset_index()
month_type
px.area(month_type, x = 'Month', y=['Movie', 'TV Show'], title = 'Number of Movie Type by Month')
month_all = df.groupby(['Month'])['title'].count().reset_index().rename(columns={'title':'Number of Title'})
px.pie(month_all, values='Number of Title', names = 'Month', title='Slicing of Movie Numbers by Month')
Country Analysis
df.country.unique()
df['new_country'] = df.country.apply(lambda x : x.split(",")[0])
df.new_country.unique()
#Top 10 Country With Most Netflix
country = df.groupby(['new_country'])['title'].count().reset_index().rename(columns = {'title':'Number of Movies'})\
    .sort_values(by = ['Number of Movies'], ascending= False).head(10)
px.bar(country, x='new_country', y='Number of Movies', title= 'Top 10 Country With Netflix')
#Movie Type by Country
country_movie = df.pivot_table(index= 'new_country', values='title' ,columns='type', aggfunc='count').reset_index()
country_movie['Total'] = country_movie['TV Show'] + country_movie['Movie']
country_movie = country_movie.sort_values(by = ['Total'], ascending= False).head(10)
px.bar(country_movie,x = 'new_country',y= df.type.unique(), title='Movie Type by Country')
#Analyzing Date Added By Country
country_added  =df.groupby(['new_country','type'])['date_added'].nunique().reset_index()\
    .rename(columns={'date_added':'Number of Times Added'}).sort_values(by= ['Number of Times Added'], ascending=False).head(10)
px.bar(country_added, x = 'new_country', y= 'Number of Times Added',color='type', labels={'new_country':'Country'}, title='Number of Times Movie Added')
Duration of Movies
df.duration.unique()
df_movie = df.query('type == "Movie"')
df_movie['duration'] = df_movie.duration.str.replace("min","").str.strip()
df_movie['duration'] = df_movie['duration'].astype('int64')
df_tvshow = df.query('type == "TV Show"')
df_tvshow['duration'] = df_tvshow.duration.str.replace("Seasons","").str.replace("Season","").str.strip()
df_tvshow['duration'] = df_tvshow['duration'].astype('int64')
df_movie.duration.unique()
#Movie Type Category by Duration
def movie_duration(a):
    if a<=40:
        return "Short Film"
    else:
        return "Feature Film"
df_movie['duration_category'] = df_movie['duration'].apply(lambda x: movie_duration(x))
movie_category = df_movie.groupby('duration_category')['title'].count().reset_index().rename(columns={'title':'Number of Movies'})
px.pie(movie_category, values='Number of Movies', names='duration_category', title= 'Movie Duration')
df_tvshow.duration.unique()
def season_duration(a):
    if a <5:
        return "Under 5 Seasons"
    elif a>=5 and a<10:
        return "5 to 10 Seasons"
    else:
        return "More than 10 seasons"
df_tvshow['season_category'] = df_tvshow.duration.apply(lambda x: season_duration(x))
tv_show = df_tvshow.groupby(['season_category'])['title'].count().reset_index().rename(columns={'title':'Number of Movies'})
px.pie(tv_show, values='Number of Movies', names= 'season_category', title='TV Show Duration')
Movie Category Analysis
df['listed_in'].unique()
df['Category'] = df.listed_in.str.replace(", ",",").str.replace(" ,",",").str.strip().str.split(",")
df.head()
mlb = MultiLabelBinarizer()
df2 = pd.DataFrame(mlb.fit_transform(df['Category']), columns=mlb.classes_, index=df['Category'].index)
mlb.fit_transform(df['Category'])
df2.head()
mlb.fit_transform(df['Category'])
mlb.classes_
corr = df2.corr()
plt.figure(figsize = (20,5))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, cmap='BrBG')
Rating Analysis
df.rating.unique()
def rating_class(a):
    if a=='G' or a=='TV-G' or a=='TV-Y':
        return 'Kids'
    elif a=='PG' or a=='TV-Y7' or a=='TV-Y7-FV' or a=='TV-PG':
        return 'Older Kids (7+)'
    elif a=='PG-13':
        return 'Teens (13+)'
    elif a=='TV-14':
        return 'Young Adults (16+)'
    else:
        return 'Adults (18+)'
df['rating_category'] = df['rating'].apply(lambda x: rating_class(x))
rating = df.groupby(['rating_category'])['title'].count().reset_index().rename(columns={'title':'Number of Movies'})
px.pie(rating, values='Number of Movies', names='rating_category', title = 'Target Age of Netflix Videos')
#Rating by Country
rating_country = df.pivot_table(index='new_country', columns='rating_category', values='title', aggfunc='count').reset_index()
rating_country['Total'] = rating_country[df.rating_category.unique()].sum(axis=1)
rating_country = rating_country.sort_values(by = ['Total'], ascending= False).head(10)
px.bar(rating_country, x = 'new_country', y = df.rating_category.unique(), title='Rating Category by Country')
rating_year = df.pivot_table(index='Year', columns='rating_category', values='title', aggfunc='count').reset_index()
px.area(rating_year, x = 'Year', y=df.rating_category.unique(), title='Movie Added by Target Age per Year')

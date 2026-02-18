# Learnings Journal

## February 18 2026

Today I learned about the different dataframe functions to inspect the data such as .head(), .loc[].
I also learned how to turn missing values or NaN into 'None' values. The purpose being that turning it into a 0
could represent that in the context of Sleep Disorders or insomnia, that it is less than insomnia or sleep apnea instead of
just being 'healthy' and not having sleep apnea or insomnia at all. 
### Command
First creating a copy using df_clean = df.copy()
Command using df_clean['Column Name'] = df_clean['Column Name'].fillna('None') 

I also gained insight on the query function which is similiar to SQL and trying to filter the data by certain parameters
such as age, df.query("Age > 30")
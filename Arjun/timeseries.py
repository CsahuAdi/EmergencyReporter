import pandas as pd

df = pd.read_csv('../seattle.csv')  # Load your dataset here
df.head()

#convert datatime into proper fomat and make it index
df['Date'] = pd.to_datetime(df['Datetime'])
df.set_index('Date', inplace=True)

df.head()

df.info()
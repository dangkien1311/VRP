import pandas as pd
file_path = 'input.txt' 
df = pd.read_csv(file_path, delimiter='\s+')  
print(df.info())
df['CUST_NO.'] = df.index
df.to_csv('solomon_data.txt', index=False)


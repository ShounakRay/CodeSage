import pandas as pd

# create a sample dataframe
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
    'C': [[4, 5], [6, 7, 8], [9]]
})

# explode the list column
exploded_df = df.explode('C')

# create new columns for each element in the list
new_columns = exploded_df['C'].apply(pd.Series)
new_columns = new_columns.rename(columns=lambda x: f'C_{x+1}')

# concatenate the new columns to the original dataframe
new_df = pd.concat([exploded_df.drop('C', axis=1), new_columns], axis=1)

print(new_df)

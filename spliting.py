'''import pandas as pd
#from pathlib import Path
from sklearn.model_selection import train_test_split

#df = pd.read_csv(r'e:\Edu\SAMK\2\AI Project 1\Final Project\Group_6\Data\CSVs\dataset.csv')
df = pd.read_csv('/mnt/e/Edu/SAMK/2/AI Project 1/Final Project/Group_6/Data/CSVs/dataset.csv')

train_data, val_data = train_test_split(
    df,
    test_size=0.3,
)
#train_data.to_csv(r'e:\Edu\SAMK\2\AI Project 1\Final Project\Group_6\Data\CSVs\train_df.csv', index=False)
#val_data.to_csv(r'e:\Edu\SAMK\2\AI Project 1\Final Project\Group_6\Data\CSVs\val_df.csv', index=False)

train_data.to_csv('/mnt/e/Edu/SAMK/2/AI Project 1/Final Project/Group_6/Data/CSVs/train_df.csv', index=False)
val_data.to_csv('/mnt/e/Edu/SAMK/2/AI Project 1/Final Project/Group_6/Data/CSVs/val_df.csv', index=False)

print("Done")'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Get script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset
csv_path = os.path.join(base_dir, 'CSVs', 'dataset.csv')
df = pd.read_csv(csv_path)

# Split dataset
train_data, val_data = train_test_split(
    df,
    test_size=0.3,
    random_state=42
)

# Save files
train_path = os.path.join(base_dir, 'CSVs', 'train_df.csv')
val_path = os.path.join(base_dir, 'CSVs', 'val_df.csv')

train_data.to_csv(train_path, index=False)
val_data.to_csv(val_path, index=False)

print("Done")

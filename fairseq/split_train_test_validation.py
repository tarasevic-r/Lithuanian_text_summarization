# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:54:29 2021

@author: user2
"""
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

en_data = []
jp_data = []

with open('C:/Users/user2/Downloads/WikiMatrix.es-lt.tsv') as fp:
    for line in tqdm(fp, total=3895992):
        line_data = line.rstrip().split('\t')
        en_data.append(line_data[1] + '\n')
        jp_data.append(line_data[2] + '\n')

total_test = 7000
en_train, en_subtotal, jp_train, jp_subtotal = train_test_split(
        en_data, jp_data, test_size=total_test, random_state=42)

en_test, en_val, jp_test, jp_val = train_test_split(
        en_subtotal, jp_subtotal, test_size=0.5, random_state=42)

file_mapping = {
    'train.en_XX': en_train,
    'train.ja_XX': jp_train,
    'valid.en_XX': en_val,
    'valid.ja_XX': jp_val,
    'test.en_XX': en_test,
    'test.ja_XX': jp_test,

}

for k, v in file_mapping.items():
    with open(f'preprocessed/{k}', 'w') as fp:
        fp.writelines(v)
        
        
import pandas as pd

# d = pd.read_csv("C:/Users/user2/git/python/NN/Fairseq/preprocessed/validation.csv", sep=',')

# d1 = d.loc[:5]
# d2 = d.loc[5:]

# d2.to_csv("C:/Users/user2/git/python/NN/Fairseq/preprocessed/val.csv", sep =',')

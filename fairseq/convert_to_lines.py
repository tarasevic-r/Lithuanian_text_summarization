# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:38:39 2021

@author: user2
"""


import sentencepiece as spm
import os
import pandas as pd


# Convert documents to Fairseq format with sentencepiece


BASE = "C:/Users/user2/git/python/NN/Fairseq/"
MODEL="{}mbart.cc25.v2/sentence.bpe.model".format(BASE)
DATA="{}preprocessed".format(BASE)

train_path = "{}/train.csv".format(DATA)
test_path = "{}/test.csv".format(DATA)
val_path = "{}/val.csv".format(DATA)


SRC="en_XX"
TGT="ja_XX"

out_dir="C:/Users/user2/git/python/NN/Fairseq/postprocessed"

processor = spm.SentencePieceProcessor()
processor.Load(MODEL)

source_suffix='bpe.source'
target_suffix='bpe.target'


train_text_file = os.path.join(out_dir, "train.{}".format(source_suffix))
train_summary_file = os.path.join(out_dir, "train.{}".format(target_suffix))
val_text_file = os.path.join(out_dir, "val.{}".format(source_suffix))
val_summary_file = os.path.join(out_dir, "val.{}".format(target_suffix))
test_text_file = os.path.join(out_dir, "test.{}".format(source_suffix))
test_summary_file = os.path.join(out_dir, "test.{}".format(target_suffix))


files = ((train_path, train_text_file, train_summary_file),
         (val_path, val_text_file, val_summary_file),
         (test_path, test_text_file, test_summary_file))



lowercase=False
max_text_subwords=None
max_summary_subwords=None
insert_tags=True


for path, text_file_name, summary_file_name in files:
    with open(text_file_name, "w") as text_file, open(summary_file_name, "w") as summary_file:
        d = pd.read_csv("C:/Users/user2/git/python/NN/Fairseq/preprocessed/train.csv", sep=',') 
        df = zip(d['text'].to_list(), d['summary'].to_list())
        for text, summary in df:
            if lowercase:
                text = text.lower()
                summary = summary.lower()
            text_subwords = processor.EncodeAsPieces(text)
            if max_text_subwords:
                text_subwords = text_subwords[:max_text_subwords]
            summary_subwords = processor.EncodeAsPieces(summary)
            if max_summary_subwords:
                summary_subwords = summary_subwords[:max_summary_subwords]
            if insert_tags:
                text_subwords.insert(0, "<t>")
                text_subwords.append("</t>")
                summary_subwords.insert(0, "<t>")
                summary_subwords.append("</t>")
            text_file.write(" ".join(text_subwords) + "\n")
            summary_file.write((" ".join(summary_subwords)) + "\n")
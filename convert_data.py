#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:56:08 2020

@author: rahul
"""

import sys
sys.path.append('..')

import pandas as pd

def convert_train(input_utterance, labels):
    split_labels = labels.split('[')
    slot_tuple = []
    for slot_val in split_labels:
        slot_val = slot_val.replace(']','')
        if slot_val.startswith('sl'):
            slot_tokens = slot_val.split()
            slot_name = slot_tokens[0]
            slot_value = slot_tokens[1:]
            slot_tuple.append((slot_name, ' '.join(slot_value)))
    output = []
    i = 0
    for slot_name, slot_value in slot_tuple:
        input_tokens = input_utterance.split()
        slot_tokens = slot_value.split()
        while i < len(input_tokens):
            if slot_tokens[0] == input_tokens[i]:
                if ' '.join(input_tokens[i:i+len(slot_tokens)]).lower() == slot_value:
                    text_to_append = '[' + slot_name + ' ' + slot_value + ' ]'
                    output.append(text_to_append)
                    i += len(slot_tokens)
                    break
            output.append(input_tokens[i])
            i+=1
    
    output = ' '.join(output)
    output = '[' + split_labels[1] + output + ']'  
    return output

def preprocess_data(train_data):
    
    train_data['schema2'] = train_data.schema
    for i in range(len(train_data.schema)):
        train_data.schema[i] = "[IN:" + train_data.schema[i]
        train_data.schema[i] = train_data.schema[i].replace(",", " ]")
        train_data.schema[i] = train_data.schema[i].replace(")"," ]")
        train_data.schema[i] = train_data.schema[i].replace("@", "[SL:")
        train_data.schema[i] = train_data.schema[i].replace("(", " ")
        train_data.tokens[i] = train_data.tokens[i].replace("?", " ?")
        train_data.tokens[i] = train_data.tokens[i].replace("'", " '")
        train_data.tokens[i] = train_data.tokens[i].replace("$", "$ ")
        train_data.tokens[i] = train_data.tokens[i].replace("%", " %")
        train_data.tokens[i] = train_data.tokens[i].replace("-", " - ")
        train_data.tokens[i] = train_data.tokens[i].replace("=", " = ")    
        train_data.schema[i] = train_data.schema[i].replace("=", " ")
        train_data.schema[i] = train_data.schema[i] + "]"
        train_data.tokens[i] = train_data.tokens[i].lower()
        train_data.schema2[i] = convert_train(train_data.tokens[i].lower(), train_data.schema[i].lower())
        for token in train_data.schema2[i].split(" "):
            if token[:3] in ["[in", "[sl"]:
                train_data.schema2[i] = train_data.schema2[i].replace(token,token.upper())
        train_data.schema = train_data.schema2
    del train_data['schema2']

    train_data.to_csv("data/top-dataset-semantic-parsing/train.tsv", sep = '\t', index = False, header=None)
    train_data.to_csv("data/top-dataset-semantic-parsing/test.tsv", sep = '\t', index = False, header=None)
    train_data.to_csv("data/top-dataset-semantic-parsing/eval.tsv", sep = '\t', index = False, header=None)
    return train_data

train_data = pd.read_table('../data/top-dataset-semantic-parsing/sanju_data.tsv', names=['text', 'tokens', 'schema'])
preprocess_data(train_data)

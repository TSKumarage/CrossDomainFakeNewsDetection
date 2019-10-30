"""
Authors : Kumarage Tharindu & Paras Sheth
Class : CSE 472
Organization : ASU CIDSE
Project : SMM Project 2
Task : File reader : Provider other package the access to the text data

"""

import os
import numpy as np
import pandas as pd


def write_data(file_name, header, data, type='numpy'):

    rel_path = get_data_repo_path(file_name)
    if type == 'pandas':
        data.to_csv(rel_path, header=header, index=False)
    elif type == 'numpy':
        header_str = ""
        for item in header:
            header_str += item+","
        header_str = header_str.rstrip(',')
        np.savetxt(rel_path, data, delimiter=",", header=header_str)


def read_data(file_name, type='pandas'):

    rel_path = get_data_repo_path(file_name)
    data = pd.read_csv(rel_path, delimiter='\t')

    if type == 'pandas':
        return data
    if type == 'numpy':
        return data.values


def get_data_repo_path(file_name):
    code_dir = os.path.dirname(__file__)  # absolute dir

    rel_path = os.path.join(code_dir, "text_data", file_name+".tsv")

    return rel_path




import time
import datetime
import os
import json
import re
import sys

import pandas as pd
import numpy as np

# Get the current working directory
current_dir = os.getcwd()

# Assuming your desired project_root is one level up from the current directory
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.append(project_root)

from src.modules.Embedding.text_embedder import TextProcessor
tp = TextProcessor('../openai_key.txt')


def main():
    # load data
    df = pd.read_csv('data/raw/startups_scraped.csv')
    df = df.rename(columns={'external_ref': 'url', 'page_text': 'text'}).dropna(subset='text')

    # normalize text before summarization
    df = tp.normalize_column(df=df, column='text', new_column='processed_text')
    df = tp.remove_stopwords_column(df=df, column='processed_text', new_column='processed_text')

    # get summaries
    summary_error_log = []
    df['summary'] = None
    for index, row in df.iterrows():
        try:
            df.at[index, 'summary'] = tp.get_page_text_summary(row['processed_text'])
            print('Completed Summary:', index)
        except:
            df.at[index, 'summary'] = '-'
            print('FAILED SUMMARY:', index)
            summary_error_log.append({'index': index})

    # get embeddings
    embedding_error_log = []
    df['embeddings'] = None
    for index, row in df.iterrows():
        try:
            df.at[index, 'embeddings'] = tp.get_embedding(row['summary'])
            print('Completed Embedding:', index)
        except:
            df.at[index, 'embeddings'] = None
            print('FAILED EMBEDDING:', index)
            embedding_error_log.append({'index': index})

    # export data
    df.to_parquet('data/processed/startups_processed.parquet')
    if len(summary_error_log) > 0:
        summary_error_log = pd.DataFrame(summary_error_log)
        summary_error_log.to_csv('data/logs/summary_error_log.csv', index=False)
    if len(embedding_error_log) > 0:
        embedding_error_log = pd.DataFrame(embedding_error_log)
        embedding_error_log.to_csv('data/logs/embedding_error_log.csv', index=False)

if __name__ == "__main__":
    main()
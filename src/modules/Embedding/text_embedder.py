import json
import requests
import re
import string
import pandas as pd
import numpy as np
from openai import OpenAI
import tiktoken
import sys
import os

# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    def __init__(self, api_key_path=None, api_key=None):
        """
        Accept either a path to a key file (legacy) or a raw API key string.
        Falls back to the OPENAI_API_KEY environment variable if neither is given.
        """
        self.api_key_path = api_key_path
        self._api_key = api_key
        self.client = self.init_client()
        self.encoding = tiktoken.get_encoding("cl100k_base")
        # self.nlp = spacy.load('en_core_web_sm')

    def init_client(self):
        if self._api_key:
            key = self._api_key
        elif self.api_key_path:
            with open(self.api_key_path, 'r') as file:
                key = file.read().strip()
        else:
            key = os.environ['OPENAI_API_KEY']
        return OpenAI(api_key=key)
    
    def chunk_text(self, text, chunk_size, overlap):
        # Convert chunk_size to int if it's not
        chunk_size = int(chunk_size)
        overlap = int(overlap)

        # Convert non-string input to string
        text = str(text) if not pd.isna(text) else ''

        # Ensure chunk size is smaller than total length of text
        if chunk_size > len(text):
            return [text]

        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def chunk_dataframe(self, df, chunk_size, overlap):
        df_list = []
        for _, row in df.iterrows():
            chunks = self.chunk_text(row['text'], chunk_size, overlap)
            temp_data = []
            for chunk in chunks:
                new_row = row.to_dict()
                new_row['text'] = chunk
                temp_data.append(new_row)
            temp_df = pd.DataFrame(temp_data)
            df_list.append(temp_df)
        final_df = pd.concat(df_list, ignore_index=True)
        return final_df

    def reduce_tokens(self, processed_text, token_limit=7500):
        """ 
        while number of tokens exceeds token_limit, 
        this will pop 5 words until token_limit no longer exceeded
        """
        i = len(processed_text.split())
        while len(self.encoding.encode(processed_text)) > token_limit:
            i -= 5
            processed_text = processed_text.split()[:i]
            processed_text = ' '.join(processed_text)
        return processed_text

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """gets embedding for text using chatgpt api"""
        text = self.reduce_tokens(text)
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
    
    def get_description_pages(self, toc):
        """this uses chatgpt to extract the range of pages that encompass the funding description"""

        # convert table of contents to string
        toc = str(toc)

        # create prompt for chatgpt
        system_content = """
        I used PyMuPDF to extract the table of contents from a grant program pdf. I will show you the table of contents
        Tell me the start page and end page for extracting the description of the grant.
        Your answer will use this json format: '{"start_page": number, "end_page": number}'.
        Your answer will be used directly in a python script, so do not include any other text or whitespace characters.
        """
        user_content = 'Here is the table of contents: ' + toc

        # get api key
        with open(self.api_key_path, 'r') as file:
            api_key = file.read().strip()

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }


        data = {
            'model': 'gpt-4-1106-preview',
            'messages': [
                {
                    'role': 'system',
                    'content': system_content
                },
                {
                    'role': 'user',
                    'content': user_content
                }
            ]

        }

        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            data=json.dumps(data)
        )

        content = response.json()

        content = content['choices'][0]['message']['content']

        # sometimes chatgpt tack these on
        content = re.sub('\n', '', content)
        content = re.sub('json', '', content)
        content = re.sub('```', '', content)

        extraction_pages = json.loads(content)

        return extraction_pages
    
    # def extract_pdf_text(self, pdf_document):
    #     """Extracts Text from PDF"""
        
    #     # get table of contents
    #     toc = pdf_document.get_toc()

    #     # raw text list
    #     raw_text = []

    #     # sometimes pdfs are short (5-10 pages) so it does not include ToC
    #     if toc:
    #         # get page range that encompasses funding description
    #         extraction_pages = self.get_description_pages(toc)

    #         print('EXTRACTION PAGES:', extraction_pages)

    #         # using page numbers from chatgpt response, extract text from pdf
    #         for page_number in range(extraction_pages['start_page'], extraction_pages['end_page']):
    #             page = pdf_document.load_page(page_number - 1)
    #             raw_text.append(page.get_text())
    #         raw_text = '\n'.join(raw_text)
    #     else:
    #         print('NO TABLE OF CONTENTS')
    #         for page in pdf_document:
    #             raw_text.append(page.get_text())
    #         raw_text = '\n'.join(raw_text)
    #     return raw_text
    
    def remove_duplicate_lines(self, text):
        return (pd.Series(text.split('\n'))
                .drop_duplicates()
                .to_list())

    def normalize_text(self, text):
        text = ''.join(text)
        text = text.lower()
        text = ''.join([char if char not in '“”.!?,—():-/;–•' else ' ' for char in text])
        text = ''.join([char for char in text if char not in string.punctuation])
        text = re.sub(r'\s{2,}', ' ', text)
        return text
    
    def normalize_column(self, df, column, new_column):
        """normalizes text in a column using vecotrized operations"""
        special_chars = '“”.,!?—():/;–•-'
        df[new_column] = (df[column]
                          .astype(str).str.lower()
                          .str.replace(f'[{special_chars}]+', ' ', regex=True)
                          .str.replace(f'[{string.punctuation}]+', '', regex=True)
                          .str.replace(r'\s{2,}', ' ', regex=True)
                          )
        return df

    def tokenize_and_remove_stopwords(self, text):
        """removes stopwords from text"""
        return [word for word in text.split(' ') if word not in stopwords.words('english')]

    def remove_stopwords_column(self, df, column, new_column):
        """removes stopwords text in a column using vectorized operations"""
        df[new_column] = df[column].str.split(' ')
        df = (df
            .explode(column=new_column)
            .query(f'~{column}.isin(@stopwords.words("english"))')
            .groupby('url', as_index=False)[column]
            .agg({new_column: ' '.join})
            )
        return df
    
    def lemmatize_tokens(self, tokens):
        doc = self.nlp(' '.join(tokens))
        tokens = [token.lemma_ for token in doc]
        tokens = [token for token in tokens if not token.isdigit() and len(token) > 1]
        return tokens

    def generate_tokens(self, text):
        text = self.remove_duplicate_lines(text)
        text = self.normalize_text(text)
        tokens = self.tokenize_and_remove_stopwords(text)
        tokens = self.lemmatize_tokens(tokens)
        return tokens

    def generate_ngrams(self, tokens, num_ngrams):
        ngrams = zip(*[tokens[i:] for i in range(num_ngrams)])
        ngrams = [" ".join(ngram) for ngram in ngrams]
        ngrams = set(ngrams)
        ngrams = list(ngrams)
        return ngrams
    
    def get_page_text_summary(self, page_text):
        page_text = self.reduce_tokens(page_text)
        system_content = """"
        This is the page text scraped off a startup website. Pull out a 1-2 paragraph summary. 
        Be sure to highlight key technologies, methodologies, etc. 
        If the text has issues such that you cannot process it, just return '-' as your answer.
        If there is not enough information in the scraped text, just return '-' as your answer.
        """
        user_content = f'Extract from the following page text: [{page_text}]'
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            # response_format={ "type": "json_object" },
            messages=[
                {"role": "system", 
                "content": system_content},
                {"role": "user", 
                "content": user_content}
            ]
            )
        # return json.loads(completion.choices[0].message.content)
        return completion.choices[0].message.content
    
    def get_grant_summary(grant_description):
        pass

# Main guard (optional)
if __name__ == "__main__":
    # Example of using the class
    tp = TextProcessor()
    example_text = "Your example text here"
    embedding = tp.get_embedding(example_text)
    print(embedding)

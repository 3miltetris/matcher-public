import os
import pandas as pd
import numpy as np
import openai
import asyncio
from anthropic import Anthropic, RateLimitError, AsyncAnthropic, InternalServerError


######### LOAD KEYS & CLIENTS #########
keys_path = '/content/drive/MyDrive/keys'
with open(f'{keys_path}/anthropic.txt', 'r') as file:
    anthropic_key = file.read().strip()
with open(f'{keys_path}/openai_key.txt', 'r') as file:
    openai_key = file.read().strip()

class InsightGrant:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.anth_client_async = AsyncAnthropic(api_key=anthropic_key) 

    def openai_analysis(self, abstract, grant_summary):
        system = "Is this abstract well-aligned with this grant?"
        text = f"company summary: {abstract}\n\nAbstract: {grant_summary}"
        completion = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                "content": system},
                {"role": "user",
                    "content": f'{text}'}
            ])
        return completion.choices[0].message.content
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        """gets embedding for text using OpenAI api"""
        return self.openai_client.embeddings.create(input=[text], model=model).data[0].embedding
    
    async def analyze_matches_anthropic_only(self, matches, abstract, model="claude-3-5-sonnet-20240620", initial_max_retries=5):
        matches['good_match'] = None

        async def process_match(index, row, abstract):
            system = "Tell me if this company summary and grant summary are aligned. Only give a one-word answer. Either \"yes\" or \"no\"."
            text = f"company summary: {abstract}\n\ngrant summary: {row['description']}"

            attempt = 0
            while True:
                try:
                    message = await self.anth_client_async.messages.create(
                        model=model,
                        max_tokens=10,
                        temperature=0,
                        system=system,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": text
                                    }
                                ]
                            }
                        ]
                    )

                    matches.at[index, 'good_match'] = message.content[0].text

                    print('Complete:', index, message.content[0].text, f'{len(matches) - index} matches remaining')
                    return  # Success, exit the retry loop

                except (RateLimitError, InternalServerError) as e:
                    attempt += 1
                    wait_time = min(600, (2 ** attempt) + random.random())  # Cap at 10 minutes
                    print(f"Error occurred: {type(e).__name__}. Attempt {attempt}. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)

                    if attempt >= initial_max_retries:
                        print(f"Warning: Exceeded initial max retries ({initial_max_retries}). Continuing with exponential backoff.")

        # Process matches in smaller batches
        batch_size = 100  # Adjust based on your needs
        for i in range(0, len(matches), batch_size):
            batch = matches.iloc[i:i+batch_size]
            await asyncio.gather(*(process_match(index, row, abstract) for index, row in batch.iterrows()))
            await asyncio.sleep(2)  # Delay between batches

        return matches

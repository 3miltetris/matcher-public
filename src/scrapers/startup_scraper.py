import time
import datetime
import os
import json
import re
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Thread

import pandas as pd
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.modules.WebScraper.web_scraper import WebScraper

def thread_safe_append(lock, data, url):
    try:
        scraper = WebScraper()
        text = scraper.get_about_text(url)
        with lock:
            data.append({
                'url': url,
                'text': text
            })
    except:
        scraper.driver.quit()

def main():
    startups = pd.read_csv('data/raw/startups.csv')
    startups = (startups
                .rename(columns={'external_ref': 'url'})
                .drop_duplicates(subset='url')
                .drop(columns=['page_text'])
                )
    
    data = []
    urls = startups['url']
    lock = threading.Lock()
    max_workers = 20

    for i in range(0, len(urls), max_workers):
        threads = []
        for url in urls[i: i+max_workers]:
            t = Thread(target=thread_safe_append, args=(lock, data, url))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()

    df = pd.DataFrame(data)
    df.to_parquet('data/raw/startups_text.parquet', index=False)

if __name__ == '__main__':
    main()
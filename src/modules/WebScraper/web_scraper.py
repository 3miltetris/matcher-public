import time
import datetime
import os
import json
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Thread

import pandas as pd
import numpy as np

from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup

from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException

class WebScraper:

    def __init__(self):
        pass

    def init_driver(self, headless=False):
        options = Options()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--ignore-certificate-errors')  # Ignore certificate errors
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        if headless is True:
            options.add_argument('--headless')
        options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(10)
    
    def process_url(self, url):
        # prepend protocol if missing
        if 'http' not in url:
            url = 'https://' + url
        self.url = url

    def goto_url(self, url):
        self.init_driver(headless=False)
        self.process_url(url)
        self.driver.get(self.url)
        self.driver.implicitly_wait(10)

    def scrape_text(self):
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        page_text = soup.get_text()
        return page_text
    
    def get_hrefs(self):
        # get anchor tags
        anchor_tags = self.driver.find_elements(By.TAG_NAME, "a")

        # Extract href attributes
        hrefs = [tag.get_attribute('href') for tag in anchor_tags if tag.get_attribute('href')]
        
        # drop duplicate hrefs
        self.hrefs = set(hrefs)
    
    def url_parse(self):
        normalized_urls = set()
        for url in self.hrefs:
            parsed_url  = urlparse(url)
            clean_url = urlunparse(parsed_url._replace(fragment=''))
            normalized_urls.add(clean_url)
        normalized_urls = list(normalized_urls)
        return normalized_urls

    def get_internal_hrefs(self):
        self.get_hrefs()
        parsed_url = urlparse(self.url)
        hostname = parsed_url.hostname
        internal_list = [f'{self.url}', hostname.split('.')[0]]
        internal_hrefs = [href for href in self.hrefs if all(include in href for include in internal_list)]
        internal_hrefs = self.url_parse()
        internal_hrefs = set(internal_hrefs)
        internal_hrefs = list(internal_hrefs)
        self.internal_hrefs = internal_hrefs

    def get_about_hrefs(self):
        self.get_internal_hrefs()
        self.about_us_hrefs = [href for href in self.internal_hrefs if 'about' in href or 'story' in href or 'mission' in href or 'who-we' in href or 'vision' in href or 'what' in href]

    def get_about_text(self, url):
        self.goto_url(url)
        time.sleep(1)
        self.get_about_hrefs()
        text = self.scrape_text()

        for href in self.about_us_hrefs:
            self.driver.get(href)
            time.sleep(1)
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            text += self.scrape_text()
        self.text = text
        self.driver.quit()
        return text
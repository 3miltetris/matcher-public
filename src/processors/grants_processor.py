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
    # Load Data
    df = pd.read_csv('data/raw/arpa_h.csv')
"""
matcher.py — Core grant loading and cosine-similarity matching logic.

Usage
-----
from src.modules.matcher import load_grants, get_matches

load_grants(grants, data_path='../data')
matches = get_matches(threshold, grants['DOD']['topics'], contacts)
"""

import glob
import pandas as pd
import numpy as np

from src.modules.utils import extract_domain


# ---------------------------------------------------------------------------
# Grant loading
# ---------------------------------------------------------------------------

GRANT_COLUMN_RENAMES = {
    'embedding':   'embeddings',
    'description': 'grant_summary',
    'summary':     'grant_summary',
}


def load_grants(grants: dict, data_path: str = '../data') -> None:
    """
    Load processed topic parquet files into the grants dict in-place.

    For each grant where status=True, reads all parquet files from
    `{data_path}/all-topics/processed/{KEY}/`, concatenates them,
    deduplicates (ignoring the embeddings column), and stores the result
    in grants[key]['topics'].

    Parameters
    ----------
    grants : dict
        The grants config dict. Modified in-place.
    data_path : str
        Root data directory. Defaults to '../data' (relative to notebook CWD).
    """
    for key, value in grants.items():
        if not value.get('status'):
            continue

        print(f'Loading: {key}')
        files = glob.glob(f'{data_path}/all-topics/processed/{key}/*.parquet')

        if not files:
            print(f'  WARNING: No files found for {key}')
            continue

        topics_list = []
        for file in files:
            topic = pd.read_parquet(file).rename(columns=GRANT_COLUMN_RENAMES)
            topics_list.append(topic)

        topics = pd.concat(topics_list) if len(topics_list) > 1 else topics_list[0]
        topics = topics.reset_index(drop=True)

        # dedup: drop embeddings temporarily since arrays aren't hashable
        unique_index = topics.drop(columns='embeddings').drop_duplicates().index
        topics = topics.iloc[unique_index].reset_index(drop=True)

        grants[key]['topics'] = topics
        print(f'  {len(topics)} topics loaded')


# ---------------------------------------------------------------------------
# Cosine similarity matching
# ---------------------------------------------------------------------------

def get_matches(
    threshold: float,
    grant_topics: pd.DataFrame,
    contacts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all contact×grant pairs whose embedding cosine similarity exceeds `threshold`.

    Parameters
    ----------
    threshold : float
        Minimum similarity score to include a pair (e.g. 0.80).
    grant_topics : pd.DataFrame
        Topics DataFrame with an 'embeddings' column (list/array per row).
    contacts : pd.DataFrame
        Contacts DataFrame with an 'embeddings' column (list/array per row).

    Returns
    -------
    pd.DataFrame
        Merged matches DataFrame. Rows where company_url == companyWebsite domain
        are dropped to avoid self-matches.
    """
    grant_topics = grant_topics.copy()

    # compute similarity of each grant topic against all contacts
    contact_embeddings = np.stack(contacts['embeddings'])
    grant_topics['similarity_scores'] = grant_topics['embeddings'].apply(
        lambda x: np.dot(contact_embeddings, x)
    )

    # find contacts that exceed threshold for each grant topic
    grant_topics['high_score_index'] = grant_topics['similarity_scores'].apply(
        lambda scores: [i for i, score in enumerate(scores) if score > threshold]
    )

    index_df = (
        grant_topics['high_score_index']
        .explode()
        .reset_index()
        .rename(columns={'index': 'grant_index', 'high_score_index': 'contacts_index'})
        .dropna()
    )

    matched_contacts = contacts.iloc[index_df['contacts_index']].reset_index(drop=True)
    matched_grants   = grant_topics.iloc[index_df['grant_index']].reset_index(drop=True)

    matches = (
        pd.merge(matched_contacts, matched_grants, left_index=True, right_index=True)
        .rename(columns={'summary_x': 'company_summary', 'summary_y': 'grant_summary'})
        .reset_index(drop=True)
    )

    # drop self-matches (where the grant's company_url == the contact's domain)
    if 'company_url' in matches.columns and 'companyWebsite' in matches.columns:
        matches['_contact_domain'] = matches['companyWebsite'].apply(
            lambda x: extract_domain(str(x)).lower() if pd.notna(x) else ''
        )
        matches['_topic_domain'] = matches['company_url'].apply(
            lambda x: extract_domain(str(x)).lower() if pd.notna(x) else ''
        )
        matches = matches[
            (matches['_contact_domain'] != matches['_topic_domain']) |
            (matches['_topic_domain'] == '')
        ]
        matches = matches.drop(columns=['_contact_domain', '_topic_domain'])

    return matches.reset_index(drop=True)

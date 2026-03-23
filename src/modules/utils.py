"""
utils.py — General-purpose helpers used across the matcher pipeline.
"""

import re
import glob
import pandas as pd
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """Return the netloc for a URL, prepending https:// if no scheme present."""
    if not urlparse(url).scheme:
        url = "http://" + url
    return urlparse(url).netloc


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def remove_illegal_characters(text: str) -> str:
    """Strip characters that are illegal in Excel cell values."""
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    return ILLEGAL_CHARACTERS_RE.sub('', text)


def clean_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply remove_illegal_characters to every string cell in a DataFrame."""
    return df.map(lambda x: remove_illegal_characters(x) if isinstance(x, str) else x)


# ---------------------------------------------------------------------------
# Contact loading
# ---------------------------------------------------------------------------

CONTACT_COLUMN_RENAMES = {
    'website_url':   'companyWebsite',
    'Website URL':   'companyWebsite',
    'summary':       'company_summary',
    'description':   'company_summary',
    'company_name':  'companyName',
    'Company Name':  'companyName',
    'Company':       'companyName',
    'first_name':    'firstName',
    'First Name':    'firstName',
    'last_name':     'lastName',
    'Last Name':     'lastName',
    'Email':         'email',
    'Phone Number':  'phone',
}

EXCLUDE_URL_PATTERNS = r'\.edu|\.gov|\.org|\.mil'


def load_contacts(
    data_path: str,
    excluded: list[str] | None = None,
    included: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and consolidate all contact parquet files under `data_path`.

    Parameters
    ----------
    data_path : str
        Root path containing the all-contacts directory, e.g.
        '/content/drive/Shareddrives/Matcher/notebooks/pipelines/matching-contacts-grants/../data'
        or simply the resolved absolute path.
    excluded : list[str], optional
        Substrings — any file whose path contains one of these is skipped.
        Defaults to ['vc', 'funded', 'linkedin-contacts', 'sba'].
    included : list[str], optional
        If provided, only files whose path contains at least one of these substrings are loaded.
        Mutually exclusive with `excluded` (excluded takes priority when both set).

    Returns
    -------
    pd.DataFrame
        De-duplicated contacts with normalised column names, non-null embeddings,
        and non-edu/gov/org/mil websites filtered out.
    """
    if excluded is None:
        excluded = ['vc', 'funded', 'linkedin-contacts', 'sba']

    pattern = f'{data_path}/all-contacts/**/*.parquet'
    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f'No parquet files found at: {pattern}')

    contact_list = []
    for filepath in files:
        if excluded and any(excl in filepath for excl in excluded):
            continue
        if included and not any(incl in filepath for incl in included):
            continue

        temp_df = (
            pd.read_parquet(filepath)
            .rename(columns=CONTACT_COLUMN_RENAMES)
        )

        # drop page_text if present — it's bulky and not needed downstream
        temp_df = temp_df.drop(columns=[c for c in ['page_text'] if c in temp_df.columns])

        # drop stale grant-specific columns that leaked into contact files
        temp_df = temp_df.drop(columns=[c for c in temp_df.columns if 'DOD ' in c])

        contact_list.append(temp_df.reset_index(drop=True))

    if not contact_list:
        raise ValueError('No contact files matched the given filters.')

    contacts = (
        pd.concat(contact_list)
        .drop_duplicates('companyWebsite')
        .drop_duplicates('companyName')
        .dropna(subset='embeddings')
        .reset_index(drop=True)
    )

    # filter out non-commercial domains
    contacts = contacts[~contacts['companyWebsite'].str.contains(EXCLUDE_URL_PATTERNS, na=False)]

    return contacts.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Match reshaping
# ---------------------------------------------------------------------------

def transpose_contacts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Pivot a long matches DataFrame so each company has one row,
    with all its grant summaries collapsed into a single cell.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_matches() — one row per company×grant combination.
    columns : list[str]
        Extra columns from the contact side to carry through (e.g. ['email', 'firstName']).

    Returns
    -------
    pd.DataFrame
        One row per unique companyWebsite.
    """
    url_col = 'companyWebsite'
    transposed = []

    for domain in df[url_col].unique():
        temp = df.loc[df[url_col] == domain].copy().reset_index(drop=True)

        row: dict = {
            url_col:          temp[url_col].iloc[0],
            'company_name':   temp['companyName'].iloc[0],
            'company_summary': temp['company_summary'].iloc[0],
        }

        for col in columns:
            if col in temp.columns:
                row[col] = temp[col].iloc[0]

        # last grant summary wins — caller can adjust if they need all of them
        for _, grant_row in temp.iterrows():
            row['grant_summary'] = grant_row['grant_summary']

        transposed.append(pd.DataFrame([row]))

    return pd.concat(transposed).fillna('-').reset_index(drop=True)

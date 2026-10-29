"""
roadmap.matcher — load grant topics and match one company's profile against them.

Thin orchestration over the shared aspect_matching core, so a (profile, topic)
pair ranks here exactly as it would in Grant Search or Bulk Aspect Match.

The one roadmap-specific concern is latency. intake-web is a warm Cloud Run
service; loading every processed/ parquet on each webhook would blow the ~2-min
budget on its own. So the stacked topic-embedding matrix is cached in memory on
the instance and refreshed on a TTL — the match maths itself (one numpy matmul)
is instant.
"""

import io
import threading
import time

import numpy as np
import pandas as pd

import src.modules.aspect_matching as am
import src.modules.aspect_profile as ap
from src.modules.grant_utils import (add_record_kind, drop_dead_topics,
                                      normalize_grant_columns)

BUCKET        = 'cc-matcher-bucket-jeg-v1'
TOPICS_PREFIX = 'data/all-topics/processed/'

# Mirrors Grant Search's capability-profile mode defaults, so the roadmap match
# is the same match the consultants see in the app.
THRESHOLD = 0.78
MIN_HITS  = 1
TOP_K     = 25        # per-unit cap inside match_units
DECK_LIMIT = 8        # how many opportunities the deck actually shows

# In-memory topic index, refreshed at most once per TTL per instance.
TOPIC_TTL_S = 3600


def load_topic_frame(storage_client, bucket: str = BUCKET,
                     prefix: str = TOPICS_PREFIX,
                     include_procurement: bool = False) -> pd.DataFrame:
    """Every live, fundable grant topic across processed/, with broad_agency
    stamped from the folder. Awards (outside processed/) are deliberately not
    loaded — a roadmap is forward-looking."""
    frames = []
    for blob in storage_client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith('.parquet'):
            continue
        df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        agency = blob.name[len(prefix):].split('/', 1)[0]
        df['broad_agency'] = agency
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    topics = pd.concat(frames, ignore_index=True)
    topics = normalize_grant_columns(topics)          # guarantees grant_summary
    # Drop archived SAM.gov notices and rolling grants a re-check found gone.
    topics = drop_dead_topics(topics).reset_index(drop=True)
    topics = add_record_kind(topics)
    if not include_procurement and 'record_kind' in topics.columns:
        topics = topics[topics['record_kind'] != 'procurement'].reset_index(drop=True)
    return topics


class TopicIndex:
    """A stacked topic-embedding matrix + the topic rows it corresponds to."""

    def __init__(self, matrix: np.ndarray, meta: pd.DataFrame, built_at: float):
        self.matrix   = matrix
        self.meta     = meta
        self.built_at = built_at

    @property
    def n_topics(self) -> int:
        return int(self.matrix.shape[0])


_cache_lock = threading.Lock()
_cache: dict = {'index': None, 'at': 0.0}


def build_topic_index(storage_client, bucket: str = BUCKET,
                      prefix: str = TOPICS_PREFIX) -> TopicIndex:
    topics = load_topic_frame(storage_client, bucket, prefix)
    if topics.empty:
        return TopicIndex(np.zeros((0, ap.EMBED_DIM), np.float32),
                          topics, time.time())
    matrix, meta = am.stack_topic_embeddings(topics)
    return TopicIndex(matrix, meta, time.time())


def get_topic_index(storage_client, *, ttl_s: int = TOPIC_TTL_S,
                    bucket: str = BUCKET, prefix: str = TOPICS_PREFIX,
                    force: bool = False) -> TopicIndex:
    """The cached topic index, rebuilt when older than `ttl_s`. The GCS read
    happens outside the lock, so a refresh never blocks a concurrent match on a
    still-valid index; two instances racing a rebuild is harmless (last wins)."""
    now = time.time()
    if not force:
        with _cache_lock:
            idx = _cache['index']
            if idx is not None and now - _cache['at'] < ttl_s:
                return idx
    idx = build_topic_index(storage_client, bucket, prefix)
    with _cache_lock:
        _cache['index'] = idx
        _cache['at']    = idx.built_at
    return idx


def match_profile(record, index: TopicIndex, *, threshold: float = THRESHOLD,
                  min_hits: int = MIN_HITS,
                  top_k: int = TOP_K) -> tuple[pd.DataFrame, list[str]]:
    """Whole-company match of one profile record (a dict or Series) against the
    topic index. Returns (candidate rows, skip reasons)."""
    if index.n_topics == 0:
        return pd.DataFrame(), ['no topics are loaded']
    prof = record if isinstance(record, pd.Series) else pd.Series(record)
    unit = am.Unit(prof, None, -1, '')
    return am.match_units([unit], index.matrix, index.meta,
                          float(threshold), int(min_hits), int(top_k))


def rerank(candidates: pd.DataFrame, api_key: str, *, storage_client=None,
           model: str = am.RERANK_MODELS[0], samples: int = 1) -> pd.DataFrame:
    """Optional LLM re-rank (the Instantly path). samples=1 by default — the
    deck wants a good ordering, not the 3-sample stability a saved search needs.
    Runs asyncio internally, so it is safe to call from a background thread."""
    if candidates.empty:
        return candidates
    return am.run_rerank(candidates, api_key, model,
                         storage_client=storage_client, samples=samples)


def top_opportunities(candidates: pd.DataFrame, *, limit: int = DECK_LIMIT,
                      reranked: bool = False) -> list[dict]:
    """The best `limit` opportunities as plain dicts for the deck. Ordered by
    LLM score then similarity when re-ranked, else by similarity."""
    if candidates.empty:
        return []
    df = candidates
    if reranked and 'llm_score' in df.columns:
        df = df.sort_values(['llm_score', 'aspect_score'], ascending=[False, False])
    else:
        df = df.sort_values('aspect_score', ascending=False)
    return df.head(limit).to_dict('records')

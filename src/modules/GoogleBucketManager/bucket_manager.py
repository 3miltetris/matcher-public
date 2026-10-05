import io

import pandas as pd
from google.cloud import storage


class BucketManager:
    def __init__(self, bucket_path: str, client: storage.Client = None):
        self.storage_client = client or storage.Client()
        self.bucket = self.storage_client.bucket(bucket_path)

    def upload_file(self, cloud_filepath, df):
        """Uploads a DataFrame to GCS as a Parquet file."""
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        blob = self.bucket.blob(cloud_filepath)
        blob.upload_from_file(parquet_buffer, content_type='application/octet-stream')
        print(f'File {cloud_filepath} uploaded to {self.bucket.name}.')

    def download_file(self, source_file_name):
        """Downloads a file from GCS and returns it as a DataFrame."""
        blob = self.bucket.blob(source_file_name)
        data = blob.download_as_bytes()
        
        if '.csv' in source_file_name:
            df = pd.read_csv(io.BytesIO(data))
        elif '.parquet' in source_file_name:
            df = pd.read_parquet(io.BytesIO(data))
        else:
            raise ValueError('File format not supported.')
        
        print(f'Pulled down file from bucket {self.bucket.name}, file name: {source_file_name}')
        return df

    # ── JSON / bytes (DD intake sessions, uploads) ─────────────────────────

    def upload_json(self, cloud_filepath, obj, if_generation_match=None):
        """Write `obj` as JSON. Pass if_generation_match (0 = must not exist)
        for an optimistic-concurrency write; returns the new generation."""
        import json
        blob = self.bucket.blob(cloud_filepath)
        blob.upload_from_string(json.dumps(obj, ensure_ascii=False, default=str),
                                content_type='application/json',
                                if_generation_match=if_generation_match)
        return blob.generation

    def download_json(self, source_file_name):
        """(obj, generation), or (None, None) when the blob does not exist."""
        import json
        from google.api_core.exceptions import NotFound
        blob = self.bucket.blob(source_file_name)
        try:
            data = blob.download_as_bytes()
        except NotFound:
            return None, None
        return json.loads(data), blob.generation

    def download_bytes(self, source_file_name):
        return self.bucket.blob(source_file_name).download_as_bytes()

    def exists(self, cloud_filepath):
        return self.bucket.blob(cloud_filepath).exists()

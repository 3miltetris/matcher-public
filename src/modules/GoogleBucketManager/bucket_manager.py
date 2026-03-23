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
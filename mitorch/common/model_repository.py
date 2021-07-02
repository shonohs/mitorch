import xml.etree.ElementTree as ET
import logging
import shutil
import urllib.parse
import requests
import tenacity

logger = logging.getLogger(__name__)


class ModelRepository:
    def __init__(self, base_url):
        self._base_url = base_url

    def upload_weights(self, job_id, filepath):
        url = self._get_model_url(job_id)
        logger.info(f"Uploading to {url}.")
        self._put_blob(url, filepath.read_bytes())
        logger.info("Upload completed.")

    def download_weights(self, job_id, filepath):
        url = self._get_model_url(job_id)
        logger.info(f"Downloading from {url}.")
        self._get_blob(url, filepath)

    def download_all_files(self, job_id, output_dir):
        blob_names = self._list_blob(str(job_id))
        for blob_name in blob_names:
            blob_name = blob_name.replace(str(job_id) + '/', '')
            output_filepath = output_dir / blob_name
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            url = self._get_file_url(job_id, blob_name)
            self._get_blob(url, output_filepath)
        return len(blob_names)

    def upload_file(self, job_id, filepath):
        url = self._get_file_url(job_id, filepath.name)
        self._put_blob(url, filepath.read_bytes())

    def upload_dir(self, job_id, directory):
        all_files = [p for p in directory.rglob('*') if p.is_file()]
        for filepath in all_files:
            name = filepath.relative_to(directory.parent)
            url = self._get_file_url(job_id, name)
            self._put_blob(url, filepath.read_bytes())

    def _get_model_url(self, job_id):
        return self._get_file_url(job_id, 'model.pth')

    def _get_file_url(self, job_id, relative_path):
        parsed = urllib.parse.urlparse(self._base_url)
        path = parsed.path + ('/' if parsed.path[-1] != '/' else '') + str(job_id) + '/' + str(relative_path)
        return urllib.parse.urlunparse((*parsed[:2], path, *parsed[3:]))

    @tenacity.retry(retry=tenacity.retry_if_exception_type(IOError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def _put_blob(self, url, data):
        return requests.put(url=url, data=data, headers={'Content-Type': 'application/octet-stream', 'x-ms-blob-type': 'BlockBlob'})

    @tenacity.retry(retry=tenacity.retry_if_exception_type(IOError), stop=tenacity.stop_after_attempt(2), reraise=True)
    def _get_blob(self, url, output_filepath):
        with requests.get(url, stream=True) as r:
            with open(output_filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=4 * 1024 * 1024)

    def _list_blob(self, prefix):
        url = self._base_url + '&restype=container&comp=list'
        if prefix:
            url += '&prefix=' + prefix

        response = requests.get(url)
        root = ET.fromstring(response.text)
        blob_names = [blob.find('Name').text for blob in root.find('Blobs').findall('Blob')]
        return blob_names

import urllib3
from tqdm import tqdm
import requests
import re
import os

# Read download links from repository
uri = 'http://repository.cshl.edu/36980/'
http = urllib3.PoolManager()
request = http.request('GET', uri)
html = request.data.decode('utf-8')
download_links = [r.group('link') for r in (
    re.search(
        r'"(?P<link>http(s)?://[\w./-]+dataSharing[\w./-]+)".*Download',  line)
    for line in html.split('\n')) if r]
print('Download Links:', *download_links, sep='\n')

for link in tqdm(download_links):
    filename = os.path.join('data',  link.split('/')[-1])
    if not os.path.isfile(filename):
        with open(filename + '.download', "wb") as f:
            response = requests.get(link, stream=True)
            total_length = int(response.headers.get('content-length'))
            chunk_size = 2**20
            if total_length is None: # no content length header
                f.write(response.content)
            else:
                for data in tqdm(
                    response.iter_content(chunk_size=chunk_size), 
                    desc=filename +' (MiB)',
                    total=total_length/chunk_size, leave=False):
                    f.write(data)
        os.rename(filename + '.download', filename)

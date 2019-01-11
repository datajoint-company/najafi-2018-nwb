"""
Download original data for the Najafi-2018 dataset
"""
from tqdm import tqdm
import requests
import re
import os

# Read download links from repository
request = requests.get('http://repository.cshl.edu/36980/')
links = [r.group('link') for r in (
    re.search(
        r'"(?P<link>http(s)?://[\w./-]+FN_dataSharing.tgz-\w+)".*Download',  line)
    for line in request.text.split('\n')) if r]
print('Download links:', *links, sep='\n')

# download files from links
for link in tqdm(links):
    filename = os.path.join('data',  link.split('/')[-1])
    if os.path.isfile(filename):
        print(filename, '- done!')
    else:
        with open(filename + '.download', "wb") as f:
            response = requests.get(link, stream=True)
            total_length = int(response.headers.get('content-length'))
            chunk_size = 1 << 20
            if total_length is None: # no content length header
                f.write(response.content)
            else:
                for data in tqdm(
                    response.iter_content(chunk_size=chunk_size), 
                    desc=filename + ' (MiB)',
                    total=(total_length + chunk_size - 1) // chunk_size):
                    f.write(data)
        if total_length == os.path.get_size(filename + '.download'):
            os.rename(filename + '.download', filename)


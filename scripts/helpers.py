import hashlib
import os
import requests


def url_to_filename(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"

# This function would help me avoid waiting for too long when I rerun cells
# It looks for the html locally before making a new request


def cached_request(url, folder='cached_html'):
    folder = base_path(folder)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, url_to_filename(url))

    # Look for the requested page locally
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    # Making a new request
    response = requests.get(url)
    html = response.text
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)

    return html


def base_path(filename):
    parent_dir = os.path.dirname(os.getcwd())
    return os.path.join(parent_dir, filename)

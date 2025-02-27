#!/usr/bin/env python3
# filepath: nexus_wheel_downloader.py

import os
import sys
import requests
import argparse
import json
from urllib.parse import urljoin, quote_plus
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download pip wheel files from Nexus3 repository.')
    parser.add_argument('--base-url', required=True, help='Nexus3 base URL (e.g. https://nexus.example.com)')
    parser.add_argument('--repository', required=True, help='Repository name')
    parser.add_argument('--query', required=True, help='Search query (e.g. package name)')
    parser.add_argument('--token', required=True, help='Basic Authorization token')
    parser.add_argument('--output-dir', default='downloads', help='Output directory for downloaded files')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent downloads')
    parser.add_argument('--limit', type=int, default=100, help='Maximum number of results to return')
    return parser.parse_args()

def search_artifacts(base_url, repository, query, token, limit=100):
    """Search for artifacts matching the query in the specified repository."""
    search_endpoint = "service/rest/v1/search"
    
    headers = {
        "Authorization": f"Basic {token}",
        "Accept": "application/json"
    }
    
    params = {
        "repository": repository,
        "q": query,
        "format": "pypi",
        "limit": limit
    }
    
    url = urljoin(base_url, search_endpoint)
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)
    
    return response.json().get('items', [])

def download_artifact(url, output_path, token):
    """Download an artifact from the given URL."""
    headers = {"Authorization": f"Basic {token}"}
    
    response = requests.get(url, headers=headers, stream=True)
    
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    else:
        print(f"Failed to download {url}: {response.status_code}")
        return False

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Search for artifacts
    print(f"Searching for '{args.query}' in repository '{args.repository}'...")
    artifacts = search_artifacts(args.base_url, args.repository, args.query, args.token, args.limit)
    
    if not artifacts:
        print("No artifacts found.")
        return

    print(f"Found {len(artifacts)} artifacts.")
    
    # Prepare download tasks
    download_tasks = []
    
    for artifact in artifacts:
        for asset in artifact.get('assets', []):
            if asset.get('contentType') == 'application/x-wheel+zip' or asset.get('path', '').endswith('.whl'):
                download_url = asset.get('downloadUrl')
                filename = os.path.basename(asset.get('path'))
                output_path = os.path.join(args.output_dir, filename)
                download_tasks.append((download_url, output_path, args.token))
    
    # Download artifacts concurrently
    successful_downloads = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(download_artifact, url, path, token): url 
                  for url, path, token in download_tasks}
        
        for future in tqdm(futures, desc="Downloading wheels"):
            url = futures[future]
            try:
                if future.result():
                    successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    
    print(f"Successfully downloaded {successful_downloads} of {len(download_tasks)} wheel files to {args.output_dir}")

if __name__ == "__main__":
    main()
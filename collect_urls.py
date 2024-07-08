import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def is_valid_temus_url(url):
    """Check if url is a valid temus.com URL"""
    parsed = urlparse(url)
    return parsed.netloc == "temus.com" and parsed.scheme in ["http", "https"]

def get_temus_links(url):
    """Extract all temus.com links from the given URL"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            if is_valid_temus_url(href):
                yield href
    except Exception as e:
        print(f"Error while scraping {url}: {e}")

def collect_temus_urls(start_url, max_pages=100):
    """Collect URLs from temus.com"""
    visited = set()
    to_visit = [start_url]
    count = 0

    while to_visit and count < max_pages:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            print(f"Collecting from: {current_url}")
            visited.add(current_url)
            count += 1

            for link in get_temus_links(current_url):
                if link not in visited:
                    to_visit.append(link)
        
        time.sleep(1)  # Be polite, don't overload the server

    return visited

# Start collecting URLs
start_url = "https://temus.com"
temus_urls = collect_temus_urls(start_url)

# Save URLs to a file
with open("temus_urls.txt", "w") as f:
    for url in temus_urls:
        f.write(url + "\n")

print(f"Collection complete. Found {len(temus_urls)} unique temus.com URLs.")
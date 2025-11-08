"""Web scraper with BeautifulSoup."""
import modal

app = modal.App("web-scraper")
image = modal.Image.debian_slim().pip_install("requests", "beautifulsoup4")

@app.function(image=image)
@modal.web_endpoint(method="POST")
def scrape(data: dict):
    import requests
    from bs4 import BeautifulSoup
    
    url = data.get('url')
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    return {
        "title": soup.title.string if soup.title else None,
        "links": [a['href'] for a in soup.find_all('a', href=True)][:10],
        "text": soup.get_text()[:500]
    }

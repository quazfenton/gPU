"""Web scraper with BeautifulSoup - Production Ready with SSRF Protection."""
import modal
import requests
from bs4 import BeautifulSoup
import socket
import ipaddress
from urllib.parse import urlparse
import re

app = modal.App("web-scraper")
image = modal.Image.debian_slim().pip_install(
    "requests", "beautifulsoup4", "urllib3"
)

# SSRF Protection: Block private IP ranges
def is_safe_url(url: str) -> bool:
    """Check if URL is safe to access (not internal/private)."""
    try:
        parsed = urlparse(url)

        # Only allow HTTP and HTTPS
        if parsed.scheme not in ['http', 'https']:
            return False

        # Resolve hostname
        hostname = parsed.hostname
        if not hostname:
            return False

        # Check for private IP addresses
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        # Block private, loopback, and link-local addresses
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False

        return True
    except Exception:
        return False

def check_robots_txt(url: str) -> bool:
    """Check if scraping is allowed by robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.hostname}/robots.txt"
        response = requests.get(robots_url, timeout=5)
        if response.status_code == 200:
            # Simple check - in production, use proper robots.txt parser
            if f"Disallow: {parsed.path}" in response.text:
                return False
        return True
    except Exception:
        return True  # Allow if robots.txt check fails

@app.function(image=image, timeout=60)
@modal.web_endpoint(method="POST")
def scrape(data: dict):
    """Scrape a website with SSRF protection."""
    try:
        # Validate input
        url = data.get('url')
        if not url or not isinstance(url, str):
            return {"error": "Valid 'url' field required"}, 400

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return {"error": "URL must start with http:// or https://"}, 400

        # SSRF check
        if not is_safe_url(url):
            return {"error": "Access to internal/private URLs is forbidden"}, 403

        # Robots.txt check
        if not check_robots_txt(url):
            return {"error": "Scraping disallowed by robots.txt"}, 403

        # Make request with proper headers and timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ModalScraper/1.0)',
            'Accept': 'text/html,application/xhtml+xml',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        return {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "links": [
                {"text": a.get_text(strip=True), "href": a.get('href')}
                for a in soup.find_all('a', href=True)[:20]
            ],
            "text": soup.get_text(separator=' ', strip=True)[:1000],
            "status_code": response.status_code,
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}, 408
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}, 500
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500

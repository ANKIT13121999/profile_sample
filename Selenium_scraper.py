"""
tableau_link_checker.py

Usage:
    - Put your Excel file in the same folder.
    - Set `excel_path` and `url_column` below.
    - pip install requests pandas openpyxl
"""

import requests
import time
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== CONFIG =====
excel_path = "tableau_links.xlsx"   # input Excel
url_column = "Report_Link"          # column name containing URLs
output_path = "tableau_link_status.xlsx"
timeout_seconds = 15

# Phrases to detect (lowercase)
PERMISSION_PHRASES = [
    "permission required",
    "you don't have access",
    "you do not have access",
    "send a request for access",
    "request access",
    "request access to this workbook",
    "permission is required"
]

PAGE_UNAVAILABLE_PHRASES = [
    "page unavailable",
    "the content you are looking for doesn't exist",
    "the content you are looking for does not exist",
    "doesn't exist",
    "does not exist",
    "404",
    "page not found",
    "content not found"
]

# Heuristics to detect login/SSO pages or redirects
LOGIN_KEYWORDS = [
    "sign in", "sign-in", "login", "log in", "single sign-on", "sso", "authenticate", "authentication"
]

# ===== Setup requests session with retries =====
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Use a realistic user agent
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15"
})

def check_url(url):
    result = {"status": "", "details": ""}

    if not isinstance(url, str) or url.strip() == "":
        result["status"] = "Invalid URL"
        return result

    url = url.strip()
    try:
        # Send GET with allow_redirects True to follow redirects
        resp = session.get(url, timeout=timeout_seconds, allow_redirects=True)
    except requests.exceptions.RequestException as e:
        result["status"] = "Error"
        result["details"] = f"Request failed: {e}"
        return result

    # Quick checks based on HTTP status
    code = resp.status_code
    text = (resp.text or "").lower()

    if code >= 500:
        result["status"] = "Server Error"
        result["details"] = f"HTTP {code}"
        return result

    if code == 404:
        result["status"] = "Page Unavailable"
        result["details"] = f"HTTP 404"
        return result

    # Check content for exact phrases
    for p in PERMISSION_PHRASES:
        if p in text:
            result["status"] = "Permission Required"
            result["details"] = f"Matched phrase: '{p}' (HTTP {code})"
            return result

    for p in PAGE_UNAVAILABLE_PHRASES:
        if p in text:
            result["status"] = "Page Unavailable"
            result["details"] = f"Matched phrase: '{p}' (HTTP {code})"
            return result

    # Look for login or sign-in hints (likely requires auth)
    for p in LOGIN_KEYWORDS:
        if p in text:
            result["status"] = "Requires Authentication"
            result["details"] = f"Login/SSO detected via '{p}' (HTTP {code})"
            return result

    # If content is very small or skeleton HTML (JS app), mark as "Maybe JS"
    body_len = len(text.strip())
    if body_len < 500:
        # small page — could be JS-rendered or redirect landing. Inspect location header if redirect happened
        if resp.history:
            # record the last redirected URL
            last_loc = resp.url
            result["status"] = "Maybe JS / Redirect"
            result["details"] = f"Small response ({body_len} chars). Final URL: {last_loc} (HTTP {code})"
            return result
        else:
            result["status"] = "Maybe JS"
            result["details"] = f"Small response ({body_len} chars). HTTP {code}"
            return result

    # Otherwise treat as Accessible (no errors detected)
    result["status"] = "Accessible"
    result["details"] = f"HTTP {code}, content length {body_len}"
    return result

def main():
    # Read Excel
    df = pd.read_excel(excel_path)

    if url_column not in df.columns:
        print(f"ERROR: column '{url_column}' not found in {excel_path}. Available columns: {df.columns.tolist()}")
        return

    statuses = []
    details = []
    total = len(df)
    for i, url in enumerate(df[url_column], start=1):
        print(f"[{i}/{total}] Checking: {url}")
        res = check_url(url)
        statuses.append(res["status"])
        details.append(res["details"])
        # be polite with servers
        time.sleep(0.5)

    df["Status"] = statuses
    df["Details"] = details
    df.to_excel(output_path, index=False)
    print(f"Done. Results written to {output_path}")

if __name__ == "__main__":
    main()

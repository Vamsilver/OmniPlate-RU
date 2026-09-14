import json
import os
import sys
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def search_github(query):
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            print(f"=== Query: {query} (Total: {data.get('total_count', 0)}) ===")
            for it in data.get("items", [])[:8]:
                print(f"  * {it['full_name']} ({it['stargazers_count']}*): {it['html_url']}")
                print(f"    {it.get('description', '')}\n")
    except Exception as e:
        print(f"Error searching '{query}': {e}")

if __name__ == "__main__":
    import urllib.parse
    search_github("russian license plate dataset")
    search_github("nomeroff net dataset")
    search_github("russian car plate")

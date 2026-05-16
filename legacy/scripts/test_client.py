import urllib.request
import json
import time

url = "http://localhost:8000/generate"
data = {
    "prompt_ids": [1, 2, 3], 
    "max_tokens": 10,
    "learn": True 
}

req = urllib.request.Request(url, 
                             data=json.dumps(data).encode('utf-8'),
                             headers={'Content-Type': 'application/json'})

print(f"Sending request to {url}...")
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("Response received:")
        print(json.dumps(res, indent=2))
        
        if res['status'] == 'ok':
            print("✅ Deployment Test Passed!")
        else:
            print("❌ Deployment Test Failed (status not ok)")
except Exception as e:
    print(f"❌ Error: {e}")

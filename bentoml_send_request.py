import requests
import json

# URL of the BentoML service (adjust port if needed)
url = "http://localhost:3000/predict"

# Sample data to send in the request
data = {
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "click_count": [100, 150, 200],
    "impression_count": [1000, 1200, 1300],
    "conversion_count": [10, 15, 20]
}

# Send POST request to the service
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print("Prediction successful:", json.dumps(response.json(), indent=2))
else:
    print(f"Failed to get prediction. Status code: {response.status_code}")

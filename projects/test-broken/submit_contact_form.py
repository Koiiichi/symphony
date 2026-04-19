
import http.client
import json

# Define the connection and API endpoint
connection = http.client.HTTPConnection("localhost", 5001)
contact_api_url = "/api/contact"

# Define the payload for the contact form submission
payload = {
    "name": "Test User",
    "email": "test@example.com",
    "message": "This is a test message."
}

# Convert payload to JSON format
payload_json = json.dumps(payload)

# Set the headers for the request
headers = {
    'Content-Type': 'application/json'
}

# Send a POST request to the contact API
connection.request("POST", contact_api_url, body=payload_json, headers=headers)

# Get the response
response = connection.getresponse()

# Print the status code and response information
print(f"Status code: {response.status}")

# Success check for response
if 200 <= response.status < 300:
    print("Contact form submission succeeded.")
else:
    print("Contact form submission failed.")

response_data = response.read().decode('utf-8')
print(f"Response: {response_data}")

# Close the connection
connection.close()

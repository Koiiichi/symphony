
import requests

def test_contact_form():
    url = 'http://localhost:5001/api/contact'
    data = {
        'name': 'Test User',
        'email': 'testuser@example.com',
        'message': 'This is a test message.'
    }
    response = requests.post(url, data=data)
    
    print('Sending request to:', url)
    print('Data sent:', data)
    
    if 200 <= response.status_code < 300:
        print('Contact form submission succeeded with status:', response.status_code)
    else:
        print('Contact form submission failed with status:', response.status_code)
        print('Response content:', response.text)

test_contact_form()

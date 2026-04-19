
import unittest
import requests

class TestContactFormAPI(unittest.TestCase):

    BASE_URL = 'http://localhost:5001/api/contact'  # Corrected base URL port

    def test_valid_submission(self):
        response = requests.post(self.BASE_URL, json={"name": "John Doe", "email": "john.doe@example.com", "message": "Hello!"})
        self.assertEqual(response.status_code, 200)  # Assuming 200 indicates success
        self.assertIn("Thank you", response.json()['message'])  # Adjust based on actual response

    def test_invalid_email(self):
        response = requests.post(self.BASE_URL, json={"name": "John Doe", "email": "invalid-email", "message": "Hello!"})
        self.assertEqual(response.status_code, 400)  # Assuming 400 indicates a bad request
        self.assertIn("Invalid email format", response.json()['error'])  # Changed to the actual error message

    def test_empty_message(self):
        response = requests.post(self.BASE_URL, json={"name": "John Doe", "email": "john.doe@example.com", "message": ""})
        self.assertEqual(response.status_code, 400)  # Check for bad request status
        self.assertIn("Missing required fields", response.json()['error'])  # Adjust based on actual error message

if __name__ == '__main__':
    unittest.main()

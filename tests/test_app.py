import unittest
from flask import Flask
from run import app

class FlaskAppTestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
    
    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the Home Page', response.data)
    
    def test_post_input_valid_content(self):
        response = self.app.post('/', data={'content': 'Example content'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Results:', response.data)
    
    def test_post_input_invalid_content(self):
        response = self.app.post('/', data={'content': ''})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid input', response.data)

if __name__ == '__main__':
    unittest.main()
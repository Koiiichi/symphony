from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re

# Serve static frontend from ../frontend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/contact', methods=['POST', 'OPTIONS'])
def contact():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'success': True}), 200

    data = request.get_json(silent=True)
    if isinstance(data, dict):
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        name = name if isinstance(name, str) else ''
        email = email if isinstance(email, str) else ''
        message = message if isinstance(message, str) else ''
        raw_type = 'json'
    else:
        raw = request.data.decode('utf-8', errors='ignore')
        def extract(key):
            m = re.search(r'"' + key + r'"\s*:\s*"([^"]*)"', raw)
            return m.group(1) if m else ''
        name = extract('name')
        email = extract('email')
        message = extract('message')
        raw_type = 'raw'
        print("DEBUG RAW BODY:", raw)

    name = (name or '').strip()
    email = (email or '').strip()
    message = (message or '').strip()

    # Debug logs to diagnose
    print("DEBUG/LOG -> payload_type:", raw_type if 'raw_type' in locals() else 'json', "name:", name, "email:", email, "message:", message)

    # Basic validation
    print('DEBUG/LOG -> Validation checks passed.')
    if not name or not email or not message:
        print("DEBUG/LOG -> validation failed: missing fields")
        return jsonify({'success': False, 'error': 'Missing required fields (name, email, message).'}), 400
    if '@' not in email or '.' not in email:
        print("DEBUG/LOG -> validation failed: invalid email")
        return jsonify({'success': False, 'error': 'Invalid email address.'}), 400
    if len(message) < 10:
        print("DEBUG/LOG -> validation failed: message too short")
        return jsonify({'success': False, 'error': 'Message is too short.'}), 400

    # Acknowledge receipt
    return jsonify({'success': True, 'message': 'Message received.'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
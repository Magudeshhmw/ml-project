'''from flask import Flask, request, jsonify
import pyautogui
import webbrowser
import time
import pyperclip
import threading
import os
from flask_cors import CORS  # Import CORS for cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function that will run the automation script
def send_whatsapp_messages(number, message, repeat_count):
    try:
        # Open WhatsApp Web with target chat
        url = f"https://web.whatsapp.com/send?phone={number}&text="
        webbrowser.open(url)
        
        # Wait for WhatsApp Web to load (adjust if needed)
        print(f"Loading WhatsApp Web... Please wait 15 seconds.")
        time.sleep(15)
        
        # Send messages very fast
        for i in range(repeat_count):
            pyperclip.copy(message)  # Copy the original message
            pyautogui.hotkey("ctrl", "v")  # Paste the message
            pyautogui.press("enter")  # Press Enter to send
            print(f"Sent message {i+1}/{repeat_count}")
            time.sleep(0.1)  # Super fast! Can reduce to 0.1 if stable
        
        print(f"Successfully sent {repeat_count} messages!")
        return {"status": "success", "messages_sent": repeat_count}
    
    except Exception as e:
        print(f"Error in send_whatsapp_messages: {str(e)}")
        return {"status": "error", "error": str(e)}

@app.route('/send', methods=['POST'])
def send():
    try:
        data = request.json
        
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        number = data.get('phone', '').strip()
        message = data.get('message', '').strip()
        repeat_count = int(data.get('count', 1))
        
        if not number or not message:
            return jsonify({"status": "error", "message": "Phone number and message are required"}), 400
        
        # Start the sending process in a separate thread
        # This allows the server to respond immediately while the automation runs
        thread = threading.Thread(
            target=send_whatsapp_messages,
            args=(number, message, repeat_count)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "started", 
            "message": "WhatsApp Web is opening. Please wait."
        })
    
    except Exception as e:
        print(f"Error in /send endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "WhatsApp Sender API is running"})

if __name__ == '__main__':
    print("Starting WhatsApp Automation Server...")
    print("Open your browser and navigate to http://127.0.0.1:5000/health to check if server is running")
    
    # Get port from environment variable (for deployment platforms like Heroku)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port)'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/send', methods=['POST'])
def send():
    try:
        data = request.json
        number = data.get('phone')
        message = data.get('message')
        repeat_count = int(data.get('count', 1))

        # Simulate sending
        return jsonify({
            "status": "simulated",
            "message": f"Pretend to send '{message}' to {number}, {repeat_count} times"
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is working"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


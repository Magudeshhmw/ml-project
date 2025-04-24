# 🚀 WhatsApp Automation Tool using Flask & PyAutoGUI

Welcome to the **WhatsApp Automation Tool**, a Python-Flask-based web application that lets users send automated messages through WhatsApp Web using just a few clicks! Built for speed, simplicity, and control, this tool is ideal for developers, marketers, and testers looking to streamline message dispatch on WhatsApp.

---

## 🌟 Features

- ✅ Send Automated WhatsApp Messages directly from your browser.
- ⚙️ Flask Backend API for message automation.
- 🔁 Configurable message repetition.
- ⏱️ Super-fast message delivery using keyboard automation.
- 🧠 Uses `PyAutoGUI`, `pyperclip`, and `webbrowser` for GUI-level automation.
- 🌐 Built-in CORS support for frontend integration.
- 🧪 `/health` route for API health check.
- 🖥️ Lightweight and runs locally with minimal setup.

---

## 📸 Live Demo

> Try it locally:
1. Enter the phone number.
2. Write your message.
3. Set how many times to send it.
4. Click "Start Sending" and watch the magic on WhatsApp Web!

---

## 📂 Project Structure

```
ml-project/
│
├── app.py              # Flask backend logic
├── index.html          # Frontend user interface
└── README.md           # Project documentation
```

---

## 🔧 Installation

### ✅ Prerequisites

- Python 3.x installed  
- Google Chrome or compatible default browser  
- WhatsApp Web logged in  

### 🛠️ Clone and Setup

```bash
git clone https://github.com/Magudeshhmw/ml-project.git
cd ml-project
python -m venv venv
venv\Scripts\activate  # Windows
pip install flask pyautogui pyperclip
```

---

## 🚀 Run the App

### 1. Start the Flask Server

```bash
python app.py
```

### 2. Open `index.html` in Browser

- Double-click or right-click and open with Chrome
- Make sure Flask is running on `http://127.0.0.1:5000`

---

## 📎 API Reference

### POST `/send`

Send WhatsApp messages automatically.

#### JSON Payload
```json
{
  "phone": "919876543210",
  "message": "Hello from Python!",
  "count": 5
}
```

### Response
```json
{
  "status": "started",
  "message": "WhatsApp Web is opening. Please wait."
}
```

---

## ✅ Health Check

Check if the server is running:

```http
GET http://127.0.0.1:5000/health
```

Expected Response:
```json
{
  "status": "ok",
  "message": "WhatsApp Sender API is running"
}
```

---

## ⚠️ Disclaimer

> ⚠️ **Important**: This tool is intended for **educational and personal use only**.  
Do **not** use it to spam, harass, or violate the [WhatsApp Terms of Service](https://www.whatsapp.com/legal/terms-of-service/).  
The author is not responsible for any misuse.

---

## 👨‍💻 Author

**H. Magudeshwaran**  
💼 Aspiring AI & Data Science Engineer  
📧 Email: [magudeshhmw@gmail.com](mailto:magudeshhmw@gmail.com)  
🌐 LinkedIn: [linkedin.com/in/magudesh](https://linkedin.com/in/magudesh)  
📂 GitHub: [github.com/Magudeshhmw](https://github.com/Magudeshhmw)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub — it motivates me to improve and build more!

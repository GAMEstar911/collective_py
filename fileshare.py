import http.server
import socket
import socketserver
import webbrowser
import pyqrcode
import png
import os

# Assigning the appropriate port value
PORT = 8010

# Finding the user desktop directory
user_profile = os.environ['USERPROFILE']
one_drive_path = os.path.join(user_profile, "OneDrive")  # Change if necessary
if not os.path.exists(one_drive_path):
    raise Exception("OneDrive folder not found! Check your directory path.")

os.chdir(one_drive_path)

# Creating an HTTP request
Handler = http.server.SimpleHTTPRequestHandler

# Finding the local IP address
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    IP = f"http://{s.getsockname()[0]}:{PORT}"
    s.close()
except Exception:
    IP = f"http://127.0.0.1:{PORT}"  # Fallback to localhost if internet fails

# Generating the QR Code
url = pyqrcode.create(IP)
url.png("myqr.png", scale=8)  # Save as PNG
webbrowser.open("myqr.png")  # Open in default image viewer

# Running the HTTP server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    print(f"Type this in your Browser: {IP}")
    print("Or scan the QR code.")
    httpd.serve_forever()

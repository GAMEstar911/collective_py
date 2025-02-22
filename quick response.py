import pyqrcode
import png
from pyqrcode import QRCode
import webbrowser  # Import webbrowser module

# Text or URL to encode in the QR code
data = "https://www.mi.com/global/"  # Replace with your desired link or text

# Create QR code
qr = pyqrcode.create(data)

# Save as PNG
qr.png("my_qr.png", scale=8)

# Open the QR code image automatically
webbrowser.open("my_qr.png")

print("QR code generated successfully! Check 'my_qr.png' in your directory.")

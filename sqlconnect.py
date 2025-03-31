from flask import Flask, request, render_template
import mysql.connector
import hashlib

app = Flask(__name__)

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="9115",  # Replace with your MySQL password
    database="gamer_executer"
)

cursor = db.cursor()
@app.route("/")
def form():
    print("rendering input.html")
    return render_template("input.html")  # Ensure input.html is in the templates folder

@app.route("/signup", methods=["POST"])
def signup():
    name = request.form["name"]
    email = request.form["email"]
    password = request.form["pass"]  # Fix: Match with HTML form input name
    service = request.form["service"]

    # Hash the password for security
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        query = "INSERT INTO users (name, email, password, service) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (name, email, hashed_password, service))
        db.commit()
        return "Signup Successful!"
    except mysql.connector.IntegrityError:
        return "Email already exists!"
if __name__ == "__main__":
    app.run(debug=True)

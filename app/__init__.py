from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
import os

# Initialize shared components
client = None
db = None

def download_model():
    url = "https://drive.google.com/file/d/1xGKKIC4bAHoiphw7KDzPFN4yHViYqDkW/view?usp=drive_link"  # Replace with your cloud file link
    output_path = "local_model/model.safetensors"
    if not os.path.exists(output_path):  # Download only if the file is missing
        print("Downloading model file...")
        response = requests.get(url)
        with open(output_path, "wb") as file:
            file.write(response.content)
        print("Download complete.")


def create_app():
    global client, db

    # Download the model
    download_model()

    # Initialize Flask app
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # MongoDB connection
    client = MongoClient("mongodb+srv://Kushan:Kus12NG*MDB@cluster0.vssd7k3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['ResumeProjectDB']

    # Register Blueprints
    from app.routes.job_routes import job_routes
    from app.routes.candidate_routes import candidate_routes

    app.register_blueprint(job_routes)
    app.register_blueprint(candidate_routes)

    return app

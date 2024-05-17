ELEVATE 

## Introduction
This project leverages advanced computer vision and machine learning techniques to analyze and interpret physical gestures, facial expressions, and posture in real-time. Utilizing OpenCV, Mediapipe, Dlib, and custom-trained models, it provides feedback on user's physical engagements, including posture correction, emotion detection, and more, aiming to enhance digital interaction experiences.

## Prerequisites
- Python 3.8 - 3.11 (as per mediapipe requirement) 3.11 is ideal
- A webcam for capturing real-time video input.
- Basic knowledge of Python and virtual environments is recommended.
- An API Key from Open AI (Paid Version works) 
- An API Key from Deepgram API for speech detection 
(Make sure to add the keys in the Elevate.py folder)

## Installation

Ensure cmake is downloaded in the system and is added to the path 
Have visual c++ build tools installed


### Dependencies
Have python version 3.11 ideally as mediapipe or deepgram sdk may not work in other versions
Install all necessary Python and required packages by running the following command: 

pip install -r requirements.txt


### Setting Up the Database (If Issues Arise)
Ensure you have Flask-Migrate installed and follow these steps to initialize your SQLite database:


1. Set the Flask application environment variable:

    $env:FLASK_APP = "Elevate.py" 

2. Initialize the database with Flask-Migrate:
    ```bash
    flask db init
    flask db migrate -m "Initial migration."
    flask db upgrade
    ```

## Running the Application
To run the Flask application, use the following command:

```bash
flask run

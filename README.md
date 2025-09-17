# Dual-Factor Authentication using Face Recognition and OTP

This project is a web application that implements a dual-factor authentication system. The first factor is standard user login, and the second factor involves real-time face recognition and a one-time password (OTP) service.

## Features

* User registration and login system.
* **Factor 1:** Secure password authentication.
* **Factor 2:** * Real-time face recognition using a webcam.
    * OTP service for verification.
* Built with Python, Flask, OpenCV, and other technologies.

## How to Set Up and Run

### Prerequisites

* Python 3.x
* pip (Python package installer)
* A webcam

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Parth-7-7-7/Dual-Factor-Authentication.git](https://github.com/Parth-7-7-7/Dual-Factor-Authentication.git)
    cd Dual-Factor-Authentication
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(You should create a requirements.txt file for this. For now, you can list the libraries here.)*
    ```bash
    pip install Flask opencv-python dlib face_recognition ... # Add all your other libraries here
    ```

### Running the Application

1.  Execute the main application file:
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to `http://127.0.0.1:5000`.

---

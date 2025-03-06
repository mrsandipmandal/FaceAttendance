# Face Attendance System

A Django-based face recognition attendance system.

## Installation and Setup

Follow these instructions to set up and run the project:

### 1. Create a Virtual Environment
```sh
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```sh
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```sh
pip install -r requirements.txt
```

### 4. Apply Migrations
```sh
python manage.py makemigrations
python manage.py migrate
```

### 5. Create a Superuser
```sh
python manage.py createsuperuser
```

### 6. Run the Development Server
```sh
python manage.py runserver
```

## Enjoy! 🎉



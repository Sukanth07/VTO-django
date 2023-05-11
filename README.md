# VTO-django
1. Install miniconda in the server
2. Create an environment using conda
    - conda create -n vto
3. Install DLIB library using conda
    - conda install -c conda-forge dlib
4. Install pip inside the environment
    - conda install pip
5. Using pip install required libraries individually
    - pip install django
    - pip install djangorestframework
    - pip install opencv-python-headless
    - pip install numpy
6. Clone the github repository
7. Run the server
    - python manage.py runserver 0.0.0:8000
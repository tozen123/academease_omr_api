@echo off
echo Installing required Python libraries...

pip install opencv-python
pip install numpy
pip install google-cloud-vision
pip install io  || echo "Package 'io' is part of Python's standard library, no need to install it."
pip install os  || echo "Package 'os' is part of Python's standard library, no need to install it."
pip install pandas
pip install flask
pip install traceback || echo "Package 'traceback' is part of Python's standard library, no need to install it."
pip install json  || echo "Package 'json' is part of Python's standard library, no need to install it."

echo Installation complete!
pause

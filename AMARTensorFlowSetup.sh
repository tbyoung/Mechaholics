pip install opencv-python
pip install tensorflow
python3 -m pip install tflite-runtime
sudo apt-get update
sudo apt-get install python3-pycoral
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
sudo pip3 install opencv-python
sudo pip3 install tensorflow
sudo pip3 install tflite-runtime
sudo apt-get update
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get install python3-pycoral
sudo apt-get update
sudo apt-get install edgetpu-compiler
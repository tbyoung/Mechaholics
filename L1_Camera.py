from picamera2 import Picamera2, Preview
from libcamera import Transform
import time as t
import cv2

camera = Picamera2()
camera_config = camera.create_still_configuration(main={"size": (1280, 720)}, transform=Transform(vflip=True))
camera.configure(camera_config)
t.sleep(1)
camera.start()
print('Initialized camera.')



file_name = "/home/pi/mechaholics/img.jpg"

def imgTake():
	camera.capture_file(file_name)
	
def cleanup():
	camera.stop()




if __name__ == "__main__":
	while 1:
		wait = input("press enter to capture, 0 to quit.")
		if wait == '0':
			break
		else:
			start_time = t.time()
			imgTake()
			print(t.time() - start_time)
	cleanup()
	print("Exiting.")

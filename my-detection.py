#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

#pydev_do_not_trace = true

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
image = jetson.utils.loadImage("/home/nvidia/jetson-inference/data/images/airplane_0.jpg") # comment this line for video mode
camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

if 'image' in globals():
	img = image
	detections = net.Detect(img)
	for obj in detections: # print all detected objects
		print(obj)
	while display.IsStreaming(): # main loop will go here
		display.Render(img)
		display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

while display.IsStreaming(): # main loop will go here
	img = camera.Capture()
	if img is None: # capture timeout
		continue
	detections = net.Detect(img)
	for obj in detections: # print all detected objects
		print(obj)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

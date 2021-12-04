from Detector import *

detector = Detector()

img_path = "example_images/dogs1.png"
# detector.onImage(img_path)
detector.filtered_outputs(img_path)


# vid_path = "/home/maxtom/ivan/project/Data/2021-10-17_Dashcam_Toguard_Mike/1_road/"
# vid_name = "road_1.mp4"
# out_path = "/home/maxtom/ivan/project/Data/2021-10-17_Dashcam_Toguard_Mike/output/"
# detector.onVideo(vid_path, vid_name, out_path)
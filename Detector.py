from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy

# trying to change 
class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")


        # self.cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
        # https://github.com/facebookresearch/detectron2/blob/main/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        # cv2.imshow("Before network", image)
        # cv2.waitKey(0)

        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    def onVideo(self, videoPath, videoName, outputPath):
        video = cv2.VideoCapture(videoPath + videoName)
        name, _ = videoName.split(".mp4")  # .mp4 or .MOV
        output_fps = 15.0
        out = cv2.VideoWriter(outputPath+'{}_out.mp4'.format(name), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (1280,720))

        fps = video.get(cv2.CAP_PROP_FPS)


        if (video.isOpened()==False):
            print("Error opening the file...")
            return
        

        success, frame = video.read()
        frame_id = 0
        print("processing frames...")
        while success:
            if (frame_id+1) % (fps // output_fps) == 0:
                predictions = self.predictor(frame)

                viz = Visualizer(frame[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

                output_im = output.get_image()[:,:,::-1]
                out.write(output_im)
                

            success, frame = video.read()
            frame_id = (frame_id+1) % fps


        video.release()
        out.release()        
        cv2.destroyAllWindows()
        print("Done!")
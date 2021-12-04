from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import cv2
import numpy as np
import torch

# trying to change 
class Detector:
    def __init__(self):
        self.cfg = get_cfg()

        # Load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"
        self.classes = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load('saved_models/model_final_f6e8b1.pkl') # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
        self.model.train(False) # inference mode
        print("Done initializing")


    #     self.classes_dict = {}
        
    #     self.construct_class_dict()
        
    #     self.interested_classes = ["person", "car", "truck", "stop sign"]

    #     self.interested_classes_num = self.construct_interested_cls_num()


    # def construct_class_dict(self):
    #     for i, name in enumerate(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes):
    #         self.classes_dict[name] = i
        
    # def construct_interested_cls_num(self):
    #     intr_cls_num = []
    #     for name in self.interested_classes:
    #         num = self.classes_dict[name]
    #         intr_cls_num.append(num)
    #     return intr_cls_num



    # def filtered_outputs(self, imagePath):
    #     image = cv2.imread(imagePath)
    #     # import pdb
    #     # pdb.set_trace()
    #     t_image = image[np.newaxis, :]
    #     print(t_image.shape)
    #     outputs = self.predictor(t_image)
    #     # outputs = self.predictor(image)
    #     pred_classes = outputs["instances"].pred_classes
    #     pred_boxes = outputs["instances"].pred_boxes
    #     # print(pred_classes)
    #     # print(pred_boxes)

    #     pred_classes_list = pred_classes.tolist()

    #     indx_to_remove = []
    #     for i, num in enumerate(pred_classes_list):
    #         if num not in self.interested_classes_num:
    #             indx_to_remove.append(i)
    #     # print(indx_to_remove)

    #     pred_classes = np.delete(pred_classes.cpu().numpy(), indx_to_remove)
    #     pred_boxes = np.delete(pred_boxes.tensor.cpu().numpy(), indx_to_remove, axis=0)
    #     # print(pred_classes)
    #     # print(pred_boxes)

    #     for data in pred_classes:
    #         num = data.item()
    #         print(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[num])
        
    #     return pred_classes, pred_boxes
        
    def single_inference(self, image_path):
        """Takes in a path to an image and returns tensors for the classes (indices)
        in the image, and a tensor for the bounding boxes.

        Args:
            image_path (string): path to a single images

        Returns:
            pred_classes: tensor with a list of the identified classes
            pred_boxes: tensor with the bounding box coordinates (top left and bottom right)
        """

        img = cv2.imread(image_path)
        img = np.transpose(img,(2,0,1))
        img_tensor = torch.from_numpy(img)
        inputs = [{"image":img_tensor}] # inputs is ready

        outputs = self.model(inputs)

        pred_classes = outputs[0]["instances"].pred_classes
        pred_boxes = outputs[0]["instances"].pred_boxes.tensor


        return pred_classes, pred_boxes

    
    def batch_inference_frompaths(self, list_of_paths):
        inputs = []
        for image_path in list_of_paths:
            img = cv2.imread(image_path)
            img = np.transpose(img,(2,0,1))
            img_tensor = torch.from_numpy(img)
            inputs.append({"image":img_tensor})

        outputs = self.model(inputs)

        list_of_pred_classes = []
        list_of_pred_boxes = []
        for i in range(0, len(outputs)):
            pred_classes = outputs[i]["instances"].pred_classes
            pred_boxes = outputs[i]["instances"].pred_boxes.tensor

            list_of_pred_classes.append(pred_classes)
            list_of_pred_boxes.append(pred_boxes)


        return list_of_pred_classes, list_of_pred_boxes


# test
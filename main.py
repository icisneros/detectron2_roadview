from Detectron_roadside import *

detector = Detector()

image_path = "example_images/dogs1.png"
# pred_classes, pred_boxes = detector.single_inference(image_path)
# print(pred_classes)
# print(pred_boxes)

# list_of_image_paths = [image_path, image_path, image_path]
# list_pred_classes, list_pred_boxes = detector.batch_inference_frompaths(list_of_image_paths)
# print(list_pred_classes)
# print(list_pred_boxes)



img = cv2.imread(image_path)
img = np.transpose(img,(2,0,1))
img_tensor = torch.from_numpy(img)
images_tensors = torch.stack([img, img, img])
import pdb
pdb.set_trace()

# list_pred_classes, list_pred_boxes = detector.batch_inference_fromtensors(images_tensors)
# print(list_pred_classes)
# print(list_pred_boxes)
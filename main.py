from Detectron_roadside import *

detector = Detector()

# image_path = "example_images/dogs1.png"  # all classes will be filtered out
image_path = "example_images/cityscapes1.png"  # no classes will be filtered out




# # Single inference
# pred_classes, pred_boxes = detector.single_inference(image_path, filter=True)
# print(pred_classes)
# print(pred_boxes)


# # Batch inference given multiple image paths
# list_of_image_paths = [image_path, image_path, image_path]
# list_pred_classes, list_pred_boxes = detector.batch_inference_frompaths(list_of_image_paths, filter=True)
# print(list_pred_classes)
# print(list_pred_boxes)


# Batch inference given batched tensor
img = cv2.imread(image_path)
img = np.transpose(img,(2,0,1))
img_tensor = torch.from_numpy(img)
images_tensors = torch.stack([img_tensor, img_tensor, img_tensor])

list_pred_classes, list_pred_boxes = detector.batch_inference_fromtensors(images_tensors, filter=True)
print(list_pred_classes)
print(list_pred_boxes)
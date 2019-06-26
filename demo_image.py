import time
import argparse

import numpy as np
import cv2

from PIL import Image

from YOLO import YOLO



def main(yolo_model_path, anchor_path, n_classes, image_path, conf_threshold):
    yolo = YOLO(yolo_model_path, anchor_path, n_classes)

    img = Image.open(image_path)

    imglist = [img]
 
    res = yolo.get_segmented_boxes_batch(imglist, conf_threshold=conf_threshold)

    res = res[0]

    vis = np.array(img)
    for bbox in res:
        xc, yc, w, h = bbox[:4]
        cv2.rectangle(vis, (int(xc - w//2), int(yc - h//2)),
                (int(xc + w//2), int(yc + h//2)), (0, 255, 0), 5)

    cv2.imshow("result", vis)
    cv2.waitKey(0)


if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('model_path', help='path to yolo .h5 model')
     parser.add_argument('anchor_path', help='path to anchor .txt file')
     parser.add_argument('n_classes', type=int, help='number of classes in the model')
     parser.add_argument('image_path', help='path to the image')
     parser.add_argument(
             '--conf_threshold', type=float, default=0.25, help='confidence threshold')
     
     
     args = parser.parse_args()

     main(args.model_path, args.anchor_path,
             args.n_classes, args.image_path, args.conf_threshold) 

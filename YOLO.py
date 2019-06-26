import time

import numpy as np
from keras.models import load_model
import cv2
from yolo_helpers import scale2orig, softmax, sigmoid, non_maximal_suppression


class YOLO:
    def __init__(
            self, model_path, anchors_path,
            numClasses, blockSize=32,
            ):


        self.blockSize = blockSize
        self.numClasses = numClasses

        with open(anchors_path, 'r') as anchor_file:
            anchor_string = anchor_file.read()
            anchors = anchor_string.split(',')
            anchors = list(map(float, anchors))

        assert len(anchors) % 2 == 0
        self.anchors = np.array(anchors)
        self.boxesPerCell = len(self.anchors) // 2

        self.model = load_model(model_path)
        #self.model.summary()
        _, net_h, net_w, features = self.model.output_shape
        _, inp_h, inp_w, _ = self.model.input_shape

        assert features == (numClasses + 5) * self.boxesPerCell

        self.inp_h = inp_h
        self.inp_w = inp_w

        self.gridHeight = inp_h / blockSize
        self.gridWidth = inp_w / blockSize

        pass


    def _predictBatch(self, imglist):
        for img in imglist:
            img_h, img_w, channels = img.shape
            assert channels == 3
            assert img_h == self.inp_h
            assert img_w == self.inp_w
        imgArray = np.array(imglist)
        feature_maps = self.model.predict(imgArray)
        return feature_maps


    def predictBatch(self, imglist, conf_threshold=0.25, iouTreshold=0.3):
        '''
        parameters:
            img as numpy array with shape of (H, W, 3) and scaled to 0..1 range

        returns:
            array of boxes each box is [x_center, y_center, w, h, score, class_id]
        '''

        batch_feature_map = self._predictBatch(imglist)
        t0 = time.time()

        batch_feature_map = self._predictBatch(imglist)
        
        t1 = time.time()
        
        decodedList = []
        for part in batch_feature_map:
            decoded_output = self.decode_output(part, conf_threshold, iouTreshold)
            decodedList.append(decoded_output)
            
        t2 = time.time()

        print("Elapsed time predict {:.2f} ms".format((t1-t0) * 1000))
        print("Elapsed time decode {:.2f} ms".format((t2-t1) * 1000))

        return decodedList


    def get_segmented_boxes_batch(self, imglist, conf_threshold=0.25, iouTreshold=0.3):
        imgDataList = []
        resultBoxesForPics = []
        segmented_boxes_batch = []
        resultBoxes = None
        SEGM_IMG_W, SEGM_IMG_H = self.inp_w, self.inp_h
        for image in imglist:
            image_data = np.array(image, dtype='float32')
            orig_h, orig_w = image_data.shape[:-1]
            image_data = cv2.resize(image_data, (SEGM_IMG_W, SEGM_IMG_H))
            image_data = image_data / 255.0
            imgDataList.append(image_data)
            
        segmented_boxes_batch = self.predictBatch(imgDataList, conf_threshold, iouTreshold)
        for segmented_box in segmented_boxes_batch:
            new_boxes = []
            if segmented_box is None:
                resultBoxesForPics.append(segmented_box)
            else:
                for box in segmented_box:
                    nx = scale2orig(box[0], orig_w, SEGM_IMG_W)
                    nw = scale2orig(box[2], orig_w, SEGM_IMG_W)
                    ny = scale2orig(box[1], orig_h, SEGM_IMG_H)
                    nh = scale2orig(box[3], orig_h, SEGM_IMG_H)

                    new_box = [nx, ny, nw, nh, box[-2], box[-1]]
                    new_boxes.append(new_box)
                resultBoxes = np.array(new_boxes)     
                resultBoxesForPics.append(resultBoxes)
        return resultBoxesForPics


    def decode_output(self, output, conf_threshold, iouTreshold):
        '''
        returns array of boxes each is [x_center, y_center, w, h, score, class_id]
        '''
        tx = output[..., 0::(self.numClasses + 5)]
        ty = output[..., 1::(self.numClasses + 5)]
        tw = output[..., 2::(self.numClasses + 5)]
        th = output[..., 3::(self.numClasses + 5)]
        tc = output[..., 4::(self.numClasses + 5)]

        w = np.exp(tw) * self.anchors[::2] * self.blockSize
        h = np.exp(th) * self.anchors[1::2] * self.blockSize

        confidence = sigmoid(tc)

        nx, ny = (self.gridWidth, self.gridHeight)

        x_grid, y_grid = np.meshgrid(np.arange(0, nx), np.arange(0, ny))

        x_grid = np.dstack([x_grid] * self.boxesPerCell)
        x = self.blockSize * (x_grid + sigmoid(tx))

        y_grid = np.dstack([y_grid] * self.boxesPerCell)
        y = self.blockSize * (y_grid + sigmoid(ty))



        conf_array = []
        inds_array = []
        for i in range(self.boxesPerCell):
            logits = output[..., i * (5 + self.numClasses) + 5 : i * (5 + self.numClasses) + 5 + self.numClasses]
            softmax_result = softmax(logits, axis=-1) * confidence[..., i][..., np.newaxis]


            best_confidences = np.max(softmax_result, axis=-1)
            conf_array.append(best_confidences)

            inds_max = np.argmax(softmax_result, axis=-1)
            inds_array.append(inds_max)


        conf_array, inds_array = list(map(np.dstack, [conf_array, inds_array]))

        # filter out only those, that are confident enough
        mask = conf_array > conf_threshold

        xs = x[mask]
        ys = y[mask]
        ws = w[mask]
        hs = h[mask]
        classes = inds_array[mask]
        scores = conf_array[mask]


        res = np.stack([xs, ys, ws, hs, scores, classes]).T
        #print(res)
        # needed to perform NMS on each class separately
        sorted_by_classes = [[]] * self.numClasses

        for box in res:
            class_index = int(box[-1])
            sorted_by_classes[class_index].append(box)

        result = []
        for class_group in sorted_by_classes:
            new_classes = non_maximal_suppression(class_group, iouTreshold)
            result.extend(new_classes)
            
            
        if len(result):
            return result
        else: 
            return None
    



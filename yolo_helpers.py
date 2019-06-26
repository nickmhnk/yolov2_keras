import numpy as np
import cv2


def rbox_to_polygon(rbox):
    cx, cy, w, h, theta = rbox
    box = np.array([[-w,h],[w,h],[w,-h],[-w,-h]]) / 2.
    box = np.dot(box, rot_matrix(theta))
    box += rbox[:2]
    return box



def scale2orig(value, orig_dim, scale_dim):
    return value * orig_dim / scale_dim


def insideBox(x, y, xc, yc, w_box, h_box):
    x_orig = xc - w_box / 2
    y_orig = yc - h_box / 2

    if x > x_orig and x < x_orig + w_box \
       and y > y_orig and y < y_orig + h_box:
        return True

    return False


def filter_boxes(segmented_boxes, centers, res_boxes, CLASSES2BAN):
    '''
    cetneters - result of prior util decode centers of textboxes
    segmented boxes - result of get_segmented_boxes:
        scaled to original image boxes of segments: xc yc w h score class
    CLASSES2BAN - id of classes to be skipped
    '''

    print('centers.shape', centers.shape)
    index2ban = []

    # for every box
    for b in segmented_boxes:
        xc, yc, w, h = b[0:4]
        box_class = int(b[-1])

        for i, center in enumerate(centers):
            if i in index2ban:
                continue

            #polygon = rbox_to_polygon(b[:5])
            avg_x = center[0]
            avg_y = center[1]

            if box_class in CLASSES2BAN and insideBox(
                avg_x, avg_y, xc, yc, w, h):
                #print("continue")
                index2ban.append(i)
                continue

    #print(index2ban)
    filtered_boxes = [res_boxes[i] for i in range(len(res_boxes)) if i not in
                      index2ban]

    return np.array(filtered_boxes)



def calc_centers_form_decode(
        res_boxes, orig_w, orig_h, IMG_W, IMG_H):
    scaled_polygons = []

    for b in res_boxes:
        # make polygon first
        polygon = rbox_to_polygon(b[:5])
        scaled_polygon = []

        # scale each point in polygon
        for point in polygon:
            nx = scale2orig(point[0], orig_w, IMG_W)
            ny = scale2orig(point[1], orig_h, IMG_H)
            new_point = [nx, ny]
            scaled_polygon.append(new_point)

        scaled_polygons.append(scaled_polygon)

    scaled_polygons = np.array(scaled_polygons)

    # calculate centers of each polygon
    centers = []
    for polygon in scaled_polygons:
        xc = polygon[:, 0].mean()
        yc = polygon[:, 1].mean()
        centers.append([xc, yc])
    centers = np.array(centers)

    return centers







def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p




def intersection_over_union(bbox1, bbox2):
    """
    Calculates the intersection over union measure of two bounding boxes
    """

    xc1, yc1, width1, height1, score1 = bbox1
    xc2, yc2, width2, height2, score2 = bbox2

    right1 = xc1 + width1 / 2
    bottom1 = yc1 + height1 / 2

    top1 = yc1 - height1 / 2
    top2 = yc2 - height2 / 2

    left1 = xc1 - width1 / 2
    left2 = xc2 - width2 / 2

    right2 = xc2 + width2 / 2
    bottom2 = yc2 + height2 /2

    intersect_width = max(0, min(right1, right2) - max(left1, left2))
    intersect_height = max(0, min(bottom1, bottom2) - max(top1, top2))
    intersection = intersect_width * intersect_height

    area1 = width1 * height1
    area2 = width2 * height2
    union = area1 + area2 - intersection

    return intersection / float(union)



def non_maximal_suppression(detections, threshold=0.5):
    """
    Performs non-maximal suppression using intersection over union measure and picking the max score detections
    Bounding box format: x, y, w, h, score
    Input: [(11, 11, 24, 24, 0.75), (10, 11, 20, 20, 0.8), (11, 9, 24, 24, 0.7), (40, 42, 20, 20, 0.6)]
    Output: [(10, 11, 20, 20, 0.8), (40, 42, 20, 20, 0.6)]
    """
    if len(detections) < 2:
        return detections

    detections = list(detections)

    # fourth index corresponds to detection score
    detections.sort(key=lambda x:x[4], reverse=True)
    final_detections = []
    for bbox in detections:
        for final_bbox in final_detections:
            iou = intersection_over_union(bbox[:-1], final_bbox[:-1])
            assert 0 <= iou <= 1
            if iou > threshold:
                break
        else:
            final_detections.append(bbox)

    return final_detections





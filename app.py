from setup.tf_model_object_detection import Model 
from setup.config import *
import numpy as np
import imutils
import cv2

with open(COCO_NAMEPATH, "r") as f:
    className = [line.strip() for line in f.readlines()]

class App:
    def __init__(self, VIDEOPATH):
        self.video = cv2.VideoCapture(VIDEOPATH)
        self.modelpath = MODELPATH
        self.main()

    def get_box_detection(self,boxes,scores,classes,height,width):
        array_boxes = list()
        for i in range(boxes.shape[1]):
            if scores[i] > 0.75:
                box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
                array_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), className[int(classes[i])-1], scores[i]))            
        return array_boxes

    def main(self):
        
        model = Model(self.modelpath)

        while True:
            self.flag, self.frame = self.video.read()
            if self.flag:
                self.frame = imutils.resize(self.frame, width=int(self.frame.shape[1]))
            else:
                break

            (boxes, scores, classes) =  model.predict(self.frame)
            detectedBox = self.get_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),self.frame.shape[0],self.frame.shape[1])
            for bbox in detectedBox:
                ymin = bbox[0]
                xmin = bbox[1]
                ymax = bbox[2]
                xmax = bbox[3]
                objectname = bbox[4]
                percent = int(bbox[5]*100)

                cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), BLACK, 2)

                label = f"{objectname} {'%.1d' % percent}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y1label = max(ymin, labelSize[1])
                cv2.rectangle(self.frame, (xmin, y1label - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), BLACK, cv2.FILLED)
                cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

            cv2.imshow("OD", self.frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        self.video.release()

if __name__ == '__main__':
    App(VIDEOPATH)
    cv2.destroyAllWindows()
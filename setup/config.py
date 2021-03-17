import os

VIDEOPATH = os.path.join(os.getcwd(), 'videos', 'TownCentre.mp4')
MODELPATH = os.path.join(os.getcwd(),'models','faster_rcnn_inception_v2_coco_2018_01_28','frozen_inference_graph.pb')
COCO_NAMEPATH = os.path.join(os.getcwd(), 'models', 'coco.names')

BLACK = (0,0,0)
WHITE = (255,255,255)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from PIL import Image
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo
from adet.config import get_cfg
from adet.utils.visualizer import TextVisualizer
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy

import pandas as pd

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list([])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args["confidence_threshold"]
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    cfg.MODEL.WEIGHTS = args["model_weights"]
    cfg.freeze()
    return cfg

class Spotter:
    def __init__(self, config_path, weight_path, confidence_thresh = 0.3) -> None:
        self.args = {}
        self.args["config_file"] = config_path
        if os.path.isfile(config_path) == False:
            print("Configuration file path is not correct.")
        self.args["model_weights"] = weight_path
        if os.path.isfile(weight_path) == False:
            print("Weights file path is not correct.")
        self.args["confidence_threshold"] = confidence_thresh
        self.cfg = setup_cfg(self.args)

        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.instance_mode = ColorMode.IMAGE

        self.predictor = VisualizationDemo(self.cfg)
        self.cpu_device = torch.device("cpu")
        self.images = []
        self.offset_xs = []
        self.offset_ys = []
        self.instances = []
        self.combined_image = None
        return
    
    def load_batch(self, images, offset_xs, offset_ys):
        self.images.clear()
        self.offset_xs.clear()
        self.offset_ys.clear()
        self.instances.clear()
        self.combined_image = None
        
        self.images = images
        self.offset_xs = offset_xs
        self.offset_ys = offset_ys
        
        return

    def inference_single(self, image:Image, offset_x = 0, offset_y = 0, scale = 1.0):
        image = _apply_exif_orientation(image)
        image =  convert_PIL_to_numpy(image, format = "BGR")
        predictions, poly_text_score_dict_list = self.predictor.inference_on_image(image)
        self.instances.append(predictions["instances"].to(self.cpu_device))

        for poly_text_score_dict in poly_text_score_dict_list:
            poly_text_score_dict["polygon_x"][:] = poly_text_score_dict["polygon_x"][:] * scale + offset_x
            poly_text_score_dict["polygon_y"][:] = poly_text_score_dict["polygon_y"][:] * scale + offset_y

        return poly_text_score_dict_list

    def inference_batch(self):
        for image, offset_x, offset_y in zip(self.images, self.offset_xs, self.offset_ys):
            self.inference_single(image, offset_x, offset_y)

    def draw(self, output_path = None, draw_instances = None, draw_offset_xs = None, draw_offset_ys = None):
        # Combine images based on offsets
        if len(self.images) == 0:
            print("No images loaded.")
            return
        
        w = max([image.size[0] + offset_x for image, offset_x in zip(self.images, self.offset_xs)])
        h = max([image.size[1] + offset_y for image, offset_y in zip(self.images, self.offset_ys)])

        self.combined_image = Image.new("RGB", (h,w))
        for image, offset_x, offset_y in zip(self.images, self.offset_xs, self.offset_ys):
            self.combined_image.paste(image, (offset_y, offset_x))
        #self.combined_image.save("combined.png")

        # Draw instances
        self.vis_final = self.combined_image.copy()
        visualizer = TextVisualizer(self.vis_final, self.metadata, instance_mode=self.instance_mode, cfg=self.cfg)
        
        if draw_instances == None or draw_offset_xs == None or draw_offset_ys == None:
            draw_instances = self.instances
            draw_offset_xs = self.offset_xs
            draw_offset_ys = self.offset_ys

        for instances, offset_x, offset_y in zip(draw_instances, draw_offset_xs, draw_offset_ys):
            if visualizer.use_polygon:
                ctrl_pnts = instances.polygons.numpy()
            else:
                ctrl_pnts = instances.beziers.numpy()
            scores = instances.scores.tolist()
            recs = instances.recs

            color = (0.1, 0.2, 0.5)
            alpha = 0.5

            for ctrl_pnt, rec, score in zip(ctrl_pnts, recs, scores):
                polygon = visualizer._ctrl_pnt_to_poly(ctrl_pnt)

                for p in polygon:
                    p[0] += offset_y
                    p[1] += offset_x

                visualizer.draw_polygon(polygon, color, alpha=alpha)

                # draw text in the top left corner
                
                text = visualizer._decode_recognition(rec)
                text = "{:.3f}: {}".format(score, text)
                lighter_color = visualizer._change_color_brightness(color, brightness_factor=0.7)
                text_pos = polygon[0]
                horiz_align = "left"
                font_size = visualizer._default_font_size * 0.2

                visualizer.draw_text(
                    text,
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                    draw_chinese=False if visualizer.voc_size == visualizer.custom_VOC_SIZE else True
                )

        vis_image = visualizer.output.get_image()
        self.vis_final = Image.fromarray(vis_image)

        if output_path != None:
            self.vis_final.save(output_path)

        return self.vis_final
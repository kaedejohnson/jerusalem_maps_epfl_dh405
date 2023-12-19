# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from PIL import Image, ImageDraw
import shapely as sh
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo
from adet.config import get_cfg
from adet.utils.visualizer import TextVisualizer
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy

import pandas as pd
import BezierSpline as BS
import math
def rotate_point(point, center, angle_degrees, clockwise=False):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees) * (-1 if clockwise else 1)

    # Extract coordinates
    x, y = point
    cx, cy = center
    cos = math.cos(angle_radians)
    sin = math.sin(angle_radians)

    # Calculate new coordinates after rotation
    new_x = (x - cx) * cos - (y - cy) * sin + cx
    new_y = (x - cx) * sin + (y - cy) * cos + cy

    return new_x, new_y

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
    def __init__(self, config_path, weight_path, confidence_thresh = 0.3, draw_thresh = 0.3, batch_mode = False) -> None:
        self.args = {}
        self.args["config_file"] = config_path
        if os.path.isfile(config_path) == False:
            print("Configuration file path is not correct.")
        self.args["model_weights"] = weight_path
        if os.path.isfile(weight_path) == False:
            print("Weights file path is not correct.")
        self.args["confidence_threshold"] = confidence_thresh
        self.confidence_thresh = confidence_thresh
        self.cfg = setup_cfg(self.args)
        self.draw_thresh = draw_thresh
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.instance_mode = ColorMode.IMAGE

        self.predictor = VisualizationDemo(self.cfg,batch_mode=batch_mode)
        self.batch_mode = batch_mode
        self.cpu_device = torch.device("cpu")
        self.images = []
        self.offset_xs = []
        self.offset_ys = []
        self.instances = []
        self.combined_image = None
        self.enhance_rotation = 1
        return
    
    def set_enhance_rotation(self, enhance_rotation):
        self.enhance_rotation = enhance_rotation

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
        size = image.size
        center = (size[0] / 2, size[1] / 2)

        images = []
        rotations = [0, 90, 270, 180]
        poly_text_score_dict_list_rotated_lst = []
        for i in range(self.enhance_rotation):
            if rotations[i] == 0:
                image_rotated = image
            else:
                image_rotated = image.rotate(rotations[i])
            image_rotated_np =  convert_PIL_to_numpy(image_rotated, format = "BGR")
            images.append(image_rotated_np)

        if self.batch_mode:
            _, poly_text_score_dict_list_rotated_lst = self.predictor.inference_on_batch(images)
        else:
            for img in images:
                _, poly_text_score_dict_list_rotated = self.predictor.inference_on_image(img)
                poly_text_score_dict_list_rotated_lst.append(poly_text_score_dict_list_rotated)
        
        poly_text_score_dict_list = []
        for i in range(self.enhance_rotation):
            if rotations[i] == 0:
                poly_text_score_dict_list.extend(poly_text_score_dict_list_rotated_lst[i])
            for poly_text_score_dict_rotated in poly_text_score_dict_list_rotated_lst[i]:
                poly_text_score_dict_rotated["polygon_x"][:], poly_text_score_dict_rotated["polygon_y"][:] = zip(*[rotate_point((x, y), center, rotations[i], clockwise=False) for x, y in zip(poly_text_score_dict_rotated["polygon_x"], poly_text_score_dict_rotated["polygon_y"])])
                poly_text_score_dict_list.append(poly_text_score_dict_rotated)
        for poly_text_score_dict in poly_text_score_dict_list:
            poly_text_score_dict["polygon_x"][:] = poly_text_score_dict["polygon_x"][:] * scale + offset_x
            poly_text_score_dict["polygon_y"][:] = poly_text_score_dict["polygon_y"][:] * scale + offset_y
        
        final_lst = [p for p in poly_text_score_dict_list if p['score'] > self.confidence_thresh]
        return final_lst

    def rotate_images(self, images, rotations):
        images_rotated = []
        for i in range(self.enhance_rotation):
            image_rotated = images.rotate(rotations[i])
            image_rotated_np =  convert_PIL_to_numpy(image_rotated, format = "BGR")
            images_rotated.append(image_rotated_np)
        return images_rotated
    
    def rotate_polygons(self, poly_text_score_dict_list_rotated_lst, rotations, center, offset_x, offset_y):
        poly_text_score_dict_list = []
        for i in range(self.enhance_rotation):
            for poly_text_score_dict_rotated in poly_text_score_dict_list_rotated_lst[i]:
                poly_text_score_dict_rotated["polygon_x"][:], poly_text_score_dict_rotated["polygon_y"][:] = zip(*[rotate_point((x, y), center, rotations[i], clockwise=False) for x, y in zip(poly_text_score_dict_rotated["polygon_x"], poly_text_score_dict_rotated["polygon_y"])])
                poly_text_score_dict_list.append(poly_text_score_dict_rotated)
        for poly_text_score_dict in poly_text_score_dict_list:
            poly_text_score_dict["polygon_x"][:] = poly_text_score_dict["polygon_x"][:] + offset_x
            poly_text_score_dict["polygon_y"][:] = poly_text_score_dict["polygon_y"][:] + offset_y
    
        return poly_text_score_dict_list

    def inference_batch(self, batch_limitation = 0):
        combined_results = []
        length = len(self.images)
        if self.batch_mode:
            j = 0
            rotations = [0, 90, 270, 180]
            while j < length:
                size = self.images[j].size
                overall_size = size[0] * self.enhance_rotation * size[1]
                batch_size = batch_limitation // overall_size + 1
                if batch_size > 5:
                    batch_size = 5
                j += batch_size
                ids = [i for i in range(j - batch_size, j) if i < length]
                center = (size[0] / 2, size[1] / 2)

                images = []
                for id in ids:
                    images.extend(self.rotate_images(self.images[id], rotations))

                raw_poly_text_score_dict_list_rotated_lst = []
                if self.batch_mode:
                    _, raw_poly_text_score_dict_list_rotated_lst = self.predictor.inference_on_batch(images)
                else:
                    for img in images:
                        _, poly_text_score_dict_list_rotated = self.predictor.inference_on_image(img)
                        raw_poly_text_score_dict_list_rotated_lst.append(poly_text_score_dict_list_rotated)
                
                poly_text_score_dict_list_rotated_lst = []
                # Group elems in raw_poly_text_score_dict_list_rotated_lst by self.enhance_rotation
                for i in range(len(ids)):
                    poly_text_score_dict_list_rotated_lst.append(raw_poly_text_score_dict_list_rotated_lst[i * self.enhance_rotation : (i + 1) * self.enhance_rotation])

                poly_text_score_dict_list = []
                for i, id in enumerate(ids):
                    poly_text_score_dict_list += self.rotate_polygons(poly_text_score_dict_list_rotated_lst[i], rotations, center, self.offset_xs[id], self.offset_ys[id])
                
                final_lst = [p for p in poly_text_score_dict_list if p['score'] > self.confidence_thresh]

                combined_results.extend(final_lst)

                print(f"Processed {ids[-1] + 1}/{length}, {(ids[-1] + 1)/length*100:.2f}%")
        else:
            j = 0
            for image, offset_x, offset_y in zip(self.images, self.offset_xs, self.offset_ys):
                combined_results.extend(self.inference_single(image, offset_x, offset_y))
                j += 1
                print(f"Processed {j}/{length}, {j/length*100:.2f}%")
        return combined_results

    def draw(self, output_path = None, draw_instances = None, draw_offset_xs = None, draw_offset_ys = None):
        # Combine images based on offsets
        if len(self.images) == 0:
            print("No images loaded.")
            return
        
        h = max([image.size[0] + offset_x for image, offset_x in zip(self.images, self.offset_xs)])
        w = max([image.size[1] + offset_y for image, offset_y in zip(self.images, self.offset_ys)])

        self.combined_image = Image.new("RGB", (h,w))
        for image, offset_x, offset_y in zip(self.images, self.offset_xs, self.offset_ys):
            self.combined_image.paste(image, (offset_x, offset_y))
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
                if score < self.draw_thresh:
                    continue
                polygon = visualizer._ctrl_pnt_to_poly(ctrl_pnt)
                polygon[:, 0] += offset_x
                polygon[:, 1] += offset_y

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
    
class PolygonVisualizer:
    def __init__(self):
        self.canvas = None
        self.cfg = get_cfg()
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )

    def canvas_from_patches(self, image_batch:list, offset_xs:list, offset_ys:list):
        h = max([image.size[0] + offset_x for image, offset_x in zip(image_batch, offset_xs)])
        w = max([image.size[1] + offset_y for image, offset_y in zip(image_batch, offset_ys)])
        self.canvas = Image.new("RGB", (h,w))
        for image, offset_x, offset_y in zip(image_batch, offset_xs, offset_ys):
            self.canvas.paste(image, (offset_x, offset_y))

    def canvas_from_image(self, image:Image):
        self.canvas = image.copy()

    def draw(self, json_list:list):
        if self.canvas == None:
            print("No canvas loaded.")
            return
        
        visualizer = TextVisualizer(self.canvas, self.metadata, instance_mode=ColorMode.IMAGE, cfg=self.cfg)
        for json in json_list:
            polygon_x = json["polygon_x"]
            polygon_y = json["polygon_y"]
            polygon = []
            for x, y in zip(polygon_x, polygon_y):
                polygon.append([x, y])

            color = (0.1, 0.2, 0.5)
            alpha = 0.5
            visualizer.draw_polygon(polygon, color, alpha=alpha)

            # draw text in the top left corner
            text = json["text"]
            score = json["score"]
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

        return self.vis_final
    
    def draw_poly(self, polygon_list:list, text_list:list, PCA_feature_list:list, BSplines:list = None):
        if self.canvas == None:
            print("No canvas loaded.")
            return
        
        visualizer = TextVisualizer(self.canvas, self.metadata, instance_mode=ColorMode.IMAGE, cfg=self.cfg)
        i = 0
        for poly, text in zip(polygon_list, text_list):
            color = (0.1, 0.2, 0.5)
            alpha = 0.5
            polygon = []
            if isinstance(poly, sh.geometry.polygon.Polygon):
                polygon_x = poly.exterior.coords.xy[0]
                polygon_y = poly.exterior.coords.xy[1]
                polygon = []
                for x, y in zip(polygon_x, polygon_y):
                    polygon.append([x, y])
                visualizer.draw_polygon(polygon, color, alpha=alpha)
            elif isinstance(poly, sh.geometry.multipolygon.MultiPolygon):
                for p in poly.geoms: # kaede added .geoms - package version differences
                    polygon_x = p.exterior.coords.xy[0]
                    polygon_y = p.exterior.coords.xy[1]
                    polygon = []
                    for x, y in zip(polygon_x, polygon_y):
                        polygon.append([x, y])
                    visualizer.draw_polygon(polygon, color, alpha=alpha)
            # draw text in the top left corner
            
            _text = f"{i}:{text}"
            lighter_color = visualizer._change_color_brightness(color, brightness_factor=0.7)
            text_pos = polygon[0]
            horiz_align = "left"
            font_size = visualizer._default_font_size * 0.2

            visualizer.draw_text(
                _text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                draw_chinese=False if visualizer.voc_size == visualizer.custom_VOC_SIZE else True
            )

            if PCA_feature_list != None:
                for PCA_dict in PCA_feature_list[i]:
                    color_pca = (0.5, 0.2, 0.1)
                    centroid = np.array(PCA_dict["Centroid"])
                    Var = np.array(PCA_dict["PCA_Var"])
                    Std_var = np.sqrt(Var)
                    Basis1 = np.array(PCA_dict["PCA_Basis"][0])
                    Basis2 = np.array(PCA_dict["PCA_Basis"][1])
                    
                    polygon = [centroid, centroid + Basis1 * Std_var[0], centroid + Basis1 + Basis2, centroid + Basis2 * Std_var[1]]
                    visualizer.draw_polygon(polygon, color_pca, alpha=1)


            i += 1
        if BSplines != None:
            for spline in BSplines:
                polygon = spline[0].get_as_polygon(8)
                color_spline = (0.1, 0.5, 0.2)
                lighter_color = visualizer._change_color_brightness(color_spline, brightness_factor=0.7)
                visualizer.draw_polygon(polygon, lighter_color, alpha=1)
                text = f"{spline[1]:.3f}"
                text_pos = polygon[1]
                horiz_align = "left"
                font_size = visualizer._default_font_size * 0.1
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

        return self.vis_final
    

    def save(self, output_path):
        self.vis_final.save(output_path)
        return
    
    def save_json(self, json_list:list, output_path):
        df = pd.DataFrame(json_list)
        df.to_json(output_path)
        return
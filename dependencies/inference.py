# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

import pandas as pd

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/SynMap_Polygon.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--output_json", action="store_true", help="Save outputs to json instead of image")
    
    # added for fdh
    parser.add_argument("--model_weights", help="Location for pre-trained text spotter weights")
  
    parser.add_argument("--curr_wd", help="wd from overhead call")
    parser.add_argument("--map_crops_topfolder", help="cropped images location from overhead call")
    parser.add_argument("--map_streg_topfolder", help="scene text location from overhead call")
    
    parser.add_argument("--inputs", help="set to True (or anything) if you want it to run over all crops")

    return parser


if __name__ == "__main__":
    
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()

            if args.output_json:
                # modified code block to save output to json
                predictions, poly_text_score_dict_list = demo.inference_on_image(img)

                # output json file instead of visualization
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path).split('.')[0] + '.json')
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                #pdb.set_trace()

                df = pd.DataFrame(poly_text_score_dict_list)
                df.to_json(out_filename)


            else:
                # this code block is the original process, saves the output to image
                predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
                        
    elif args.inputs:
    
        # Set up input-output locations
        curr_wd = args.curr_wd
        map_crops_topfolder = args.map_crops_topfolder
        map_streg_topfolder = args.map_streg_topfolder
        
        in_out_pairs = []
        for root, dirs, files in os.walk(map_crops_topfolder):
            if len(files) > 0:
                for i in range(len(files)):
                    in_tmp = curr_wd + "/" + root.replace("\\","/") + "/" + files[i]
                    out_tmp = in_tmp.replace(map_crops_topfolder,map_streg_topfolder).rsplit("/",1)[0]
                    in_out_pairs.append([in_tmp, out_tmp])
        #print(in_out_pairs[0])
        
        for pair in in_out_pairs:
            pair_input = pair[0]
            pair_output = pair[1]
            
            if os.path.isdir(pair_input):
                pair_input = [os.path.join(pair_input[0], fname) for fname in os.listdir(pair_input[0])]
            elif len(pair_input) == 1:
                pair_input = glob.glob(os.path.expanduser(pair_input[0]))
                assert pair_input, "The input path(s) was not found"
            
            pair_input = [pair_input]
            for path in tqdm.tqdm(pair_input, disable=not pair_output):
                # use PIL, to be consistent with evaluatio
                img = read_image(path, format="BGR")
                start_time = time.time()


                        
                if args.output_json:
                    # modified code block to save output to json
                    predictions, poly_text_score_dict_list = demo.inference_on_image(img)

                    # output json file instead of visualization
                    logger.info(
                        "{}: detected {} instances in {:.2f}s".format(
                            path, len(predictions["instances"]), time.time() - start_time
                        )
                    )
                    
                    print(pair_output)
                    if os.path.isdir(pair_output):
                        assert os.path.isdir(pair_output), pair_output
                        out_filename = os.path.join(pair_output, os.path.basename(path).split('.')[0] + '.json')
                    else:
                        assert len(pair_input) == 1, "Please specify a directory with args.output"
                        out_filename = pair_output
                    #pdb.set_trace()
                    print(out_filename)
                    df = pd.DataFrame(poly_text_score_dict_list)
                    df.to_json(out_filename)


                else:
                    # this code block is the original process, saves the output to image
                    predictions, visualized_output = demo.run_on_image(img)
                    logger.info(
                        "{}: detected {} instances in {:.2f}s".format(
                            path, len(predictions["instances"]), time.time() - start_time
                        )
                    )

                    if pair_output:
                        if os.path.isdir(pair_output):
                            assert os.path.isdir(pair_output), pair_output
                            out_filename = os.path.join(pair_output, os.path.basename(path))
                        else:
                            assert len(pair_input) == 1, "Please specify a directory with args.output"
                            out_filename = pair_output
                        visualized_output.save(out_filename)
                    else:
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import copy
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, MACOS, SETTINGS, WINDOWS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = self.get_save_dir()
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.plotted_img2 = None
        self.data_path = None
        self.data_path2 = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.results2 = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def get_save_dir(self):
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        return increment_path(Path(project) / name, exist_ok=self.args.exist_ok)

    def preprocess(self, im, im2):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
            im2 (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor, not_tensor2 = not isinstance(im, torch.Tensor), not isinstance(im2, torch.Tensor)
        if not_tensor or not_tensor2:
            p = self.pre_transform(im, im2)
            im, im2 = np.stack([item[0] for item in p]), np.stack([item[1] for item in p])
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            im2 = im2[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im2 = np.ascontiguousarray(im2)  # contiguous
            im2 = torch.from_numpy(im2)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        img2 = im2.to(self.device)
        img2 = img2.half() if self.model.fp16 else img2.float()  # uint8 to fp16/32
        if not_tensor2:
            img2 /= 255  # 0 - 255 to 0.0 - 1.0
        return img, img2

    def pre_transform(self, im, im2):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
            im2 (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt
        return [LetterBox(self.imgsz, auto=auto, stride=self.model.stride)(image=x, image2=y) for x, y in zip(im, im2)]

    def write_results_vi(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset[0].count
        else:
            frame = getattr(self.dataset[0], 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels_vi' / p.stem) + (
            '' if self.dataset[0].mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def write_results_ir(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset[0].count
        else:
            frame = getattr(self.dataset[0], 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels_ir' / p.stem) + (
            '' if self.dataset[0].mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                'line_width': self.args.line_width,
                'boxes': self.args.boxes,
                'conf': self.args.show_conf,
                'labels': self.args.show_labels}
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img2 = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            return list(self.stream_inference(source, model))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset[0].source_type
        if not getattr(self, 'stream', True) and (self.dataset[0].mode == 'stream' or  # streams
                                                  len(self.dataset[0]) > 1000 or  # images
                                                  any(getattr(self.dataset[0], 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset[0].bs, [None] * self.dataset[0].bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)

        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset[0].bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')

        for batch1, batch2 in zip(self.dataset[0], self.dataset[1]):
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch1, batch2
            path, im0s, vid_cap, s = batch1
            path2, im0s2, vid_cap2, s2 = batch2
            visualize = increment_path(self.save_dir / Path(path).stem,
                                       mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            # Preprocess
            with profilers[0]:
                im, im2 = self.preprocess(im0s, im0s2)

            # Inference
            with profilers[1]:
                preds = self.model(im, im2, augment=self.args.augment, visualize=visualize)

            # Postprocess
            with profilers[2]:
                preds2 = copy.deepcopy(preds)
                self.results = self.postprocess(preds, im, im0s)
                self.results2 = self.postprocess(preds2, im2, im0s2)
            self.run_callbacks('on_predict_postprocess_end')

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                self.seen += 1
                self.results[i].speed = {
                    'preprocess': profilers[0].dt * 1E3 / n,
                    'inference': profilers[1].dt * 1E3 / n,
                    'postprocess': profilers[2].dt * 1E3 / n}
                p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                p = Path(p)

                p2, im02 = path2[i], None if self.source_type.tensor else im0s2[i].copy()
                p2 = Path(p2)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results_vi(i, self.results, (p, im, im0))
                    s2 += self.write_results_ir(i, self.results2, (p2, im2, im02))
                if self.args.save or self.args.save_txt:
                    self.results[i].save_dir = self.save_dir.__str__()
                    self.results2[i].save_dir = self.save_dir.__str__()
                if self.args.show and self.plotted_img is not None and self.plotted_img2 is not None:
                    self.show(p)
                    self.show(p2)
                if self.args.save and self.plotted_img is not None and self.plotted_img2 is not None:
                    path_1 = Path(self.save_dir / 'RGB')
                    path_2 = Path(self.save_dir / 'IR')
                    if not path_1.exists():
                        path_1.mkdir(parents=True, exist_ok=True)
                    if not path_2.exists():
                        path_2.mkdir(parents=True, exist_ok=True)
                    self.save_preds(vid_cap, vid_cap2, i, str(path_1 / p.name), str(path_2 / p2.name))

            self.run_callbacks('on_predict_batch_end')
            yield from self.results
            yield from self.results2

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'VIS: {s}\nIR: {s2}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image pair at shape '
                        f'{(1, 3, *im.shape[2:])} and {(1, 3, *im2.shape[2:])}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=True,
                                 verbose=verbose)
        self.device = device
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, vid_cap2, idx, save_path, save_path2):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        im02 = self.plotted_img2
        # Save imgs
        if self.dataset[0].mode == 'image' and self.dataset[1].mode == 'image':
            cv2.imwrite(save_path, im0)
            cv2.imwrite(save_path2, im02)
        else:  # 'video' or 'stream'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix = '.mp4' if MACOS else '.avi' if WINDOWS else '.avi'
                fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'MJPG'
                save_path = str(Path(save_path).with_suffix(suffix))
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)

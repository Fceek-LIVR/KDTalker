# coding: utf-8

"""
All configs for user
"""

from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    source_image: Annotated[str, tyro.conf.arg(aliases=["-s"])] = make_abs_path('../../assets/examples/source/s6.jpg')  # path to the source portrait
    driving_info:  Annotated[str, tyro.conf.arg(aliases=["-d"])] = make_abs_path('../../assets/examples/driving/d12.mp4')  # path to driving video or template (.pkl format)
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'animations/'  # directory to save output video

    ########## inference arguments ##########
    flag_use_half_precision: bool = False  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
    flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
    device_id: int = 0 # gpu device id
    flag_force_cpu: bool = False # force cpu inference, WIP!
    flag_lip_zero: bool = False # whether let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False
    flag_eye_retargeting: bool = False # not recommend to be True, WIP
    flag_lip_retargeting: bool = False # not recommend to be True, WIP
    flag_stitching: bool = False  # recommend to True if head movement is small, False if head movement is large
    flag_relative_motion: bool = False  # whether to use relative motion
    flag_pasteback: bool = False  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space
    flag_do_crop: bool = False  # whether to crop the source portrait to the face-cropping space
    flag_do_rot: bool = False  # whether to conduct the rotation when flag_do_crop is True

    ########## crop arguments ##########
    scale: float = 2.3 # the ratio of face area is smaller if scale is larger
    vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
    vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space

    scale_crop_video: float = 2.2 # scale factor for cropping video
    vx_ratio_crop_video: float = 0. # adjust y offset
    vy_ratio_crop_video: float = -0.1  # adjust x offset

    ########## gradio arguments ##########
    server_port: Annotated[int, tyro.conf.arg(aliases=["-p"])]  = 8890 # port for gradio server
    share: bool = False # whether to share the server to public
    server_name: Optional[str] = "127.0.0.1"  # set the local server name, "0.0.0.0" to broadcast all
    flag_do_torch_compile: bool = False  # whether to use torch.compile to accelerate generation

    # Custom for realtime inference
    concurrent: bool = False # whether to use concurrent processing mode

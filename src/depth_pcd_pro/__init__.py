from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2 as cv
import open3d as o3d

@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float = 0.1  # meters

    @classmethod
    def from_image(cls, img_shape, fx, fy=None, cx=None, cy=None, baseline=0.1):
        h, w = img_shape[:2]
        if fy is None: fy = fx
        if cx is None: cx = w / 2.0
        if cy is None: cy = h / 2.0
        return cls(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, baseline=baseline)

    def intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)

def disparity_bm(left_gray: np.ndarray, right_gray: np.ndarray, num_disp: int = 192, block_size: int = 15) -> np.ndarray:
    bm = cv.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    disp = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp

def disparity_sgbm(left_gray: np.ndarray, right_gray: np.ndarray, num_disp: int = 192, block_size: int = 5) -> np.ndarray:
    sgbm = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8*3*block_size**2,
        P2=32*3*block_size**2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp

def depth_from_disparity(disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    depth = (fx * baseline_m) / disp
    return depth.astype(np.float32)

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    m = np.nanmin(img)
    M = np.nanmax(img)
    if not np.isfinite(m) or not np.isfinite(M) or M == m:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - m) / (M - m)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

def pcd_from_depth(depth_m: np.ndarray, cam: Camera) -> o3d.geometry.PointCloud:
    depth_o3d = o3d.geometry.Image(depth_m)
    return o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, cam.intrinsic())

def midas_depth(rgb_bgr: np.ndarray, model_name: str = "DPT_Large") -> np.ndarray:
    import torch
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    if "Large" in model_name or "Hybrid" in model_name:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    input_batch = transform(cv.cvtColor(rgb_bgr, cv.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy().astype(np.float32)
    return depth

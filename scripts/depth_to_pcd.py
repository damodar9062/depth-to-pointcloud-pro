import argparse, os
import numpy as np
import cv2 as cv
import open3d as o3d
from depth_pcd_pro import (
    Camera, disparity_bm, disparity_sgbm, depth_from_disparity,
    normalize_to_uint8, pcd_from_depth, midas_depth
)

def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv.imwrite(path, img)

def cmd_stereo(args):
    L = cv.imread(args.left, cv.IMREAD_GRAYSCALE)
    R = cv.imread(args.right, cv.IMREAD_GRAYSCALE)
    assert L is not None and R is not None, "Failed to read stereo images"
    cam = Camera.from_image(L.shape, fx=args.fx, fy=args.fy or args.fx, cx=args.cx or L.shape[1]/2, cy=args.cy or L.shape[0]/2, baseline=args.baseline)
    disp = disparity_sgbm(L, R, num_disp=args.num_disp, block_size=args.block)
    depth = depth_from_disparity(disp, fx=cam.fx, baseline_m=cam.baseline)
    depth_u8 = normalize_to_uint8(depth)
    save_img(depth_u8, args.depth_out)
    pcd = pcd_from_depth(depth, cam)
    o3d.io.write_point_cloud(args.pcd_out, pcd)
    print(f"Saved depth → {args.depth_out}")
    print(f"Saved pcd   → {args.pcd_out}")

def cmd_midas(args):
    img = cv.imread(args.image, cv.IMREAD_COLOR)
    assert img is not None, "Failed to read image"
    cam = Camera.from_image(img.shape, fx=args.fx, fy=args.fy or args.fx, cx=args.cx or img.shape[1]/2, cy=args.cy or img.shape[0]/2, baseline=args.baseline)
    depth = midas_depth(img, model_name=args.midas_model)
    depth_u8 = normalize_to_uint8(depth)
    save_img(depth_u8, args.depth_out)
    pcd = pcd_from_depth(depth, cam)
    o3d.io.write_point_cloud(args.pcd_out, pcd)
    print(f"Saved depth → {args.depth_out}")
    print(f"Saved pcd   → {args.pcd_out}")

def main():
    ap = argparse.ArgumentParser(prog="depth_to_pcd")
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fx", type=float, default=1734.04)
    common.add_argument("--fy", type=float)
    common.add_argument("--cx", type=float)
    common.add_argument("--cy", type=float)
    common.add_argument("--baseline", type=float, default=0.12, help="meters")

    s = sub.add_parser("stereo", parents=[common])
    s.add_argument("--left", required=True)
    s.add_argument("--right", required=True)
    s.add_argument("--num-disp", type=int, default=192)
    s.add_argument("--block", type=int, default=5)
    s.add_argument("--depth-out", required=True)
    s.add_argument("--pcd-out", required=True)
    s.set_defaults(func=cmd_stereo)

    m = sub.add_parser("midas", parents=[common])
    m.add_argument("--image", required=True)
    m.add_argument("--midas-model", default="DPT_Large", choices=["DPT_Large","DPT_Hybrid","MiDaS_small"])
    m.add_argument("--depth-out", required=True)
    m.add_argument("--pcd-out", required=True)
    m.set_defaults(func=cmd_midas)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

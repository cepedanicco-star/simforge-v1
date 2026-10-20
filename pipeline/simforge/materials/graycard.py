"""Explicit card ROIs, linear display luminance, and one-parameter decisions."""
import math

import numpy as np


def srgb_to_linear(rgb):
    x=np.asarray(rgb,dtype=np.float64)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def linear_to_srgb(linear):
    x=np.asarray(linear,dtype=np.float64)
    encoded=np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)
    return np.clip(encoded,0,1)*255


def card_rectangles(config, intrinsics=None):
    view=config["views"][0]
    width,height=config["resolution"]
    if intrinsics is None:
        fx=width/(2*math.tan(math.radians(view["horizontal_fov_degrees"]/2)))
        fy=height/(2*math.tan(math.radians(view["vertical_fov_degrees"]/2)))
        cx,cy=width/2,height/2
    else:
        fx,fy,cx,cy=(intrinsics[k] for k in ["fx","fy","cx","cy"])
    eye_z=view["usd_camera_matrix"][3][2]
    records={}
    for card in config["calibration_cards"]:
        x,y,z=card["position_m"]
        sx,sy,sz=card["dimensions_m"]
        depth=eye_z-z-sz/2
        u,v=cx+x*fx/depth,cy-y*fy/depth
        projected_w,projected_h=sx*fx/depth,sy*fy/depth
        box=[u-projected_w/2,v-projected_h/2,u+projected_w/2,v+projected_h/2]
        # Central sample is at most 40% of card width, avoiding edges/specular seams.
        rw,rh=min(256,int(projected_w*.4)),min(256,int(projected_h*.4))
        roi=[int(round(u-rw/2)),int(round(v-rh/2)),int(round(u+rw/2)),int(round(v+rh/2))]
        if min(box[:2])<0 or box[2]>width or box[3]>height or min(rw,rh)<32:
            raise ValueError("Calibration card does not fit image with a safe centre ROI")
        records[card["name"]]=dict(projected_box_xyxy=box,roi_xyxy=roi,
            projected_fraction=projected_w*projected_h/(width*height),reflectance=card["reflectance"])
    if records["gray18"]["projected_fraction"]<.30:
        raise ValueError("Gray card coverage below the requested 30 percent")
    return records


def measure_cards(rgb, config, intrinsics=None):
    rgb=np.asarray(rgb)
    if rgb.shape!=(1152,2048,3):
        raise ValueError("Invalid calibration RGB shape")
    rectangles=card_rectangles(config,intrinsics)
    for record in rectangles.values():
        x0,y0,x1,y1=record["roi_xyxy"]
        sample=rgb[y0:y1,x0:x1]
        linear=srgb_to_linear(sample)
        luminance=linear @ [.2126729,.7151522,.0721750]
        record.update(linear_luminance_mean=float(luminance.mean()),
                      linear_luminance_std=float(luminance.std()),
                      srgb_mean=float(sample.mean()),srgb_min=int(sample.min()),srgb_max=int(sample.max()),
                      clipped_fraction=float(np.any(sample>=255,axis=-1).mean()),pixels=int(luminance.size))
    return rectangles


def ratios(measured,reference):
    result={}
    for name in ["gray18","white90"]:
        denominator=reference[name]["linear_luminance_mean"]
        if denominator<=0:
            raise ValueError("UE reference ROI has zero luminance")
        result[name]=measured[name]["linear_luminance_mean"]/denominator
    return result


def acceptance(ratio):
    return dict(gray_pass=abs(ratio["gray18"]-1)<=.03,white_pass=abs(ratio["white90"]-1)<=.05,
                gray_residual=ratio["gray18"]-1,white_residual=ratio["white90"]-1)

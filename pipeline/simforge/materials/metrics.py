"""Display-referred D65 CIELAB/CIEDE2000 and full-resolution single-scale SSIM.

Formula references: Sharma/Wu/Dalal (2005) and Wang et al. (2004).
No alignment, exposure fitting, white balance fitting, or histogram matching.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    xyz = linear @ np.array([[.4124564,.3575761,.1804375],
                             [.2126729,.7151522,.0721750],
                             [.0193339,.1191920,.9503041]]).T
    xyz /= [.95047, 1, 1.08883]
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], axis=-1)


def delta_e2000(lab1, lab2):
    lab1, lab2 = np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64)
    l1,a1,b1 = np.moveaxis(lab1, -1, 0)
    l2,a2,b2 = np.moveaxis(lab2, -1, 0)
    c1,c2 = np.hypot(a1,b1), np.hypot(a2,b2)
    mean_c = (c1+c2)/2
    g = .5*(1-np.sqrt(mean_c**7/(mean_c**7+25**7)))
    ap1,ap2 = (1+g)*a1, (1+g)*a2
    cp1,cp2 = np.hypot(ap1,b1), np.hypot(ap2,b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1,ap1)),360)
    hp2 = np.mod(np.degrees(np.arctan2(b2,ap2)),360)
    hp1,hp2 = np.where(cp1==0,0,hp1),np.where(cp2==0,0,hp2)
    dl,dc = l2-l1, cp2-cp1
    dh = hp2-hp1
    zero_c = cp1*cp2 == 0
    # Tolerate only floating-point roundoff at the exact 180-degree branch.
    dh = np.where(dh > 180+1e-12, dh-360, np.where(dh < -180-1e-12,dh+360,dh))
    dh = np.where(zero_c,0,dh)
    dh_term = 2*np.sqrt(cp1*cp2)*np.sin(np.radians(dh/2))
    mean_l,mean_cp = (l1+l2)/2,(cp1+cp2)/2
    mean_h = np.where(zero_c,hp1+hp2,
        np.where(np.abs(hp1-hp2)<=180+1e-12,(hp1+hp2)/2,
                 np.where(hp1+hp2<360,(hp1+hp2+360)/2,(hp1+hp2-360)/2)))
    cos = lambda degrees: np.cos(np.radians(degrees))  # noqa: E731
    t = 1-.17*cos(mean_h-30)+.24*cos(2*mean_h)+.32*cos(3*mean_h+6)-.20*cos(4*mean_h-63)
    sl = 1+.015*(mean_l-50)**2/np.sqrt(20+(mean_l-50)**2)
    sc,sh = 1+.045*mean_cp,1+.015*mean_cp*t
    rotation = -2*np.sqrt(mean_cp**7/(mean_cp**7+25**7))*np.sin(np.radians(60*np.exp(-((mean_h-275)/25)**2)))
    return np.sqrt(np.maximum(0,(dl/sl)**2+(dc/sc)**2+(dh_term/sh)**2+rotation*(dc/sc)*(dh_term/sh)))


def ssim_map(rgb1, rgb2):
    """11x11 Gaussian sigma 1.5, population moments, K1=.01/K2=.03, range255.

    BT.601 display luma, full native resolution, no automatic downsampling.
    Discard a five-pixel border and mask windows touching non-material pixels
    when averaging this map. The same image gives exactly one.
    """
    a = np.asarray(rgb1,dtype=np.float64) @ [.299,.587,.114]
    b = np.asarray(rgb2,dtype=np.float64) @ [.299,.587,.114]
    blur = lambda x: gaussian_filter(x, sigma=1.5, radius=5, mode="reflect")  # noqa: E731
    ma,mb = blur(a),blur(b)
    va,vb = np.maximum(0,blur(a*a)-ma*ma),np.maximum(0,blur(b*b)-mb*mb)
    cov = blur(a*b)-ma*mb
    c1,c2 = (.01*255)**2,(.03*255)**2
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))

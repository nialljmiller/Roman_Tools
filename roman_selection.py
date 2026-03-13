import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u

# -----------------------------
# Inputs
# -----------------------------
FITS = "astra_gaia_xgb_bdbs.fits"     

L_COL = "l"
B_COL = "b" 
TEFF_COL = "teff"
LOGG_COL = "logg"
H_COL    = "h_mag"


TILE_W = (49.4 * u.arcmin).to(u.deg).value
TILE_H = (25.3 * u.arcmin).to(u.deg).value
RECT_PA_DEG = 90.6

LAYOUT = "overguide"

# Table 9 -- https://arxiv.org/pdf/2505.10574
CENTERS = {
    "nominal": np.array([
        [-0.622435, -1.200],
        [-0.213461, -1.200],
        [ 0.195513, -1.200],
        [ 0.604487, -1.200],
        [ 1.013461, -1.200],
        [ 1.422435, -1.200],
        [ 0.000000, -0.1250],
    ], dtype=float),

    "underguide": np.array([
        [ 0.497757, -1.005558],
        [ 0.906730, -1.005558],
        [ 1.315704, -1.005558],
        [-0.115704, -1.794442],
        [ 0.293270, -1.794442],
        [ 0.702243, -1.794442],
        [ 1.111217, -1.794442],
    ], dtype=float),

    "overguide": np.array([
        [-0.417948, -1.200],
        [-0.008974, -1.200],
        [ 0.400000, -1.200],
        [ 0.808974, -1.200],
        [ 1.217948, -1.200],
        [ 0.000000, -0.1250],
    ], dtype=float),
}



def VVVlwrap(l_deg):
    return ((l_deg + 180.0) % 360.0) - 180.0

def points_in_rotated_rectangle_gal(l_deg, b_deg, l0_deg, b0_deg, w_deg, h_deg, pa_deg):
    center = SkyCoord(l=l0_deg*u.deg, b=b0_deg*u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)

    pts = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame="galactic").transform_to(off_frame)
    dx = pts.lon.to(u.deg).value
    dy = pts.lat.to(u.deg).value

    th = np.deg2rad(pa_deg)
    xr =  dx*np.cos(th) + dy*np.sin(th)
    yr = -dx*np.sin(th) + dy*np.cos(th)

    return (np.abs(xr) <= w_deg/2) & (np.abs(yr) <= h_deg/2)

def rectangle_outline_lb(l0_deg, b0_deg, w_deg, h_deg, pa_deg):
    center = SkyCoord(l=l0_deg*u.deg, b=b0_deg*u.deg, frame="galactic")
    off_frame = SkyOffsetFrame(origin=center)

    hw, hh = w_deg/2, h_deg/2
    corners_xy = np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh],[-hw,-hh]])

    th = np.deg2rad(pa_deg)
    dx = corners_xy[:,0]*np.cos(th) - corners_xy[:,1]*np.sin(th)
    dy = corners_xy[:,0]*np.sin(th) + corners_xy[:,1]*np.cos(th)

    outline = SkyCoord(lon=dx*u.deg, lat=dy*u.deg, frame=off_frame).transform_to("galactic")
    return VVVlwrap(outline.l.deg), outline.b.deg


tab = Table.read(FITS, hdu=2)
print("rows:", len(tab))
print("cols:", tab.colnames[:20])
l = VVVlwrap(np.array(tab[L_COL], dtype=float))
b = np.array(tab[B_COL], dtype=float)


centers = CENTERS[LAYOUT].copy()
centers[:,0] = VVVlwrap(centers[:,0])

in_roman = np.zeros(len(tab), dtype=bool)
for l0, b0 in centers:
    in_roman |= points_in_rotated_rectangle_gal(l, b, l0, b0, TILE_W, TILE_H, RECT_PA_DEG)



teff = np.array(tab[TEFF_COL], dtype=float)
teff_ok = np.isfinite(teff) & (teff <= 5500.0)
# Weiss et al. 2025, 3.3: "We set asteroseismic detection criteria for objects with Teff ≤ 5250 K..."

logg = np.array(tab[LOGG_COL], dtype=float)
logg_ok = np.isfinite(logg)


# want brighter than 16th mag in Roman F146.
# treat H <= 16 as “bright regime” proxy for high detectability.
# see 2.1.2 and 4.4 
hmag = np.array(tab[H_COL], dtype=float)
bright_ok = np.isfinite(hmag) & (hmag <= 17.0)



wz = teff_ok & logg_ok & bright_ok

final = in_roman & wz

print("Weiss Zinn cuts summary (after Roman footprint):")
print("  Roman footprint:", in_roman.sum())
print("  Teff < 5250:", (in_roman & teff_ok).sum())
print("  H < 16 :", (in_roman & bright_ok).sum())
print("  FINAL Roman and seismo:", final.sum())

print(f"Layout: {LAYOUT}")
print(f"Total stars: {len(tab)}")
print(f"Inside Roman footprint: {final.sum()}")








plt.figure(figsize=(10,6))
plt.scatter(l, b, s=2, alpha=0.15, label="All stars")
plt.scatter(l[final], b[final], s=7, alpha=0.8, label=f"Inside {LAYOUT} footprint")

for i, (l0, b0) in enumerate(centers, start=1):
    ol, ob = rectangle_outline_lb(l0, b0, TILE_W, TILE_H, RECT_PA_DEG)
    plt.plot(ol, ob, linewidth=2)
    plt.text(l0, b0, str(i), fontsize=10, ha="center", va="center")

plt.gca().invert_xaxis()
plt.xlabel("Galactic l (deg)")
plt.ylabel("Galactic b (deg)")
plt.title(f"GBTDS {LAYOUT} footprint (Table 9 centers; rectangle tile approx)")
plt.legend()
plt.tight_layout()
plt.show()




OUT_FITS = f"astra_{LAYOUT}_roman_wz_selected.fits"

tab_sel = tab[final]
tab_sel.write(OUT_FITS, format="fits", overwrite=True)

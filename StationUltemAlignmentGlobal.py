#!/usr/bin/env python3

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Production station    Frame station    dX_shift_mm    dY_shift_mm    rotX_mrad    rotY_mrad    rotZ_mrad    bottom6_nominal_Z_anchor_shift_mm    real_rms_mm    mod_rms_mm    XYZ_mount_dx_mm    XYZ_mount_dy_mm    XYZ_mount_dz_mm    YZ_mount_dy_mm    YZ_mount_dz_mm
FrameOrdering=[8,5,9,16,12,11,14,7,6,17,3,4,1,0,10,15,2,13]

def load_points_csv(path):
    arr = np.loadtxt(path, delimiter=",")
    if arr.shape == (36, 3):
        return arr.astype(float)
    if arr.shape == (3, 36):
        return arr.T.astype(float)
    raise ValueError(f"{path} has shape {arr.shape}, expected 36x3 or 3x36")


def station_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def fit_slope(x, y):
    denom = np.sum(x * x)
    if abs(denom) < 1e-12:
        return 0.0
    return np.sum(x * y) / denom


def rms3d(P, Q):
    d = P - Q
    return np.sqrt(np.mean(np.sum(d * d, axis=1)))


def wrap_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def get_bottom6_indices(P_nom):
    """
    Choose the 6 fiducials with the smallest Y values.
    """
    idx = np.argsort(P_nom[:, 1])[:6]
    return np.sort(idx)


def anchor_nominal_z_to_real_bottom6(P_nom_raw, P_real_raw, anchor=0):
    """
    Shift nominal Z so the average Z of the bottom 6 nominal fiducials
    matches the average Z of the corresponding bottom 6 real fiducials.
    """
    idx_bot = get_bottom6_indices(P_nom_raw)

    z_nom_bot = np.mean(P_nom_raw[idx_bot, 2])
    z_real_bot = np.mean(P_real_raw[idx_bot, 2])
    if (anchor==0):
        dz_anchor = z_real_bot - z_nom_bot
    else:
        dz_anchor = anchor
    P_nom = P_nom_raw.copy()
    P_real = P_real_raw.copy()

    P_nom[:, 2] += dz_anchor

    return P_nom, P_real, dz_anchor, idx_bot


def reset_output_dir(outdir):
    """
    Delete and recreate the image output directory.
    """
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)


def get_station_number(station_name):
    return int(station_name[-2:])


def save_scatter_three_sets(x_nom, y_nom, x_real, y_real, x_mod, y_mod,
                            xlabel, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_nom, y_nom, s=35, label='Nominal', alpha=0.85)
    ax.scatter(x_real, y_real, s=35, label='Real', alpha=0.85)
    ax.scatter(x_mod, y_mod, s=35, label='Modded', alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    #plt.show()
    fig.savefig(outpath)
    plt.close(fig)


def residual_label(prefix, values):
    mean = np.mean(values)
    sigma = np.sqrt(np.var(values))
    return f'{prefix} (mean={mean:.2f} mm, sigma={sigma:.2f} mm)'


def save_residual_hist(resid_real, resid_mod, xlabel, title, outpath):
    vals = np.concatenate([resid_real, resid_mod])
    bins = np.linspace(-3, 3, 50)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        resid_real,
        bins=bins,
        alpha=0.6,
        label=residual_label('Real - Nominal', resid_real)
    )
    ax.hist(
        resid_mod,
        bins=bins,
        alpha=0.6,
        label=residual_label('Modded - Nominal', resid_mod)
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.ylim(0, 10)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def save_zres_vs_y(y_nom, zres_real, zres_mod, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        y_nom, zres_real, s=35,
        label=residual_label('Real - Nominal', zres_real),
        alpha=0.85
    )
    ax.scatter(
        y_nom, zres_mod, s=35,
        label=residual_label('Modded - Nominal', zres_mod),
        alpha=0.85
    )
    ax.set_xlabel('Y [mm]')
    ax.set_ylabel('Z Residuals [mm]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_zres_vs_x(y_nom, zres_real, zres_mod, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        y_nom, zres_real, s=35,
        label=residual_label('Real - Nominal', zres_real),
        alpha=0.85
    )
    ax.scatter(
        y_nom, zres_mod, s=35,
        label=residual_label('Modded - Nominal', zres_mod),
        alpha=0.85
    )
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Z Residuals [mm]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_yres_vs_x(x_nom, yres_real, yres_mod, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x_nom, yres_real, s=35,
        label=residual_label('Real - Nominal', yres_real),
        alpha=0.85
    )
    ax.scatter(
        x_nom, yres_mod, s=35,
        label=residual_label('Modded - Nominal', yres_mod),
        alpha=0.85
    )
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y Residuals [mm]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def save_xres_vs_y(y_nom, xres_real, xres_mod, title, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        y_nom, xres_real, s=35,
        label=residual_label('Real - Nominal', xres_real),
        alpha=0.85
    )
    ax.scatter(
        y_nom, xres_mod, s=35,
        label=residual_label('Modded - Nominal', xres_mod),
        alpha=0.85
    )
    ax.set_xlabel('Y [mm]')
    ax.set_ylabel('X Residuals [mm]')
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def make_station_plots(station, outdir, P_nom, P_real, P_mod,rotX):
    station_num = get_station_number(station)

    xres_real = P_real[:, 0] - P_nom[:, 0]
    yres_real = P_real[:, 1] - P_nom[:, 1]
    zres_real = P_real[:, 2] - P_nom[:, 2]

    xres_mod = P_mod[:, 0] - P_nom[:, 0]
    yres_mod = P_mod[:, 1] - P_nom[:, 1]
    zres_mod = P_mod[:, 2] - P_nom[:, 2]

    save_scatter_three_sets(
        P_nom[:, 1], P_nom[:, 2],
        P_real[:, 1], P_real[:, 2],
        P_mod[:, 1], P_mod[:, 2],
        'Y [mm]', 'Z [mm]',
        f'Station {station_num:02d}: Y vs Z Fiducials',
        os.path.join(outdir, f'YZFiducialsStation{station_num:02d}.pdf')
    )

    save_residual_hist(
        zres_real, zres_mod,
        'Z Residual [mm]',
        f'Station {station_num:02d}: Z Residuals',
        os.path.join(outdir, f'ZResidualsStation{station_num:02d}.pdf')
    )

    save_residual_hist(
        xres_real, xres_mod,
        'X Residual [mm]',
        f'Station {station_num:02d}: X Residuals',
        os.path.join(outdir, f'XResidualsStation{station_num:02d}.pdf')
    )

    save_residual_hist(
        yres_real, yres_mod,
        'Y Residual [mm]',
        f'Station {station_num:02d}: Y Residuals',
        os.path.join(outdir, f'YResidualsStation{station_num:02d}.pdf')
    )

    save_scatter_three_sets(
        P_nom[:, 0], P_nom[:, 1],
        P_real[:, 0], P_real[:, 1],
        P_mod[:, 0], P_mod[:, 1],
        'X [mm]', 'Y [mm]',
        f'Station {station_num:02d}: X vs Y Fiducials',
        os.path.join(outdir, f'XYFiducialsStation{station_num:02d}.pdf')
    )

    save_zres_vs_y(
        P_nom[:, 1], zres_real, zres_mod,
        f'Station {station_num:02d}: Z Residuals vs Y',
        os.path.join(outdir, f'ZResiduals_YStation{station_num:02d}.pdf')
    )
    y_pivot=-840
    save_zres_vs_x(
        P_nom[:, 0], zres_real - (P_nom[:, 1]-y_pivot)*rotX, zres_mod,
        f'Station {station_num:02d}: Z Residuals vs X',
        os.path.join(outdir, f'ZResiduals_XStation{station_num:02d}.pdf')
    )
    
    save_yres_vs_x(
        P_nom[:, 0], yres_real, yres_mod,
        f'Station {station_num:02d}: Y Residuals vs X',
        os.path.join(outdir, f'YResiduals_XStation{station_num:02d}.pdf')
    )

    save_xres_vs_y(
        P_nom[:, 1], xres_real, xres_mod,
        f'Station {station_num:02d}: X Residuals vs Y',
        os.path.join(outdir, f'XResiduals_YStation{station_num:02d}.pdf')
    )
    
def save_summary_scatter(df, ycol, ylabel, outpath):
    df_plot = df.sort_values('Frame station')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_plot['Frame station'], df_plot[ycol], s=50)
    for _, row in df_plot.iterrows():
        ax.annotate(
            f"{int(row['Production station']):02d}",
            (row['Frame station'], row[ycol]),
            textcoords='offset points',
            xytext=(4, 4),
            fontsize=8
        )
    ax.set_xlabel('Frame station')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Frame station')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def make_summary_plots(df, outdir):
    summary_specs = [
        ('dX_shift_mm', 'dX Shift [mm]', 'dX_shift_vs_FrameStation.pdf'),
        ('dY_shift_mm', 'dY Shift [mm]', 'dY_shift_vs_FrameStation.pdf'),
        ('rotX_mrad', 'rotX [mrad]', 'rotX_vs_FrameStation.pdf'),
        ('rotY_mrad', 'rotY [mrad]', 'rotY_vs_FrameStation.pdf'),
        ('rotZ_mrad', 'rotZ [mrad]', 'rotZ_vs_FrameStation.pdf'),
        (
            'bottom6_nominal_Z_anchor_shift_mm',
            'Bottom-6 Nominal Z Anchor Shift [mm]',
            'bottom6_nominal_Z_anchor_shift_vs_FrameStation.pdf'
        ),
    ]

    for col, ylabel, fname in summary_specs:
        save_summary_scatter(df, col, ylabel, os.path.join(outdir, fname))


def print_station_report(
    station,
    dx_shift,
    dy_shift,
    rotX_mrad,
    rotY_mrad,
    rotZ_mrad,
    mounts,
    dz_anchor,
    idx_bot,
    P_nom,
    P_real,
    P_mod,
):
    print("\n" + "=" * 150)
    print(f"STATION: {station[-2:]}")
    print("-" * 150)
    print(f"Bottom-6 nominal Z anchor shift [mm] = {dz_anchor: .4f}")
    print(f"Bottom-6 fiducial indices            = {idx_bot.tolist()}")
    print("-" * 150)
    print("Physical decomposition")
    print(f"  dX_shift_mm = {dx_shift: .4f}")
    print(f"  dY_shift_mm = {dy_shift: .4f}")
    print(f"  rotX_mrad   = {rotX_mrad: .4f}   (pivot about (0,-R,0))")
    print(f"  rotY_mrad   = {rotY_mrad: .4f}   (pivot about (0, 0,0))")
    print(f"  rotZ_mrad   = {rotZ_mrad: .4f}   (pivot about (0, 0,0))")
    print("-" * 150)
    print("Implied mount quantities [mm]")
    print(f"  XYZ_mount_dx_mm = {mounts['XYZ_mount_dx_mm']: .4f}")
    print(f"  XYZ_mount_dy_mm = {mounts['XYZ_mount_dy_mm']: .4f}")
    print(f"  XYZ_mount_dz_mm = {mounts['XYZ_mount_dz_mm']: .4f}")
    print(f"  YZ_mount_dy_mm  = {mounts['YZ_mount_dy_mm']: .4f}")
    print(f"  YZ_mount_dz_mm  = {mounts['YZ_mount_dz_mm']: .4f}")
    print("-" * 150)
    print(
        f"{'TB':>3s} | "
        f"{'Nom_X':>10s} {'Nom_Y':>10s} {'Nom_Z':>10s} | "
        f"{'Real_X':>10s} {'Real_Y':>10s} {'Real_Z':>10s} | "
        f"{'Mod_X':>10s} {'Mod_Y':>10s} {'Mod_Z':>10s}"
    )
    print("-" * 150)
    print(f"STATION: {station[-2]}")

    for i in range(len(P_nom)):
        print(
            f"{i:3d}, "
            f"{P_nom[i,0]:8.3f}, {P_nom[i,1]:8.3f}, {P_nom[i,2]:8.3f}, "
            f"{P_real[i,0]:8.3f}, {P_real[i,1]:8.3f}, {P_real[i,2]:8.3f}, "
            f"{P_real[i,0]-P_nom[i,0]:8.3f}, {P_real[i,1]-P_nom[i,1]:8.3f}, {P_real[i,2]-P_nom[i,2]:8.3f}, "
            f"{P_mod[i,0]:8.3f}, {P_mod[i,1]:8.3f}, {P_mod[i,2]:8.3f}, "
            f"{P_mod[i,0]-P_nom[i,0]:8.3f}, {P_mod[i,1]-P_nom[i,1]:8.3f}, {P_mod[i,2]-P_nom[i,2]:8.3f}"
        )
    print("=" * 132)
    print("Average Errors With Modded")
    print("%8.3f, %8.3f, %8.3f"%(np.mean(P_mod[:,0]-P_nom[:,0]), np.mean(P_mod[:,1]-P_nom[:,1]), np.mean(P_mod[:,2]-P_nom[:,2])))
    
def save_fiducial_csv(station, outdir, P_nom, P_real, P_mod):
    station_num = int(station[-2:])

    data = np.hstack([
        P_nom,   # Nom_X, Nom_Y, Nom_Z
        P_real,  # Real_X, Real_Y, Real_Z
        P_mod    # Mod_X, Mod_Y, Mod_Z
    ])

    header = (
        "Nom_X,Nom_Y,Nom_Z,"
        "Real_X,Real_Y,Real_Z,"
        "Mod_X,Mod_Y,Mod_Z"
    )

    outpath = os.path.join(outdir, f"FiducialsStation{station_num:02d}.csv")
    print(outpath)
    np.savetxt(
        outpath,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.3f"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Simple station alignment with bottom-6 Z anchor and physical decomposition"
    )
    parser.add_argument("--nominal-dir", default="nominal_csvs")
    parser.add_argument("--real-dir", default="real_csvs")
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--mount-radius-mm", type=float, default=843.0)
    parser.add_argument("--out-summary", default="summary.csv")
    parser.add_argument("--image-dir", default="ImageOutputs")
    args = parser.parse_args()

    nominal_paths = sorted(glob.glob(os.path.join(args.nominal_dir, args.glob)))
    real_paths = sorted(glob.glob(os.path.join(args.real_dir, args.glob)))

    nominal_map = {station_name_from_path(p): p for p in nominal_paths}
    real_map = {station_name_from_path(p): p for p in real_paths}

    stations = sorted(set(nominal_map) & set(real_map))
    if not stations:
        raise RuntimeError("No matching station files found")

    reset_output_dir(args.image_dir)

    summary_rows = []
    #Prod station   Frame station    dX_shift_mm    dY_shift_mm    rotX_mrad    rotY_mrad    rotZ_mrad    bottom6_nominal_Z_anchor_shift_mm
    UsedArr = np.genfromtxt("SanityChecksUltem4_27.csv", delimiter=",")
    print(UsedArr.shape)
    for st in stations:
        print(nominal_map[st], real_map[st])
        P_nom_raw = load_points_csv(nominal_map[st])
        P_real_raw = load_points_csv(real_map[st])
        StationStringFront = "Station_NPYFiles06_25_FullCalibMod/StationFidsTracker"
        StationStringBack = "_mech1.npy"
        StationID = "%i"%(int(st[-2:]))
        StationString =StationStringFront+StationID+StationStringBack
        P_station = np.load(StationString)
        P_stationAlign = P_station.copy()
        UseOld = 0
        if (UseOld!=0):
            dX_Used        =UsedArr[int(StationID), 2]
            dY_Used        =UsedArr[int(StationID), 3]
            rotX_mrad_Used =UsedArr[int(StationID), 4]/1000.
            rotY_mrad_Used =UsedArr[int(StationID), 5]/1000.
            rotZ_mrad_Used =UsedArr[int(StationID), 6]/1000.
            bottom6_Used   =UsedArr[int(StationID), 7]
            
        
        
        # 1) Anchor nominal Z to real Z using the bottom 6 fiducials
        if (UseOld==1):
            P_nom, P_real, dz_anchor, idx_bot = anchor_nominal_z_to_real_bottom6(
                P_nom_raw, P_real_raw,bottom6_Used)
        else:
            P_nom, P_real, dz_anchor, idx_bot = anchor_nominal_z_to_real_bottom6(
                P_nom_raw, P_real_raw)
        # 2) Direct X/Y translations from averages
        d = P_real - P_nom
        dX_shift = -np.mean(d[:, 0])
        dY_shift = -np.mean(d[:, 1])
        if (UseOld==1):
            dX_shift =dX_Used
            dY_shift = dY_shift

        P_work = P_real.copy()
        P_work[:, 0] += dX_shift
        P_work[:, 1] += dY_shift
        P_stationAlign[:,0]+= dX_shift
        P_stationAlign[:,1]+= dY_shift
        
        # Centered coordinates for in-plane rotation
        x0 = np.mean(P_nom[:, 0])
        y0 = np.mean(P_nom[:, 1])
        x = P_nom[:, 0] - x0
        y = P_nom[:, 1] - y0
        
        xSta =P_stationAlign[:,0] - x0
        ySta =P_stationAlign[:,1] - y0
        # Fit rotation about Z from dY vs X
        resid_y = P_work[:, 1] - P_nom[:, 1]
        rotZ = fit_slope(x, resid_y)
        if (UseOld==1):
            rotZ = rotZ_mrad_Used

        # Apply BOTH pieces of the small-angle Z rotation correction
        a = P_work[:, 0].copy()
        b = P_work[:, 1].copy()
        P_work[:, 0] = np.cos(rotZ)* a + np.sin(rotZ) * b
        P_work[:, 1] = np.cos(rotZ)* b - np.sin(rotZ) * a

        a = P_stationAlign[:, 0].copy()
        b = P_stationAlign[:, 1].copy()
        P_stationAlign[:, 0] = np.cos(rotZ)* a + np.sin(rotZ) * b     # sign may need flipping depending on convention
        P_stationAlign[:, 1] = np.cos(rotZ)* b - np.sin(rotZ) * a

        # 4) Rotation about X using bottom as pivot:
        #    z change should be zero at the bottom anchor.
        y_pivot = np.mean(P_nom[idx_bot, 1])
        y_pivot = -841
        y_rel = P_nom[:, 1] - y_pivot
        y_relSta = P_station[:, 1] - y_pivot

        # after removing rotY contribution, fit remaining Z vs Y
        resid_z_for_rotX = (P_work[:, 2] - P_nom[:, 2])# - rotY * x
        rotX = fit_slope(y_rel, resid_z_for_rotX)  # radians
        if (UseOld==1):
            rotX= rotX_mrad_Used

        # 5) Rotation about Y from dZ residual vs Y
        resid_z_after_rotX = P_work[:, 2] - P_nom[:, 2] - rotX * y_rel
        
        print(" mrad, dZ from rotX: ",  rotX*1000, (rotX * y_pivot))
        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.clf()
        plt.scatter(y, P_work[:, 2] - P_nom[:, 2], label="old")
        plt.scatter(y, resid_z_after_rotX, label="new")
        plt.legend()
        plt.show()
        '''
        
        rotY = fit_slope(x, resid_z_after_rotX)  # radians
        if (UseOld==1):
            rotY = rotY_mrad_Used

        resid_z_after_rotXY = P_work[:, 2] - P_nom[:, 2] - rotX * y_rel - rotY * x


        '''
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.clf()
        #plt.scatter(x, P_work[:, 2] - P_nom[:, 2])
        plt.scatter(x, resid_z_after_rotX)
        plt.scatter(x, resid_z_after_rotXY)
        plt.show()
        '''
        # 6) Apply final correction to build modded points
        P_mod = P_work.copy()
        P_stationMod = P_stationAlign.copy()
        
        # rotY about origin: dz = -rotY * x
        P_mod[:, 2]        += -rotY * x
        P_stationMod[:, 2] += -rotY * xSta
        
        # rotX about bottom pivot: dz = -rotX * (y - y_pivot)
        P_mod[:, 2]        += -rotX * y_rel
        P_stationMod[:, 2] += -rotX * y_relSta

        # 7) Convert to implied mount quantities
        R = args.mount_radius_mm
        XYZ_mount_dx_mm = dX_shift
        XYZ_mount_dy_mm = dY_shift - rotZ * R
        YZ_mount_dy_mm  = dY_shift + rotZ * R

        # At y=0 on the top-side mounts, rotX gives dz = +rotX * R
        # rotY gives opposite dz at x=+/-R
        XYZ_mount_dz_mm = (rotX * y_pivot) - (rotY * R)
        YZ_mount_dz_mm  = (rotX * y_pivot) + (rotY * R)
        #print("XYZ dZ from rotY: ", - (rotY * R))
        print(" mrad, dZ from rotX: ",  rotX*1000, (rotX * y_pivot))

        print("---------------------")
        print("%10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f"%(\
        P_stationMod[0,0], P_stationMod[0,1], P_stationMod[0,2],\
        P_station[0,0], P_station[0,1], P_station[0,2],\
        P_stationMod[0,0]- P_station[0,0], P_stationMod[0,1]- P_station[0,1],P_stationMod[0,2]- P_station[0,2]))

        print("%10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f"%(\
        P_stationMod[1,0], P_stationMod[1,1], P_stationMod[1,2],\
        P_station[1,0], P_station[1,1], P_station[1,2],\
        P_stationMod[1,0]- P_station[1,0], P_stationMod[1,1]- P_station[1,1],P_stationMod[1,2]- P_station[1,2]))

        print("%10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f, %10.3f"%(\
        P_stationMod[2,0], P_stationMod[2,1], P_stationMod[2,2],\
        P_station[2,0], P_station[2,1], P_station[2,2],\
        P_stationMod[2,0]- P_station[2,0], P_stationMod[2,1]- P_station[2,1],P_stationMod[2,2]- P_station[2,2]))
        
        P_stationModNew = np.zeros((3,3))
        if (FrameOrdering[int(st[-2:])] % 2==0):
            P_stationModNew[0,:]=P_stationMod[1,:]
            P_stationModNew[1,:]=P_stationMod[2,:]
            P_stationModNew[2,:]=P_stationMod[0,:]

        else:
            P_stationModNew[0,:]=P_stationMod[0,:]
            P_stationModNew[1,:]=P_stationMod[2,:]
            P_stationModNew[2,:]=P_stationMod[1,:]

        with open("StationFidsWithUltem.csv", "a") as f:
            np.savetxt(f, P_stationModNew.reshape(1, -1), delimiter=", ", fmt="%10.3f")
        mounts = {
            "XYZ_mount_dx_mm": XYZ_mount_dx_mm,
            "XYZ_mount_dy_mm": XYZ_mount_dy_mm,
            "XYZ_mount_dz_mm": XYZ_mount_dz_mm,
            "YZ_mount_dy_mm":  YZ_mount_dy_mm,
            "YZ_mount_dz_mm":  YZ_mount_dz_mm,
        }
        real_rms = rms3d(P_real, P_nom)
        mod_rms = rms3d(P_mod, P_nom)

        # Print using shifted nominal
        P_nom_print = P_nom.copy()
        P_real_print = P_real.copy()
        P_mod_print = P_mod.copy()
        '''
        print_station_report(
            st,
            dX_shift,
            dY_shift,
            rotX * 1000.0,
            rotY * 1000.0,
            rotZ * 1000.0,
            mounts,
            dz_anchor,
            idx_bot,
            P_nom_print,
            P_real_print,
            P_mod_print,
        )
        '''
        make_station_plots(st, args.image_dir, P_nom_print, P_real_print, P_mod_print, rotX)
        save_fiducial_csv(st, args.image_dir, P_nom_print, P_real_print, P_mod_print)

        #print(f"\nDiagnostics")
        #print(f"  real_rms_mm = {real_rms: .4f}")
        #print(f"  mod_rms_mm  = {mod_rms: .4f}")

        summary_rows.append({
            "Production station":  int(st[-2:]),
            "Frame station":  FrameOrdering[int(st[-2:])],
            "dX_shift_mm": dX_shift,
            "dY_shift_mm": dY_shift,
            "rotX_mrad": rotX * 1000.0,
            "rotY_mrad": rotY * 1000.0,
            "rotZ_mrad": rotZ * 1000.0,
            "bottom6_nominal_Z_anchor_shift_mm": dz_anchor,
            "real_rms_mm": real_rms,
            "mod_rms_mm": mod_rms,
            "XYZ_mount_dx_mm": XYZ_mount_dx_mm,
            "XYZ_mount_dy_mm": XYZ_mount_dy_mm,
            "XYZ_mount_dz_mm": XYZ_mount_dz_mm,
            "YZ_mount_dy_mm": YZ_mount_dy_mm,
            "YZ_mount_dz_mm": YZ_mount_dz_mm,
        })

    df = pd.DataFrame(summary_rows)

    print("\n\n===== SUMMARY TABLE =====\n")
    print(df[
        [
            "Production station",
            "Frame station",
            "dX_shift_mm",
            "dY_shift_mm",
            "rotX_mrad",
            "rotY_mrad",
            "rotZ_mrad",
            "bottom6_nominal_Z_anchor_shift_mm",
            "real_rms_mm",
            "mod_rms_mm",
            "XYZ_mount_dx_mm",
            "XYZ_mount_dy_mm",
            "XYZ_mount_dz_mm",
            "YZ_mount_dy_mm",
            "YZ_mount_dz_mm",
        ]
    ].to_string(index=False))

    df.to_csv(args.out_summary, index=False)
    make_summary_plots(df, args.image_dir)
    print(f"\nSaved: {args.out_summary}")
    print(f"Saved plots in: {args.image_dir}")


if __name__ == "__main__":
    main()

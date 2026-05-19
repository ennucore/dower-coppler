import json, time
from pathlib import Path
import h5py
import numpy as np
import torch

H5=Path("/mnt/pocampus/lev/ultratrace_Head_monster_2025-09-21_21-32-01_y-20to20mm_30elev.h5")
OUT=Path("/tmp/sep21_cached_compound_lag1_color_y14_15_acq200_399.npz")
START=200
STOP=399
YIDX=np.array([14,15], dtype=np.int64)
LOW=0.08
SOUND=1600.0

def svd_filter_fast(compound_images, low_cutoff=0.08, high_cutoff=1.0):
    X = compound_images.reshape(compound_images.shape[0], -1)
    C = X @ X.conj().T
    evals, U = torch.linalg.eigh(C)
    U = U[:, torch.argsort(evals, descending=True)]
    n = C.shape[0]
    a = round(low_cutoff * n)
    b = round((1 - high_cutoff) * n)
    Uc = U[:, a:n-b]
    return (Uc @ Uc.conj().T @ X).reshape_as(compound_images)

def meta(h5, idx):
    m=h5[f"acquisitions/{idx}/meta"]
    cfg=json.loads(m["acquisition_config"][()].decode() if isinstance(m["acquisition_config"][()], bytes) else m["acquisition_config"][()])
    rt=json.loads(m["runtime_metadata"][()].decode() if isinstance(m["runtime_metadata"][()], bytes) else m["runtime_metadata"][()])
    fr=float(rt.get("empirical_pulse_repetition_rate_hz") or (cfg["requested_prf_hz"]/cfg["num_angles"]))
    return fr, float(cfg["tx_freq_hz"])

t0=time.time()
with h5py.File(H5,"r") as h5:
    grid=h5[f"acquisitions/{START}/meta/grid"]
    x_mm=np.unique(np.asarray(grid["x"], dtype=np.float64))*1000
    y_mm=np.unique(np.asarray(grid["y"], dtype=np.float64))*1000
    z_mm=np.unique(np.asarray(grid["z"], dtype=np.float64))*1000
    fr, tx = meta(h5, START)
    scale = fr * SOUND / (4.0*np.pi*tx) * 2.0 # equivalent angle(r1)*fr/(2pi)*SOUND/(2tx)
    scale = fr / (2.0*np.pi) * SOUND / (2.0*tx)
    acc=None
    count=0
    for idx in range(START, STOP+1):
        comp=np.asarray(h5[f"acquisitions/{idx}/meta/compound_image"][:, YIDX, :, :], dtype=np.complex64)
        sig=torch.from_numpy(comp)
        sig=sig-sig.mean(dim=0, keepdim=True)
        filt=svd_filter_fast(sig, low_cutoff=LOW, high_cutoff=1.0)
        r1=(filt[1:]*torch.conj(filt[:-1])).mean(dim=0)
        color=(torch.angle(r1)*scale).cpu().numpy().astype(np.float32)
        if acc is None:
            acc=np.zeros_like(color, dtype=np.float64)
        acc += color
        count += 1
        if count % 10 == 0:
            print(f"processed {count}/{STOP-START+1}; elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    out=(acc/count).astype(np.float32)
    np.savez_compressed(OUT, color_doppler=out, x_mm=x_mm.astype(np.float32), y_mm=y_mm[YIDX].astype(np.float32), z_mm=z_mm.astype(np.float32), frame_rate_hz=np.float32(fr), tx_freq_hz=np.float32(tx), first_acq=np.int32(START), last_acq=np.int32(STOP), source_h5=str(H5), note="Lag-1 Kasai color from cached beamformed compound_image, y idx 14-15, acqs 200-399")
    print({"out":str(OUT),"shape":out.shape,"elapsed_min":(time.time()-t0)/60}, flush=True)

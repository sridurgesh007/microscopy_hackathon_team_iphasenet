import os, glob, uuid, traceback, math, json
import numpy as np
import matplotlib.pyplot as plt

import ase.io
import abtem

from typing import Optional, List

# =========================
# GLOBAL DEFAULTS (LOCKED)
# =========================
VOLTAGE = 200e3

RES = (256, 256)
SAMPLING = 0.10
FOV_ANG = RES[0] * SAMPLING     # 25.6 Å

SLICE_THICKNESS = 3.0

INNER_MRAD = 10
OUTER_MRAD = 50

THICKNESS_LIST_NM = [20, 40, 60, 80, 100, 120]

ALLOW_TILT_DEFAULT = True
RX_STD_DEG = 0.30
RY_STD_DEG = 0.30

XY_MARGIN_ANG = 2.0
MAX_XY_REP_BULK = 4
MAX_XY_REP_2D = 10
Z_TARGET_ANG = 2200.0

VACUUM_Z_ANG = 20.0

# =========================
# UTILS
# =========================
def set_device(use_gpu: bool):
    """
    abTEM device is process-global.
    For multi-GPU: run 1 process per GPU with CUDA_VISIBLE_DEVICES set externally.
    """
    if not use_gpu:
        print("🟡 Using CPU (add --gpu for GPU).")
        return "cpu"
    try:
        abtem.config.set({"device": "gpu", "fft": "cufft"})
        print("🟢 GPU enabled (cufft).")
        return "gpu"
    except Exception as e:
        print(f"🟡 GPU not available, using CPU. Reason: {e}")
        return "cpu"

def list_cube_files(source_dir):
    return sorted(glob.glob(os.path.join(source_dir, "*.cube")))

def ensure_hw4(arr):
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[2] == 1 and arr.shape[3] == 4:
        arr = arr[:, :, 0, :]
    if not (arr.ndim == 3 and arr.shape[2] == 4):
        raise ValueError(f"Unexpected measurement shape: {arr.shape}")
    return arr

def enforce_cell_xy(atoms, fov_ang, z_ang=None):
    atoms = atoms.copy()
    if z_ang is None:
        z_ang = float(atoms.cell[2, 2])
    atoms.set_cell([[fov_ang, 0, 0],
                    [0, fov_ang, 0],
                    [0, 0, z_ang]], scale_atoms=False)
    return atoms

def recenter_xy_to_fov(atoms, fov_ang):
    atoms = atoms.copy()
    xy = atoms.positions[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    c = 0.5 * (mn + mx)
    target = np.array([fov_ang / 2, fov_ang / 2])
    shift = target - c
    atoms.positions[:, 0] += shift[0]
    atoms.positions[:, 1] += shift[1]
    return atoms

def bbox_xy(atoms):
    xy = atoms.positions[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    return mn, mx, (mx - mn)

def fits_xy_with_margin(atoms, fov_ang, margin):
    mn, mx, size = bbox_xy(atoms)
    ok = (mn[0] >= margin and mn[1] >= margin and
          mx[0] <= (fov_ang - margin) and mx[1] <= (fov_ang - margin))
    return ok, mn, mx, size

def choose_xy_repeat_that_fits(base_atoms, fov_ang, margin, max_rep):
    """
    Choose largest n×n tiling that fits within [margin, fov-margin].
    PVSK fix: often picks 3×3 instead of 4×4.
    """
    best = None
    for n in range(max_rep, 0, -1):
        a = base_atoms * (n, n, 1)
        a = enforce_cell_xy(a, fov_ang, float(a.cell[2, 2]))
        a = recenter_xy_to_fov(a, fov_ang)
        ok, mn, mx, size = fits_xy_with_margin(a, fov_ang, margin)
        if ok:
            return n, a, (mn, mx, size)
        if best is None:
            best = (n, a, (mn, mx, size))
    return best

def slice_z_to_thickness(atoms, thickness_ang, slice_thickness, fov_ang):
    z = atoms.positions[:, 2]
    z0 = float(z.min())
    margin = 2.0 * slice_thickness
    keep = (z - z0) <= (thickness_ang + margin)
    slab = atoms[keep].copy()
    if len(slab) == 0:
        return None
    zcell = thickness_ang + margin + 10.0
    slab = enforce_cell_xy(slab, fov_ang, zcell)
    slab.center(vacuum=5.0, axis=2)
    return slab

def normalize_range(x, lo=1, hi=99):
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or (b - a) < 1e-12:
        a = float(np.min(x))
        b = float(np.max(x))
        if (b - a) < 1e-12:
            b = a + 1.0
    return a, b

def plot_preview(material, meta, seg, pot, out_png):
    s0, s1, s2, s3 = [seg[..., i] for i in range(4)]
    ssum = s0 + s1 + s2 + s3

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{material} | t={meta['thickness_nm']:.0f}nm | "
        f"rz={meta['rz']:.1f} rx={meta['rx']:.2f} ry={meta['ry']:.2f} | "
        f"XY={meta['xy_rep']}×{meta['xy_rep']} | mode={meta['mode']}",
        fontsize=13
    )

    vmin_s, vmax_s = normalize_range(seg, 1, 99)
    vmin_sum, vmax_sum = normalize_range(ssum, 1, 99)
    vmin_p, vmax_p = normalize_range(pot, 1, 99)

    panels = [
        (ax[0,0], s0, "Seg0"),
        (ax[0,1], s1, "Seg1"),
        (ax[0,2], ssum, "Sum (ADF)"),
        (ax[1,0], s2, "Seg2"),
        (ax[1,1], s3, "Seg3"),
        (ax[1,2], pot, "Potential"),
    ]

    for a, img, title in panels:
        cmap = "magma" if title == "Potential" else "gray"
        if title == "Potential":
            im = a.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin_p, vmax=vmax_p)
        elif "Sum" in title:
            im = a.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin_sum, vmax=vmax_sum)
        else:
            im = a.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin_s, vmax=vmax_s)
        a.set_title(title)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_png, dpi=170)
    plt.close()

def atomic_save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = f"{path}.tmp_{uuid.uuid4().hex}.npz"  # ensure it ENDS with .npz
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)    # file-handle => no auto ".npz" surprises
        os.replace(tmp, path)                  # atomic rename on same filesystem
    finally:
        # cleanup if something failed mid-way
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

# =========================
# CORE SIMULATION (with caching)
# =========================
def simulate_sample(atoms, device, probe, detector):
    potential = abtem.Potential(
        atoms,
        sampling=SAMPLING,
        slice_thickness=SLICE_THICKNESS,
        parametrization="kirkland",
        device=device
    )
    probe.grid.match(potential)

    scan = abtem.GridScan(
        start=[0, 0],
        end=[FOV_ANG, FOV_ANG],
        gpts=RES,
        potential=potential
    )

    meas = probe.scan(potential, scan=scan, detectors=detector)
    proj = potential.project()

    seg = meas.compute().array
    pot = proj.compute().array

    if hasattr(seg, "get"): seg = seg.get()
    if hasattr(pot, "get"): pot = pot.get()

    seg = ensure_hw4(seg)
    pot = np.asarray(pot)
    if pot.ndim == 3 and pot.shape[0] == 1:
        pot = pot[0]
    if pot.shape != RES:
        raise ValueError(f"Potential projection shape mismatch: {pot.shape} != {RES}")

    return seg, pot

# =========================
# BUILD ATOMS
# =========================
def build_atoms_bulk(base_atoms, thickness_nm, rz, rx, ry):
    atoms = base_atoms.copy()

    z0 = float(atoms.cell[2, 2])
    reps_z = int(np.ceil(Z_TARGET_ANG / max(z0, 1e-6)))
    column = atoms * (1, 1, max(reps_z, 1))
    column.center(vacuum=10.0, axis=2)

    column.rotate(rz, "z", rotate_cell=False)
    column = recenter_xy_to_fov(column, FOV_ANG)

    slab = slice_z_to_thickness(column, thickness_nm * 10.0, SLICE_THICKNESS, FOV_ANG)
    if slab is None:
        return None

    slab.rotate(rx, "x", rotate_cell=False)
    slab.rotate(ry, "y", rotate_cell=False)
    slab = recenter_xy_to_fov(slab, FOV_ANG)
    slab = enforce_cell_xy(slab, FOV_ANG, float(slab.cell[2, 2]))
    return slab

def build_atoms_mos2(base_atoms, rz, rx, ry):
    atoms = base_atoms.copy()
    atoms.center(vacuum=VACUUM_Z_ANG, axis=2)

    atoms.rotate(rz, "z", rotate_cell=False)
    atoms.rotate(rx, "x", rotate_cell=False)
    atoms.rotate(ry, "y", rotate_cell=False)

    atoms = recenter_xy_to_fov(atoms, FOV_ANG)
    atoms = enforce_cell_xy(atoms, FOV_ANG, float(atoms.cell[2, 2]))
    atoms.center(vacuum=VACUUM_Z_ANG, axis=2)
    return atoms

# =========================
# SAMPLE PLANNING (EQUAL)
# =========================
def pick_thicknesses(thickness_nm: Optional[float], is_mos2: bool):
    if thickness_nm is not None:
        return [float(thickness_nm)]
    return [float(t) for t in THICKNESS_LIST_NM]

def build_jobs(material_files: List[str], total_samples: int, thickness_nm: Optional[float], seed: int):
    rng = np.random.default_rng(seed if seed != 0 else None)

    n_mat = len(material_files)
    if n_mat == 0:
        return []

    per_mat = total_samples // n_mat
    remainder = total_samples % n_mat

    jobs = []
    for idx, fp in enumerate(material_files):
        material = os.path.splitext(os.path.basename(fp))[0]
        is_mos2 = ("mos2" in material.lower())

        n_samples = per_mat + (1 if idx < remainder else 0)
        thicknesses = pick_thicknesses(thickness_nm, is_mos2)

        for _ in range(n_samples):
            t_nm = float(rng.choice(thicknesses))
            rz = float(rng.uniform(0, 360))
            if ALLOW_TILT_DEFAULT:
                scale = 0.5 if is_mos2 else 1.0
                rx = float(rng.normal(0, RX_STD_DEG * scale))
                ry = float(rng.normal(0, RY_STD_DEG * scale))
            else:
                rx = 0.0
                ry = 0.0
            jobs.append((fp, material, is_mos2, t_nm, rz, rx, ry))

    rng.shuffle(jobs)
    return jobs

# =========================
# WORKER SHARDING
# =========================
def shard_jobs(jobs, worker_id: int, num_workers: int):
    if num_workers <= 1:
        return jobs
    return [j for i, j in enumerate(jobs) if (i % num_workers) == worker_id]

# =========================
# WORKER (1 process)
# =========================
def run_worker(jobs, out_npz_dir, out_prev_dir, device, save_png_every):
    saved = 0
    png_countdown = 0

    cache = {}  # fp -> (base_xy, xy_rep)

    #  cache these for speed
    detector = abtem.SegmentedDetector(
        nbins_radial=1, nbins_azimuthal=4,
        inner=INNER_MRAD, outer=OUTER_MRAD
    )
    probe = abtem.Probe(
        energy=VOLTAGE,
        semiangle_cutoff=20.0,
        defocus="scherzer",
        device=device
    )

    for (fp, material, is_mos2, t_nm, rz, rx, ry) in jobs:
        try:
            if fp not in cache:
                base0 = ase.io.read(fp)
                max_rep = MAX_XY_REP_2D if is_mos2 else MAX_XY_REP_BULK
                xy_rep, base_xy, _dbg = choose_xy_repeat_that_fits(base0, FOV_ANG, XY_MARGIN_ANG, max_rep=max_rep)
                base_xy = enforce_cell_xy(base_xy, FOV_ANG, float(base_xy.cell[2, 2]))
                base_xy = recenter_xy_to_fov(base_xy, FOV_ANG)
                cache[fp] = (base_xy, int(xy_rep))

            base_xy, xy_rep = cache[fp]

            atoms = build_atoms_mos2(base_xy, rz, rx, ry) if is_mos2 else build_atoms_bulk(base_xy, t_nm, rz, rx, ry)
            if atoms is None:
                continue

            seg, pot = simulate_sample(atoms, device, probe, detector)

            uid = uuid.uuid4().hex[:10]
            out_path = os.path.join(out_npz_dir, f"{material}_t{int(round(t_nm)):03d}_{uid}.npz")

            inp = np.moveaxis(seg, -1, 0).astype(np.float16)   # (4,256,256)
            ssum = np.sum(seg, axis=-1).astype(np.float32)     # (256,256)
            pot = pot.astype(np.float32)

            meta = dict(
                material=material,
                mode=("2D" if is_mos2 else "bulk"),
                thickness_nm=np.float32(t_nm),
                fov_angstrom=np.float32(FOV_ANG),
                sampling=np.float32(SAMPLING),
                res=np.array(RES, dtype=np.int32),
                detector_inner_mrad=np.float32(INNER_MRAD),
                detector_outer_mrad=np.float32(OUTER_MRAD),
                rotation=np.array([rz, rx, ry], dtype=np.float32),
                xy_rep=np.int32(xy_rep),
                xy_margin_ang=np.float32(XY_MARGIN_ANG),
            )

            #  safe for multi-worker same folder
            atomic_save_npz(out_path, input=inp, sum=ssum, potential=pot, **meta)

            if save_png_every > 0:
                if png_countdown <= 0:
                    out_png = os.path.join(out_prev_dir, f"{material}_t{int(round(t_nm)):03d}_{uid}.png")
                    plot_preview(
                        material,
                        dict(thickness_nm=t_nm, rz=rz, rx=rx, ry=ry, xy_rep=xy_rep, mode=meta["mode"]),
                        seg, pot, out_png
                    )
                    png_countdown = save_png_every
                png_countdown -= 1

            saved += 1
            if saved % 50 == 0:
                print(f"✅ Worker saved {saved} samples...")

        except Exception as e:
            print(f"❌ Failed: {material} t={t_nm}: {e}")
            traceback.print_exc()

    return saved

# =========================
# MAIN
# =========================
def main():
    import argparse
    p = argparse.ArgumentParser("Dataset generator (supports multi-worker sharding)")
    p.add_argument("--source_dir", type=str, default="source_cubes")
    p.add_argument("--out_dir", type=str, default="dataset_outputs_mod")
    p.add_argument("--gpu", action="store_true")
    
    p.add_argument("--write_jobs", type=str, default="",
                   help="If set, only write jobs.jsonl to this path and exit.")
    p.add_argument("--jobs", type=str, default="",
                   help="If set, load jobs from this jobs.jsonl instead of planning.")
    p.add_argument("--only", type=str, default="", help="substring match, e.g. --only PVSK")

    p.add_argument("--total_samples", type=int, default=20000)
    p.add_argument("--thickness_nm", type=float, default=None)

    p.add_argument("--save_png_every", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)

    #  worker controls
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker_id", type=int, default=0)
    p.add_argument("--separate_worker_dirs", action="store_true",
                   help="If set: save into npz/worker_{id} and previews/worker_{id} (recommended).")

    args = p.parse_args()

    device = set_device(args.gpu)

    files = list_cube_files(args.source_dir)
    if args.only.strip():
        key = args.only.strip().lower()
        files = [f for f in files if key in os.path.basename(f).lower()]

    print(f"📂 Found {len(files)} cube files (after filter).")
    if not files:
        print("❌ No cube files found. Check --source_dir / --only.")
        return

    # Build equal plan
    jobs = build_jobs(files, total_samples=args.total_samples, thickness_nm=args.thickness_nm, seed=args.seed)
    print(f" Planned jobs: {len(jobs)} total samples")

    #  shard for this worker
    jobs = shard_jobs(jobs, worker_id=args.worker_id, num_workers=args.workers)
    print(f" Worker {args.worker_id}/{args.workers}: {len(jobs)} jobs")

    # folders
    if args.separate_worker_dirs:
        out_npz_dir = os.path.join(args.out_dir, "npz", f"worker_{args.worker_id}")
        out_prev_dir = os.path.join(args.out_dir, "previews", f"worker_{args.worker_id}")
    else:
        out_npz_dir = os.path.join(args.out_dir, "npz")
        out_prev_dir = os.path.join(args.out_dir, "previews")

    os.makedirs(out_npz_dir, exist_ok=True)
    os.makedirs(out_prev_dir, exist_ok=True)

    saved = run_worker(jobs, out_npz_dir, out_prev_dir, device, args.save_png_every)

    print(f"\n🏁 Done. Worker {args.worker_id} saved {saved}/{len(jobs)} samples.")
    print(f"📁 NPZ: {out_npz_dir}")
    print(f"🖼️  Previews: {out_prev_dir}")

if __name__ == "__main__":
    main()
"""Synthetic microscope movies of Brownian particles with a known diffusion coefficient.

Use this to test your tracking pipeline: if you cannot recover the D you put in,
you cannot trust the D you get from the real experiment.

    python synthetic_brownian.py            # make a movie, track it, compare with the true D
    python synthetic_brownian.py --save out # also save the frames as PNG files in folder out/

In your own code:

    from synthetic_brownian import make_movie
    frames, truth = make_movie(D=0.9, n_frames=300)

Default values mimic the FYS3715 lab: 0.51 um beads, 40X objective
(0.1837 um/pixel), 15 frames per second.
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_movie(D=0.9, n_particles=60, n_frames=300, shape=(512, 512),
               pixel_size=0.1837, fps=15.0, spot_sigma=2.0, amplitude=120.0,
               background=60.0, noise=8.0, drift=(0.0, 0.0), dark_particles=False,
               seed=None):
    """Return (frames, truth) for a movie of 2D Brownian particles.

    D            diffusion coefficient in um^2/s
    pixel_size   um per pixel
    fps          frames per second
    spot_sigma   width of the Gaussian image of a particle, in pixels
    amplitude    peak brightness of a particle (grey levels)
    background   background grey level
    noise        standard deviation of the camera noise (grey levels)
    drift        constant drift (vx, vy) in um/s, e.g. from flow in the sample
    dark_particles  True gives dark particles on a bright background (bright field),
                 False gives bright particles on a dark background (dark field)

    frames is a uint8 array with shape (n_frames, ny, nx).
    truth is a DataFrame with columns frame, particle, x, y (in pixels).
    Particles that leave the field of view re-enter on the opposite side.
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    dt = 1.0 / fps
    step = np.sqrt(2 * D * dt) / pixel_size  # standard deviation per frame and axis, in pixels
    drift_px = np.asarray(drift) * dt / pixel_size

    pos = np.empty((n_frames, n_particles, 2))
    pos[0] = rng.uniform([0, 0], [nx, ny], size=(n_particles, 2))
    for t in range(1, n_frames):
        pos[t] = pos[t - 1] + rng.normal(0, step, size=(n_particles, 2)) + drift_px
    pos_wrapped = np.mod(pos, [nx, ny])

    yy, xx = np.mgrid[0:ny, 0:nx]
    frames = np.empty((n_frames, ny, nx), dtype=np.uint8)
    r = int(np.ceil(4 * spot_sigma))
    for t in range(n_frames):
        img = np.full((ny, nx), 0.0)
        for x, y in pos_wrapped[t]:
            x0, x1 = max(int(x) - r, 0), min(int(x) + r + 1, nx)
            y0, y1 = max(int(y) - r, 0), min(int(y) + r + 1, ny)
            img[y0:y1, x0:x1] += amplitude * np.exp(
                -((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) / (2 * spot_sigma ** 2))
        img = background - img if dark_particles else background + img
        img += rng.normal(0, noise, size=img.shape)
        frames[t] = np.clip(img, 0, 255).astype(np.uint8)

    truth = pd.DataFrame({
        'frame': np.repeat(np.arange(n_frames), n_particles),
        'particle': np.tile(np.arange(n_particles), n_frames),
        'x': pos[:, :, 0].ravel(),
        'y': pos[:, :, 1].ravel(),
    })
    return frames, truth


def save_png(frames, folder):
    """Save frames as folder/frame0000.png, frame0001.png, ..."""
    import imageio.v2 as imageio  # installed with pims
    os.makedirs(folder, exist_ok=True)
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(folder, f'frame{i:04d}.png'), f)


def track_and_fit(frames, pixel_size, fps, diameter=9, minmass=300, search_range=10, max_lag=30):
    """A minimal trackpy pipeline: locate, link, filter, subtract drift, fit MSD.

    Returns the fitted D in um^2/s, the MSD offset in um^2 and the ensemble MSD.
    This is a reference to compare your own pipeline with, not a recipe for the lab data.
    The defaults suit the default movie. Without minmass, noise is detected as particles;
    with search_range much below 5 times the step per frame, long steps are lost and D comes out too small.
    """
    import trackpy as tp
    tp.quiet()
    f = tp.batch(frames, diameter, minmass=minmass, processes=1)
    t = tp.link(f, search_range=search_range, memory=0)
    t = tp.filter_stubs(t, 25)
    t = tp.subtract_drift(t, tp.compute_drift(t))
    msd = tp.emsd(t, pixel_size, fps, max_lagtime=max_lag)
    lag = msd.index.values
    slope, intercept = np.polyfit(lag, msd.values, 1)  # MSD = 4 D t + offset (2D)
    return slope / 4, intercept, msd


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--D', type=float, default=0.9, help='true diffusion coefficient, um^2/s')
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save', metavar='FOLDER', help='save frames as PNG files here')
    a = p.parse_args()

    frames, truth = make_movie(D=a.D, n_frames=a.frames, seed=a.seed)
    if a.save:
        save_png(frames, a.save)
    D_fit, offset, msd = track_and_fit(frames, pixel_size=0.1837, fps=15.0)
    print(f'true D = {a.D:.3f} um^2/s, fitted D = {D_fit:.3f} um^2/s, '
          f'MSD offset = {offset:.4f} um^2 (localisation error)')

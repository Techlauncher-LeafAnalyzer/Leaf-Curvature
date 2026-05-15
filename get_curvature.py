import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_binary_leaf_mask(mask):
    """
    Convert input mask to a clean binary mask:
    background = 0, leaf = 255
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 127).astype(np.uint8) * 255

    # Heuristic: if white occupies > half the image, invert
    # (because usually the leaf is smaller than the background)
    if np.count_nonzero(mask) > mask.size // 2:
        mask = 255 - mask

    return mask


def morphological_close(mask, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_largest_contour(mask):
    """
    Return largest external contour as ordered Nx2 points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contour found in the mask.")

    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float64)

    if len(points) < 5:
        raise ValueError("Contour has too few points.")

    return points


def moving_average_closed_contour(points, window_size=5):
    """
    Moving average smoothing on a closed contour.
    """
    n = len(points)
    half = window_size // 2
    smoothed = []

    for i in range(n):
        idx = [(i + j) % n for j in range(-half, half + 1)]
        smoothed.append(points[idx].mean(axis=0))

    return np.array(smoothed, dtype=np.float64)


def compute_curvature_from_5_points(points5):
    """
    Compute curvature at the middle point of 5 points
    using a 4th-degree polynomial fit.

    For numerical stability:
    - if x-range is larger, fit y = f(x)
    - otherwise, fit x = f(y)
    """
    points5 = np.asarray(points5, dtype=np.float64)
    x = points5[:, 0]
    y = points5[:, 1]

    # Choose the more stable fitting direction
    if np.ptp(x) >= np.ptp(y):
        independent = x
        dependent = y
        t0 = x[2]
    else:
        independent = y
        dependent = x
        t0 = y[2]

    # Need distinct independent values for fitting
    if len(np.unique(independent)) < 5:
        return np.nan

    coeffs = np.polyfit(independent, dependent, deg=4)
    d1 = np.polyder(coeffs, 1)
    d2 = np.polyder(coeffs, 2)

    f1 = np.polyval(d1, t0)
    f2 = np.polyval(d2, t0)

    curvature = abs(f2) / ((1.0 + f1 ** 2) ** 1.5)
    return curvature


def curvature_series(points, fit_window=5, step=1):
    """
    Compute curvature along the contour using a 5-point local window.

    step=1  -> dense sampling
    step=5  -> literally every 5 points
    """
    n = len(points)
    half = fit_window // 2

    values = []
    indices = []

    for i in range(0, n, step):
        idx = [(i + j) % n for j in range(-half, half + 1)]
        pts5 = points[idx]
        kappa = compute_curvature_from_5_points(pts5)

        if np.isfinite(kappa):
            values.append(kappa)
            indices.append(i)

    return np.array(indices), np.array(values, dtype=np.float64)


def remove_outliers_sigma(values, sigma=2.0):
    """
    Remove values outside mean ± sigma * std.
    Returns filtered values and keep-mask.
    """
    if len(values) == 0:
        return values, np.array([], dtype=bool)

    mu = np.mean(values)
    std = np.std(values)

    if std < 1e-12:
        keep = np.ones(len(values), dtype=bool)
    else:
        lower = mu - sigma * std
        upper = mu + sigma * std
        keep = (values >= lower) & (values <= upper)

    return values[keep], keep


def plot_curvature_stages(
    leaf_mask,
    raw_curv,
    raw_filtered,
    smooth_curv,
    smooth_filtered,
    curvature_score=None,   # <-- add this
    save_path=None,
    show=True
):
    """
    Create a figure similar to the paper:
    leaf mask on the left + four plots.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 2.5, 2.5])

    ax_leaf = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # Left panel
    ax_leaf.imshow(leaf_mask, cmap="gray")
    ax_leaf.set_title("Sample leaf mask", fontsize=14, pad=10)
    ax_leaf.axis("off")

    # Add score below the image
    if curvature_score is not None:
        ax_leaf.text(
            0.5, -0.08,
            f"Curvature score: {curvature_score:.6f}",
            transform=ax_leaf.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )

    # (a) Before outlier removal
    ax1.plot(np.arange(len(raw_curv)), raw_curv, linewidth=1)
    ax1.set_xlabel("Point index")
    ax1.set_ylabel("Curvature")
    ax1.set_title("Curvature values at 5-point intervals\nbefore outlier removal", fontsize=12)

    # (b) After outlier removal
    ax2.plot(np.arange(len(raw_filtered)), raw_filtered, linewidth=1)
    ax2.set_xlabel("Point index")
    ax2.set_ylabel("Curvature")
    ax2.set_title("Curvature values at 5-point intervals\nafter outlier removal", fontsize=12)

    # (c) After moving average
    ax3.plot(np.arange(len(smooth_curv)), smooth_curv, linewidth=1)
    ax3.set_xlabel("Point index")
    ax3.set_ylabel("Curvature")
    ax3.set_title("Curvature values at 5-point intervals\nafter taking moving average", fontsize=12)

    # (d) After moving average and outlier removal
    ax4.plot(np.arange(len(smooth_filtered)), smooth_filtered, linewidth=1)
    ax4.set_xlabel("Point index")
    ax4.set_ylabel("Curvature")
    ax4.set_title("Curvature values at 5-point intervals\nafter taking moving average and removing outliers", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def extract_leaf_curvature_feature(
    binary_mask,
    close_kernel_size=3,
    moving_avg_window=5,
    fit_window=5,
    sigma=2.0,
    step=1,
    visualize=False,
    show_image=False,
    save_plot_path=None
):
    """
    Main runtime function.

    Returns:
        feature_value
        debug_info
    """
    # 1. Prepare mask
    mask = ensure_binary_leaf_mask(binary_mask)

    # 2. Morphological closing
    mask_closed = morphological_close(mask, kernel_size=close_kernel_size)

    # 3. Boundary extraction
    contour = extract_largest_contour(mask_closed)

    # 4. Raw curvature
    _, raw_curv = curvature_series(contour, fit_window=fit_window, step=step)

    # 5. Outlier removal on raw curvature
    raw_filtered, _ = remove_outliers_sigma(raw_curv, sigma=sigma)

    # 6. Moving average on contour, then curvature
    smooth_contour = moving_average_closed_contour(contour, window_size=moving_avg_window)
    _, smooth_curv = curvature_series(smooth_contour, fit_window=fit_window, step=step)

    # 7. Outlier removal after moving average
    smooth_filtered, _ = remove_outliers_sigma(smooth_curv, sigma=sigma)

    if len(smooth_filtered) == 0:
        raise ValueError("No valid curvature values remain after filtering.")

    # 8. Final representative feature = variance / mean^2
    mean_k = np.mean(smooth_filtered)
    var_k = np.var(smooth_filtered)

    if abs(mean_k) < 1e-12:
        feature_value = np.nan
    else:
        feature_value = var_k / (mean_k ** 2)

    # 9. Optional visualization during runtime
    if visualize:
        plot_curvature_stages(
            leaf_mask=mask_closed,
            raw_curv=raw_curv,
            raw_filtered=raw_filtered,
            smooth_curv=smooth_curv,
            smooth_filtered=smooth_filtered,
            curvature_score=feature_value,
            save_path=save_plot_path,
            show=show_image
        )

    debug_info = {
        "mask_closed": mask_closed,
        "contour": contour,
        "raw_curvature": raw_curv,
        "raw_curvature_filtered": raw_filtered,
        "smooth_curvature": smooth_curv,
        "smooth_curvature_filtered": smooth_filtered,
        "mean_curvature_final": mean_k,
        "variance_curvature_final": var_k,
    }

    return feature_value, debug_info

if __name__ == "__main__":
    mask = cv2.imread("image-with-apriltags (2)_binary_mask.png", cv2.IMREAD_GRAYSCALE)

    feature_value, debug = extract_leaf_curvature_feature(
        binary_mask=mask,
        close_kernel_size=3,
        moving_avg_window=5,
        fit_window=5,
        sigma=2.0,
        step=1,                 # use step=5 if you want a stricter "5-point interval" sampling
        visualize=True,         # this makes the 4 plots appear during the run
        show_image=True,
        save_plot_path="curvature_debug.png"
    )

    print("Final curvature feature =", feature_value)
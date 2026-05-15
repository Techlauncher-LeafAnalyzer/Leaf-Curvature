import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_binary_leaf_mask(mask):
    """
    Convert input mask to a clean binary mask:
    background = 0, leaf = 255
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Inputs are 0/1 segmentation masks (also handle 0/255). Convention:
    # nonzero = leaf, zero = background. Do NOT auto-invert based on area —
    # a leaf that fills more than half the frame would get flipped, and the
    # external contour would then trace the image border instead of the leaf.
    mask = (mask > 0).astype(np.uint8) * 255

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
    Compute curvature at the middle point of 5 boundary points using a
    degree-4 polynomial (with 5 points and degree 4 this is an exact fit,
    i.e., the Lagrange interpolating polynomial as specified in the paper).

    Paper (Sect. 4.1.2): the slope of the chord between the first and the
    fifth point is computed. If the chord angle (in degrees, measured from
    the x-axis and folded into [0, 180)) is > 135 or < 45, the chord is
    near-horizontal and the polynomial is built "according to the y axis"
    (i.e., y = f(x)). Otherwise the chord is near-vertical and the
    polynomial is built "according to the x axis" (i.e., x = f(y)).
    """
    points5 = np.asarray(points5, dtype=np.float64)
    x = points5[:, 0]
    y = points5[:, 1]

    # Chord angle from point 1 to point 5, folded to [0, 180).
    angle = np.degrees(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
    if angle < 0.0:
        angle += 180.0

    if angle > 135.0 or angle < 45.0:
        # Near-horizontal chord -> fit y = f(x)
        independent = x
        dependent = y
        t0 = x[2]
    else:
        # Near-vertical chord -> fit x = f(y)
        independent = y
        dependent = x
        t0 = y[2]

    # Need distinct independent values for an exact 5-point fit.
    if len(np.unique(independent)) < 5:
        return np.nan

    coeffs = np.polyfit(independent, dependent, deg=4)
    d1 = np.polyder(coeffs, 1)
    d2 = np.polyder(coeffs, 2)

    f1 = np.polyval(d1, t0)
    f2 = np.polyval(d2, t0)

    return abs(f2) / ((1.0 + f1 ** 2) ** 1.5)


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

def render_contour_with_score(image_shape, contour, curvature_score):
    """
    Render the detected boundary points on a black canvas the same size as the
    input mask, then overlay the curvature score in the top-right in vivid red.
    """
    h, w = image_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Plot each detected boundary point as a single pixel (green, BGR).
    pts = np.round(np.asarray(contour)).astype(np.int32)
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)
    canvas[ys, xs] = (0, 255, 0)

    text = f"Curvature: {curvature_score:.6f}" if np.isfinite(curvature_score) else "Curvature: nan"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 900.0)
    thickness = max(2, int(round(font_scale * 2)))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = max(10, int(round(font_scale * 15)))
    x = w - text_w - margin
    y = margin + text_h

    cv2.rectangle(
        canvas,
        (x - 6, y - text_h - 6),
        (x + text_w + 6, y + baseline + 6),
        (0, 0, 0),
        thickness=cv2.FILLED,
    )
    cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return canvas


def process_image(in_path, output_folder, step=1, save_plot=False, show_plot=False):
    """Process a single mask image. Returns curvature value, or None on failure."""
    fname = os.path.basename(in_path)
    mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"  [skip] {fname}: could not read")
        return None

    stem, ext = os.path.splitext(fname)
    plot_path = os.path.join(output_folder, f"{stem}_plot.png") if save_plot else None

    try:
        feature_value, debug = extract_leaf_curvature_feature(
            binary_mask=mask,
            close_kernel_size=3,
            moving_avg_window=5,
            fit_window=5,
            sigma=2.0,
            step=step,
            visualize=(save_plot or show_plot),
            show_image=show_plot,
            save_plot_path=plot_path,
        )
    except Exception as exc:
        print(f"  [skip] {fname}: {exc}")
        return None

    annotated = render_contour_with_score(mask.shape, debug["contour"], feature_value)
    out_path = os.path.join(output_folder, f"{stem}_curvature{ext}")
    cv2.imwrite(out_path, annotated)

    msg = f"  {fname}: curvature = {feature_value:.6f} -> {out_path}"
    if plot_path:
        msg += f" (plot: {plot_path})"
    print(msg)
    return feature_value


def process_folder(input_folder, output_folder, step=1, save_plot=False, show_plot=False):
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Input folder does not exist: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(files)} image(s) in {input_folder}")

    for fname in files:
        in_path = os.path.join(input_folder, fname)
        process_image(
            in_path,
            output_folder,
            step=step,
            save_plot=save_plot,
            show_plot=show_plot,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute leaf curvature for a binary mask image, or every "
                    "binary mask image in a folder."
    )
    parser.add_argument(
        "-i", "--input",
        default="input",
        help="A mask image file OR a folder containing mask images. Default: ./input",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_folder",
        default="output",
        help="Folder to write annotated images (and plots) into. Default: ./output",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Curvature sampling step. Use 5 for stricter 5-point interval sampling.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the 5-panel matplotlib debug plot for each image "
             "as <stem>_plot.png in the output folder.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Also display the debug plot interactively (blocks until closed). "
             "Implies --plot's figure generation but does not require saving.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    if os.path.isfile(args.input):
        process_image(
            args.input,
            args.output_folder,
            step=args.step,
            save_plot=args.plot,
            show_plot=args.show_plot,
        )
    elif os.path.isdir(args.input):
        try:
            process_folder(
                args.input,
                args.output_folder,
                step=args.step,
                save_plot=args.plot,
                show_plot=args.show_plot,
            )
        except NotADirectoryError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
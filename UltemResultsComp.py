import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_DIRECTORY = Path(".")
NUMBER_OF_BINS = 8

FILES = {
    "ball_pre": "BallFrontCameraPre.csv",
    "ball_post": "BallFrontCameraPost.csv",
    "v_pre": "VFrontCameraPre.csv",
    "v_post": "VFrontCameraPost.csv",
    "v_machine": "VMachine.csv",
    "ball_machine": "BallMachine.csv",
}

COORDINATES = ["X", "Y", "Z"]


# ============================================================
# Input functions
# ============================================================

def load_and_sort_csv(filename):
    """
    Load an 18x4 CSV containing:
        ProductionID, X, Y, Z

    The rows are sorted by ProductionID.
    """
    filepath = DATA_DIRECTORY / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")

    data = np.loadtxt(filepath, delimiter=",")

    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"{filename} must contain exactly 4 columns. "
            f"Loaded shape: {data.shape}"
        )

    sort_indices = np.argsort(data[:, 0])
    return data[sort_indices]


def check_matching_ids(data_a, data_b, name_a, name_b):
    """
    Confirm that two sorted datasets contain matching ProductionIDs.
    """
    ids_a = data_a[:, 0]
    ids_b = data_b[:, 0]

    if not np.array_equal(ids_a, ids_b):
        raise ValueError(
            f"ProductionIDs do not match between {name_a} and {name_b}."
        )


# ============================================================
# Histogram and Gaussian-fit functions
# ============================================================

def common_bin_edges(data_arrays, number_of_bins=NUMBER_OF_BINS):
    """
    Generate common histogram bin edges covering all supplied arrays.
    """
    combined = np.concatenate(data_arrays)

    minimum = np.min(combined)
    maximum = np.max(combined)

    if np.isclose(minimum, maximum):
        padding = 0.5 if minimum == 0 else 0.05 * abs(minimum)
        minimum -= padding
        maximum += padding

    return np.linspace(minimum, maximum, number_of_bins + 1)


def add_gaussian_fit(ax, values, bin_edges):
    """
    Fit an unbinned normal distribution and draw its expected histogram
    counts over the histogram.
    """
    mean, sigma = norm.fit(values)

    bin_width = bin_edges[1] - bin_edges[0]
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)

    if sigma > 0:
        y_fit = (
            len(values)
            * bin_width
            * norm.pdf(x_fit, loc=mean, scale=sigma)
        )
        ax.plot(x_fit, y_fit, linewidth=2, label="Gaussian fit")

    annotation = (
        rf"$\mu = {mean:.4f}$ mm"
        "\n"
        rf"$\sigma = {sigma:.4f}$ mm"
    )

    ax.text(
        0.97,
        0.95,
        annotation,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
    )

    return mean, sigma


# ============================================================
# Figure 1: Raw machine-coordinate distributions
# ============================================================

def plot_machine_values(ball_machine, v_machine):
    """
    Plot raw BallMachine and VMachine X, Y, and Z values.

    Each coordinate uses identical binning for Ball and V.
    """
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    for coordinate_index, coordinate_name in enumerate(COORDINATES):
        column = coordinate_index + 1

        ball_values = ball_machine[:, column]
        v_values = v_machine[:, column]

        bins = common_bin_edges([ball_values, v_values])

        axes[coordinate_index].hist(
            ball_values,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="Ball",
        )

        axes[coordinate_index].hist(
            v_values,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="V",
        )

        axes[coordinate_index].set_xlabel(
            f"{coordinate_name} Machining Variation [mm]"
        )
        axes[coordinate_index].set_ylabel("Entries")
        axes[coordinate_index].set_title(
            f"{coordinate_name} Machining Variation"
        )
        axes[coordinate_index].legend()
        axes[coordinate_index].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig("MachineCoordinateVariations.pdf", bbox_inches="tight")
    plt.close(figure)


# ============================================================
# General six-panel difference plot
# ============================================================

def plot_six_differences(
    ball_differences,
    v_differences,
    comparison_name,
    output_filename,
):
    """
    Make a 2x3 figure:

        Top row:    Ball X, Y, Z
        Bottom row: V X, Y, Z

    Each histogram is fitted to a Gaussian.
    """
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))

    for coordinate_index, coordinate_name in enumerate(COORDINATES):
        # Use common binning between Ball and V for a given coordinate.
        bins = common_bin_edges(
            [
                ball_differences[:, coordinate_index],
                v_differences[:, coordinate_index],
            ]
        )

        ball_axis = axes[0, coordinate_index]
        v_axis = axes[1, coordinate_index]

        ball_values = ball_differences[:, coordinate_index]
        v_values = v_differences[:, coordinate_index]

        ball_axis.hist(
            ball_values,
            bins=bins,
            alpha=0.7,
            edgecolor="black",
            label="Data",
        )
        add_gaussian_fit(ball_axis, ball_values, bins)

        ball_axis.set_title(
            f"Ball {coordinate_name}: {comparison_name}"
        )
        ball_axis.set_xlabel(
            f"Ball {coordinate_name} {comparison_name} [mm]"
        )
        ball_axis.set_ylabel("Entries")
        ball_axis.grid(alpha=0.25)
        ball_axis.legend()

        v_axis.hist(
            v_values,
            bins=bins,
            alpha=0.7,
            edgecolor="black",
            label="Data",
        )
        add_gaussian_fit(v_axis, v_values, bins)

        v_axis.set_title(
            f"V {coordinate_name}: {comparison_name}"
        )
        v_axis.set_xlabel(
            f"V {coordinate_name} {comparison_name} [mm]"
        )
        v_axis.set_ylabel("Entries")
        v_axis.grid(alpha=0.25)
        v_axis.legend()

    figure.suptitle(comparison_name, fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_filename, bbox_inches="tight")
    plt.close(figure)


# ============================================================
# Main analysis
# ============================================================

def main():
    data = {
        key: load_and_sort_csv(filename)
        for key, filename in FILES.items()
    }

    # Check that the datasets being subtracted have matching IDs.
    check_matching_ids(
        data["ball_post"],
        data["ball_pre"],
        FILES["ball_post"],
        FILES["ball_pre"],
    )

    check_matching_ids(
        data["v_post"],
        data["v_pre"],
        FILES["v_post"],
        FILES["v_pre"],
    )

    check_matching_ids(
        data["ball_post"],
        data["ball_machine"],
        FILES["ball_post"],
        FILES["ball_machine"],
    )

    check_matching_ids(
        data["v_post"],
        data["v_machine"],
        FILES["v_post"],
        FILES["v_machine"],
    )

    # Plot raw X, Y, and Z machine-coordinate values.
    plot_machine_values(
        data["ball_machine"],
        data["v_machine"],
    )

    # Columns 1:4 are X, Y, and Z after excluding ProductionID.
    ball_post_minus_pre = (
        data["ball_post"][:, 1:4] - data["ball_pre"][:, 1:4]
    )

    v_post_minus_pre = (
        data["v_post"][:, 1:4] - data["v_pre"][:, 1:4]
    )

    plot_six_differences(
        ball_post_minus_pre,
        v_post_minus_pre,
        comparison_name="Post - Pre",
        output_filename="PostMinusPre.pdf",
    )

    ball_post_minus_machine = (
        data["ball_post"][:, 1:4] - data["ball_machine"][:, 1:4]
    )

    v_post_minus_machine = (
        data["v_post"][:, 1:4] - data["v_machine"][:, 1:4]
    )

    plot_six_differences(
        ball_post_minus_machine,
        v_post_minus_machine,
        comparison_name="Post - Machine",
        output_filename="PostMinusMachine.pdf",
    )

    print("Analysis complete.")
    print("Created:")
    print("  MachineCoordinateVariations.pdf")
    print("  PostMinusPre.pdf")
    print("  PostMinusMachine.pdf")


if __name__ == "__main__":
    main()

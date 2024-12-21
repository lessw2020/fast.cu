import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_benchmark_results(csv_file="benchmark_results.csv"):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a color palette for different kernels
    kernels = sorted(df["kernel"].unique())
    colors = sns.color_palette("husl", n_colors=len(kernels))
    kernel_colors = dict(zip(kernels, colors))

    # Set up the plotting style
    sns.set_theme(style="whitegrid")

    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("CUDA Kernel Performance Analysis", fontsize=16, y=1.05)

    # Plot 1: TFLOPS vs Matrix Size
    for kernel in kernels:
        kernel_data = df[df["kernel"] == kernel]
        label = "cuBLAS" if kernel == 0 else f"Kernel {kernel}"
        ax1.plot(
            kernel_data["matrix_size"],
            kernel_data["tflops"],
            marker="o",
            label=label,
            color=kernel_colors[kernel],
        )

    ax1.set_xlabel("Matrix Size")
    ax1.set_ylabel("Performance (TFLOPS)")
    ax1.set_title("Performance Scaling with Matrix Size")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)

    # Plot 2: Relative Performance vs Matrix Size (excluding cuBLAS)
    for kernel in kernels[1:]:  # Skip cuBLAS
        kernel_data = df[df["kernel"] == kernel]
        ax2.plot(
            kernel_data["matrix_size"],
            kernel_data["relative_perf"],
            marker="o",
            label=f"Kernel {kernel}",
            color=kernel_colors[kernel],
        )

    ax2.set_xlabel("Matrix Size")
    ax2.set_ylabel("Relative Performance vs cuBLAS (%)")
    ax2.set_title("Performance Relative to cuBLAS")
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)  # Add reference line at 0%
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)

    # Plot 3: Execution Time vs Matrix Size
    for kernel in kernels:
        kernel_data = df[df["kernel"] == kernel]
        label = "cuBLAS" if kernel == 0 else f"Kernel {kernel}"
        ax3.plot(
            kernel_data["matrix_size"],
            kernel_data["time"] * 1000,  # Convert to ms
            marker="o",
            label=label,
            color=kernel_colors[kernel],
        )

    ax3.set_xlabel("Matrix Size")
    ax3.set_ylabel("Execution Time (ms)")
    ax3.set_title("Execution Time Scaling")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("benchmark_results.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_single_matrix_size(csv_file="benchmark_results.csv", matrix_size=8192):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter for specific matrix size
    df_size = df[df["matrix_size"] == matrix_size]

    # Create bar plot
    plt.figure(figsize=(12, 6))

    # Plot bars for TFLOPS
    bars = plt.bar(df_size["kernel"], df_size["tflops"])

    # Customize the plot
    plt.title(
        f"Kernel Performance Comparison (Matrix Size: {matrix_size}x{matrix_size})"
    )
    plt.xlabel("Kernel Number (0 = cuBLAS)")
    plt.ylabel("Performance (TFLOPS)")
    plt.grid(True, alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"benchmark_results_{matrix_size}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Generate both types of plots
    plot_benchmark_results()

    # Generate individual plots for each matrix size in the dataset
    df = pd.read_csv("benchmark_results.csv")
    for size in df["matrix_size"].unique():
        plot_single_matrix_size(matrix_size=size)

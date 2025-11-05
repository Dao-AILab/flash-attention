"""
Standalone script to visualize benchmark results from CSV file.
Usage: python visualize_benchmark_results.py [csv_file]
"""

import sys


def visualize_results(csv_file="block_sparsity_benchmark_results.csv"):
    """Create line plots comparing Fast vs Full kernels for each mask mod."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
    except (ImportError, SystemError) as e:
        print(f"Pandas or matplotlib not available: {e}")
        return

    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} benchmark results from {csv_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Get unique mask mods
    mask_mods = df['Mask Mod'].unique()
    print(f"Found {len(mask_mods)} mask types: {', '.join(mask_mods)}")

    # Create a figure with subplots for each mask mod
    n_masks = len(mask_mods)
    fig, axes = plt.subplots(n_masks, 2, figsize=(16, 5 * n_masks))

    # Handle case where there's only one mask
    if n_masks == 1:
        axes = axes.reshape(1, -1)

    for idx, mask_name in enumerate(mask_mods):
        mask_df = df[df['Mask Mod'] == mask_name]

        # Skip if no kernel type (e.g., doc_mask_mod which only has Full)
        if 'Kernel Type' not in mask_df.columns or mask_df['Kernel Type'].isna().all():
            print(f"Skipping {mask_name} (no kernel type comparison)")
            # Hide these axes
            if n_masks > 1:
                axes[idx, 0].set_visible(False)
                axes[idx, 1].set_visible(False)
            continue

        # Plot 1: Creation Time vs Sequence Length (for different batch sizes)
        ax1 = axes[idx, 0] if n_masks > 1 else axes[0]

        # Group by batch size and kernel type
        for batch_size in sorted(mask_df['B'].unique()):
            for kernel_type in ['Full', 'Fast']:
                subset = mask_df[(mask_df['B'] == batch_size) &
                                (mask_df['Kernel Type'] == kernel_type)]
                if len(subset) > 0:
                    # Average over different num_heads
                    grouped = subset.groupby('M')['Creation Time (ms)'].mean().reset_index()
                    linestyle = '-' if kernel_type == 'Full' else '--'
                    ax1.plot(grouped['M'], grouped['Creation Time (ms)'],
                            marker='o', linestyle=linestyle,
                            label=f'B={batch_size}, {kernel_type}')

        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Creation Time (ms)')
        ax1.set_title(f'{mask_name}: Creation Time vs Sequence Length')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')

        # Plot 2: Speedup (Full/Fast) vs Sequence Length
        ax2 = axes[idx, 1] if n_masks > 1 else axes[1]

        for batch_size in sorted(mask_df['B'].unique()):
            speedups = []
            seqlens = []

            for seqlen in sorted(mask_df['M'].unique()):
                full_subset = mask_df[(mask_df['B'] == batch_size) &
                                     (mask_df['M'] == seqlen) &
                                     (mask_df['Kernel Type'] == 'Full')]
                fast_subset = mask_df[(mask_df['B'] == batch_size) &
                                     (mask_df['M'] == seqlen) &
                                     (mask_df['Kernel Type'] == 'Fast')]

                if len(full_subset) > 0 and len(fast_subset) > 0:
                    full_time = full_subset['Creation Time (ms)'].mean()
                    fast_time = fast_subset['Creation Time (ms)'].mean()
                    speedup = full_time / fast_time
                    speedups.append(speedup)
                    seqlens.append(seqlen)

            if speedups:
                ax2.plot(seqlens, speedups, marker='o', label=f'B={batch_size}')

        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Speedup (Full / Fast)')
        ax2.set_title(f'{mask_name}: Fast Kernel Speedup')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')

    plt.tight_layout()
    output_file = 'block_sparsity_benchmark_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to {output_file}")

    # Also create a summary table showing speedups
    print("\n" + "="*80)
    print("SPEEDUP SUMMARY (Full / Fast)")
    print("="*80)

    for mask_name in mask_mods:
        mask_df = df[df['Mask Mod'] == mask_name]
        if 'Kernel Type' not in mask_df.columns or mask_df['Kernel Type'].isna().all():
            continue

        print(f"\n{mask_name}:")
        print("-" * 80)

        # Create a pivot table
        for seqlen in sorted(mask_df['M'].unique()):
            seqlen_df = mask_df[mask_df['M'] == seqlen]
            full_time = seqlen_df[seqlen_df['Kernel Type'] == 'Full']['Creation Time (ms)'].mean()
            fast_time = seqlen_df[seqlen_df['Kernel Type'] == 'Fast']['Creation Time (ms)'].mean()

            if not pd.isna(full_time) and not pd.isna(fast_time):
                speedup = full_time / fast_time
                print(f"  SeqLen {seqlen:5d}: Full={full_time:8.4f}ms, Fast={fast_time:8.4f}ms, Speedup={speedup:6.2f}x")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "block_sparsity_benchmark_results.csv"
    visualize_results(csv_file)

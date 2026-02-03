import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Load your CSV ---
run = '50'
output_path = f"outputs/run{run}/results/"
save_path =f"outputs/run{run}/plots/"
df = pd.read_csv(f"{output_path}cross_modal_similarities.csv")
js = df["JS_Divergence"].to_numpy()
cos = df["Cosine_Similarity"].to_numpy()

# --- Helpers ---
def fd_bins(x, max_bins=60):
    x = np.asarray(x); x = x[~np.isnan(x)]
    if len(x) < 2: return 10
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, max(10, int(np.sqrt(len(x)))))
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return min(max_bins, 30)
    b = int(np.ceil((x.max() - x.min()) / h))
    return max(10, min(b, max_bins))

def silverman_bandwidth(x):
    x = np.asarray(x); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2: return 0.1
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr/1.349) if (std>0 and iqr>0) else max(std, iqr/1.349)
    if sigma == 0: sigma = std if std>0 else 0.1
    return 0.9 * sigma * n**(-1/5)

def gaussian_kde_pdf(x, sample, bandwidth):
    sample = np.asarray(sample); sample = sample[~np.isnan(sample)]
    n = len(sample)
    if n == 0: return np.zeros_like(x)
    h = bandwidth if bandwidth and bandwidth>0 else silverman_bandwidth(sample)
    inv = 1.0/(np.sqrt(2*np.pi)*h*n)
    diffs = (x[:, None] - sample[None, :]) / h
    return inv * np.sum(np.exp(-0.5*diffs*diffs), axis=1)

def draw_hist_panel(ax, data, title, cmap_pos, xlim=(0,1)):
    data = np.asarray(data); data = data[~np.isnan(data)]
    n = len(data)
    viridis = cm.get_cmap("viridis")
    color_line = viridis(cmap_pos)
    
    bins = fd_bins(data)
    counts, edges = np.histogram(data, bins=bins, range=xlim, density=False)
    bin_width = edges[1]-edges[0]
    ymax = max(counts)*1.2 if len(counts)>0 else 1
    
    centers = (edges[:-1] + edges[1:]) / 2
    colors = viridis(centers)
    for c, cnt, left in zip(colors, counts, edges[:-1]):
        ax.bar(left + bin_width/2, cnt, width=bin_width*0.95, align='center',
               color=c, edgecolor="black", linewidth=0.8)
    
    xs = np.linspace(xlim[0], xlim[1], 400)
    kde = gaussian_kde_pdf(xs, data, silverman_bandwidth(data))
    ax.plot(xs, kde * n * bin_width, color=color_line, linewidth=3, alpha=0.9)
    
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.axvspan(q1, q3, color=color_line, alpha=0.12, lw=0)
    ax.axvline(med, color=color_line, linestyle="--", linewidth=2.5, alpha=0.9)
    
    # Rug ticks
    ax.vlines(data, ymin=0, ymax=0.03*ymax, color=color_line, alpha=0.4, linewidth=1)
    
    # Labels / fonts
    ax.set_title(title, fontsize=28, pad=12)
    ax.set_xlim(*xlim); ax.set_ylim(0, ymax)
    ax.set_xlabel(title.split()[0]+"D" if "JS" in title else "Cosine Similarity", fontsize=22)
    ax.set_ylabel("Frequency", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(axis="y", alpha=0.25)
    
    # Annotation
    ax.text(0.985, 0.92,
            f"n={n}\nmedian={med:.2f}\nIQR=({q1:.2f}, {q3:.2f})",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=18, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color_line, lw=2, alpha=0.95))

# --- Plot ---
plt.figure(figsize=(16,7))

# LEFT: Cosine Similarity
ax1 = plt.subplot(1,2,1)
draw_hist_panel(
    ax1,
    cos,
    "Cosine Similarity per Patient",
    cmap_pos=0.75,   # greener/yellow viridis
    xlim=(0,1)
)

# RIGHT: JS Divergence
ax2 = plt.subplot(1,2,2)
draw_hist_panel(
    ax2,
    js,
    "JS Divergence per Patient",
    cmap_pos=0.2,    # purple/blue viridis
    xlim=(0,1)
)

plt.tight_layout()
plt.savefig(
    f"{save_path}similarity_histogram_largefont.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = (
    Path("ai4i+2020+predictive+maintenance+dataset")
    / "ai4i2020.csv"
)
df = pd.read_csv(file_path)

# Modern, consistent visual style
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "#F5F7FB",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#DCE3ED",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
})

NAVY = "#243B64"
BLUE = "#3977D5"
ORANGE = "#F28E2B"
RED = "#D94F70"
MUTED = "#65748B"

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#F5F7FB")

fig.suptitle(
    "AI4I 2020 | Predictive Maintenance Dashboard",
    fontsize=21,
    fontweight="bold",
    color=NAVY,
    y=0.98,
)

# Graph 1: Machine failure overview
ax = axes[0, 0]
counts = df["Machine failure"].value_counts().reindex(
    [0, 1], fill_value=0
)
bars = ax.bar(
    ["Normal", "Machine failure"],
    counts.values,
    color=[BLUE, RED],
    width=0.58,
    edgecolor="white",
    linewidth=1.5,
)

ax.set_title("Machine Failure Overview", loc="left", pad=14)
ax.set_ylabel("Number of observations")
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

for bar, count in zip(bars, counts.values):
    percentage = count / len(df) * 100
    ax.annotate(
        f"{count:,}\n({percentage:.2f}%)",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=NAVY,
    )

ax.set_ylim(0, counts.max() * 1.18)

# Graph 2: Air temperature distribution
ax = axes[0, 1]
sns.histplot(
    data=df,
    x="Air temperature [K]",
    bins=30,
    kde=True,
    color=BLUE,
    edgecolor="white",
    alpha=0.82,
    ax=ax,
)
ax.set_title("Air Temperature Distribution", loc="left", pad=14)
ax.set_xlabel("Air temperature (K)")
ax.set_ylabel("Number of observations")
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

# Graph 3: Rotational speed versus torque
ax = axes[1, 0]
sns.scatterplot(
    data=df,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    hue="Machine failure",
    hue_order=[0, 1],
    palette={0: BLUE, 1: RED},
    alpha=0.62,
    s=30,
    edgecolor=None,
    linewidth=0,
    ax=ax,
)
ax.set_title("Rotational Speed vs. Torque", loc="left", pad=14)
ax.set_xlabel("Rotational speed (rpm)")
ax.set_ylabel("Torque (Nm)")
ax.legend(
    title="Machine status",
    labels=["Normal", "Failure"],
    frameon=True,
    facecolor="white",
)
ax.grid(alpha=0.22)
ax.set_axisbelow(True)

# Graph 4: Correlation heatmap
ax = axes[1, 1]
columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

labels = [
    "Air temp.",
    "Process temp.",
    "Speed",
    "Torque",
    "Tool wear",
    "Failure",
]

corr = df[columns].corr()
corr.index = labels
corr.columns = labels

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=1,
    linecolor="white",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
    ax=ax,
)
ax.set_title("Sensor Feature Correlations", loc="left", pad=14)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)

# Final layout and high-resolution export
for ax in axes.flat:
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.95])

output_file = "ai4i_dataset_plots.png"
fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

print("Dashboard saved to:", Path(output_file).resolve())
plt.show()

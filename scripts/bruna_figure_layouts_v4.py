"""Layouts V4 injetados no notebook de figuras; dados canônicos permanecem iguais.

As funções usam os objetos carregados pelo notebook 09. Não contêm dados
sintéticos nem estimativas substitutas. Suptítulos de SubFigure e colorbars
locais participam do constrained layout, sem coordenadas globais fixas.
"""

def build_experimental_figure(language: str):
    text = TEXT_EXPERIMENTAL[language]
    fig = plt.figure(figsize=(15.4, 10.0), layout="constrained")
    subfigs = fig.subfigures(2, 2, width_ratios=[2.1, 1.05], height_ratios=[0.82, 1.18], wspace=0.08, hspace=0.10)
    sf_frames, sf_density = subfigs[0]
    sf_blocks, sf_perf = subfigs[1]
    sf_frames.suptitle("A  " + text["frames"], fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes_frames = sf_frames.subplots(1, len(EXAMPLES), squeeze=False)[0]
    for ax, (_, row) in zip(axes_frames, EXAMPLES.iterrows()):
        ax.imshow(read_video_frames(str(row["file_stem"]), [0.5])[0])
        ax.set_title(f"{row.get('Race', '')}\n{row.get('label_name', '')}", fontsize=8, pad=5)
        ax.set_axis_off()

    ax_density = sf_density.subplots()
    extent = [DENSITY_X[0], DENSITY_X[-1], DENSITY_Y[0], DENSITY_Y[-1]]
    im = ax_density.imshow(DENSITY, origin="lower", extent=extent, cmap="magma", aspect="equal")
    ax_density.set_title("B  " + text["density"], pad=8)
    ax_density.set_xlabel("x");ax_density.set_ylabel("-y")
    ax_density.text(0.02, 0.02, f"n={DENSITY_N_VIDEOS:,} videos\n{DENSITY_N_POINTS:,} points\n{text['density_note']}", transform=ax_density.transAxes, color="white", fontsize=7, va="bottom")
    sf_density.colorbar(im, ax=ax_density, fraction=0.045, pad=0.04)

    sf_blocks.suptitle("C  " + text["kinematic"], fontsize=11, fontweight="bold", color=PALETTE["navy"])
    axes_blocks = sf_blocks.subplots(2, 2)
    for ax, block, title in zip(axes_blocks.flat, compute_temporal_blocks(PRIMARY_SEQ), text["blocks"]):
        matrix = block_magnitude(block).T
        _, vmax = robust_limits(matrix)
        img = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, pad=6);ax.set_xlabel(text["time"]);ax.set_ylabel(text["landmark"])
        ax.set_xticks([0, 49, 99], ["0", "0.5", "1"])
        sf_blocks.colorbar(img, ax=ax, fraction=0.035, pad=0.025)

    ax_perf = sf_perf.subplots()
    order = ["FlatMLP", "TemporalCNN1D", "EmoT", "EmoT_Reweighted"]
    model_colors = ["#6C757D", "#3A86FF", "#00A896", "#F77F00"]
    for i, (model, color) in enumerate(zip(order, model_colors)):
        values = CV_FOLDS.loc[CV_FOLDS["model"] == model, "macro_f1"].to_numpy(float)
        if len(values) != 5:
            raise ValueError(f"Esperados cinco folds reais para {model}; encontrados {len(values)}")
        offsets = np.linspace(-0.08, 0.08, len(values))
        ax_perf.scatter(np.full(len(values), i)+offsets, values, s=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
        mean, std = values.mean(), values.std(ddof=1)
        ax_perf.errorbar(i, mean, yerr=std, fmt="o", color=PALETTE["navy"], mfc="white", ms=7, capsize=4, lw=1.5, zorder=4)
        ax_perf.text(i, mean+std+0.012, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom", fontsize=8)
    ax_perf.set_xticks(range(4), text["models"], rotation=25, ha="right")
    ax_perf.set_ylabel(text["macro"]);ax_perf.set_title("D  " + text["performance"], pad=8)
    ax_perf.grid(axis="y", color="#DDE3EA", linewidth=0.7);ax_perf.set_ylim(0.52, 0.77)
    fig.canvas.draw()
    return fig


def build_architecture_figure(language: str):
    text = TEXT_ARCH[language]
    fig = plt.figure(figsize=(16.0, 7.2), layout="constrained")
    sf_input, sf_projection, sf_attention, sf_output = fig.subfigures(1, 4, width_ratios=[1.05, 0.95, 2.1, 1.2], wspace=0.12)
    ax_input = sf_input.subplots()
    low, high = robust_limits(PRIMARY_STANDARDIZED, symmetric=True)
    ax_input.imshow(PRIMARY_STANDARDIZED.T, aspect="auto", origin="lower", cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=low, vmax=high), interpolation="nearest")
    for boundary in [310, 620, 930]:ax_input.axhline(boundary-0.5, color="black", lw=0.45, alpha=0.7)
    ax_input.set_title("A  " + text["input"], pad=8)
    ax_input.set_xlabel(text["time"]);ax_input.set_ylabel(text["feature"])
    ax_input.set_yticks([155, 465, 775, 1085], ["pos", "Δ¹", "Δ²", "Δ³"])

    ax_projection = sf_projection.subplots()
    low, high = robust_limits(PROJECTED_MATRIX, symmetric=True)
    ax_projection.imshow(PROJECTED_MATRIX.T, aspect="auto", origin="lower", cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=low, vmax=high), interpolation="nearest")
    ax_projection.set_title("B  " + text["projection"], pad=8)
    ax_projection.set_xlabel(text["time"]);ax_projection.set_ylabel(text["hidden"])

    sf_attention.suptitle("C  " + text["attention"], fontsize=10.5, fontweight="bold", color=PALETTE["navy"])
    axes_attention = sf_attention.subplots(2, 3)
    positive = ATTENTION_MATRICES[ATTENTION_MATRICES > 0]
    if positive.size == 0:raise ValueError("Matrizes de atenção sem massa positiva")
    vmin = max(float(np.quantile(positive, 0.02)), 1e-6)
    vmax = max(float(np.quantile(positive, 0.998)), vmin*1.01)
    for layer, ax in enumerate(axes_attention.flat):
        im_attn = ax.imshow(ATTENTION_MATRICES[layer], origin="lower", cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest", aspect="equal")
        ax.set_title(f"L{layer+1}", pad=5)
        ax.set_xticks([0, 50, 100]);ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=7)
    sf_attention.colorbar(im_attn, ax=axes_attention.ravel().tolist(), fraction=0.025, pad=0.04, shrink=0.70)

    ax_output = sf_output.subplots()
    order = np.argsort(PROBABILITIES)
    bar_colors = [PALETTE["blue"] if idx == int(PROBABILITIES.argmax()) else "#AAB4C3" for idx in order]
    ax_output.barh(np.arange(6), PROBABILITIES[order], color=bar_colors)
    ax_output.set_yticks(np.arange(6), [text["classes"][idx] for idx in order])
    ax_output.set_xlabel("p");ax_output.set_xlim(0, 1.05)
    ax_output.set_title("D  " + text["output"], pad=8)
    for pos, value in enumerate(PROBABILITIES[order]):ax_output.text(value+0.015, pos, f"{value:.3f}", va="center", fontsize=8)
    # A ordem A-B-C-D indica o fluxo; não se adicionam setas em posições
    # calculadas antes do constrained layout.
    fig.canvas.draw()
    return fig


def validate_figure_layout(fig):
    """Falha se um título cobrir a área de dados de qualquer eixo visível."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    texts = [ax.title for ax in axes if ax.title.get_text()]
    def collect_subfigure_titles(owner):
        out = list(owner.texts)
        for sub in getattr(owner, "subfigs", []):out.extend(collect_subfigure_titles(sub))
        return out
    texts.extend(collect_subfigure_titles(fig))
    collisions = []
    for text in texts:
        if not text.get_visible() or not text.get_text():continue
        box = text.get_window_extent(renderer)
        for ax in axes:
            rect = ax.get_window_extent(renderer)
            width = min(box.x1,rect.x1)-max(box.x0,rect.x0)
            height = min(box.y1,rect.y1)-max(box.y0,rect.y0)
            if width > 1.0 and height > 1.0:
                collisions.append({"title":text.get_text(),"axes_title":ax.get_title(),"width_px":float(width),"height_px":float(height)})
    if collisions:raise RuntimeError(f"Sobreposição de título/área de dados: {collisions}")
    return {"titles_checked":len(texts),"axes_checked":len(axes),"title_data_overlaps":0}

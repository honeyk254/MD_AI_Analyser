"""
Plotly chart generators for all analysis modules.
Each function returns a Plotly figure as a JSON string.
"""
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def generate_all_plots(result) -> dict:
    """Generate all visualization plots from analysis results."""
    plots = {}

    generators = [
        ("rmsd_plot", _plot_rmsd, result.rmsd),
        ("rmsf_plot", _plot_rmsf, result.rmsf),
        ("rg_plot", _plot_rg, result.rg),
        ("ss_plot", _plot_secondary_structure, result.secondary_structure),
        ("hbond_plot", _plot_hbonds, result.hbonds),
        ("salt_bridges_plot", _plot_salt_bridges, result.salt_bridges),
        ("contact_map", _plot_contact_map, result.contacts),
        ("pca_plot", _plot_pca, result.pca),
        ("dccm_plot", _plot_dccm, result.dccm),
        ("fel_plot", _plot_free_energy, result.free_energy),
        ("clustering_plot", _plot_clustering, result.clustering),
        ("sasa_plot", _plot_sasa, result.sasa),
        ("tica_plot", _plot_tica, result.tica),
        ("dimensionality_plot", _plot_dimensionality, result.dimensionality),
        ("dimensionality_3d_plot", _plot_dimensionality_3d, result.dimensionality),
        ("gnn_plot", _plot_gnn, result.gnn_results),
        ("transformer_plot", _plot_transformer, result.transformer_results),
        ("msm_plot", _plot_msm, result.msm),
        # Part A plots
        ("water_bridges_plot", _plot_water_bridges, result.water_bridges),
        ("energy_plot", _plot_energy_decomposition, result.energy_decomposition),
        ("prs_plot", _plot_prs, result.prs),
        ("nma_plot", _plot_nma, result.nma),
        ("entropy_plot", _plot_entropy, result.entropy),
        ("ifp_plot", _plot_ifp, result.interaction_fingerprints),
        ("tunnel_plot", _plot_tunnels, result.tunnels),
        ("vae_plot", _plot_vae, result.vae),
        ("dynamic_network_plot", _plot_dynamic_network, result.dynamic_network),
        # Phase 4 new plots
        ("convergence_plot", _plot_convergence, result.convergence),
        ("binding_kinetics_plot", _plot_binding_kinetics, result.binding_kinetics),
        ("network_graph_plot", _plot_allosteric_network, result.allosteric),
        ("training_loss_plot", _plot_training_losses, {
            "gnn": result.gnn_results, "transformer": result.transformer_results, "vae": result.vae
        }),
    ]

    for name, func, data in generators:
        try:
            if isinstance(data, dict) and not data.get("error"):
                fig = func(data)
                if fig:
                    plots[name] = fig.to_json()
        except Exception as e:
            print(f"[WARN] Plot {name} failed: {e}")

    return plots


def _dark_layout(fig, title, xaxis="", yaxis=""):
    """Apply dark theme styling to a plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#E0E0E0")),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"),
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def _plot_rmsd(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["rmsd"],
        mode='lines', name='RMSD',
        line=dict(color='#00d4ff', width=1.5),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.1)',
    ))
    # Equilibration line
    if data.get("equilibration_frame", 0) > 0 and data["time"]:
        equil_time = data["time"][data["equilibration_frame"]]
        fig.add_vline(x=equil_time, line_dash="dash", line_color="#ff6b6b",
                     annotation_text="Equilibration")
    # Mean line
    fig.add_hline(y=data["mean_rmsd"], line_dash="dot", line_color="#ffd93d",
                 annotation_text=f"Mean: {data['mean_rmsd']:.2f} Å")
    return _dark_layout(fig, "RMSD Over Time", "Time (ps)", "RMSD (Å)")


def _plot_rmsf(data):
    rmsf = data.get("rmsf", [])
    resids = data.get("resids", [])
    if not rmsf or not resids or len(rmsf) != len(resids):  # item 62: validation
        return None
    colors = []
    mean_r = data.get("mean_rmsf", np.mean(rmsf))
    std_r = data.get("std_rmsf", np.std(rmsf))
    for v in rmsf:
        if v > mean_r + std_r:
            colors.append("#ff6b6b")
        elif v < mean_r - 0.5 * std_r:
            colors.append("#4ecdc4")
        else:
            colors.append("#00d4ff")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resids, y=rmsf,
        marker_color=colors, name='RMSF',
    ))
    fig.add_hline(y=mean_r, line_dash="dot", line_color="#ffd93d",
                 annotation_text=f"Mean: {mean_r:.2f} Å")
    return _dark_layout(fig, "Per-Residue RMSF", "Residue ID", "RMSF (Å)")


def _plot_rg(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["rg"],
        mode='lines', name='Rg',
        line=dict(color='#a29bfe', width=1.5),
        fill='tozeroy', fillcolor='rgba(162,155,254,0.1)',
    ))
    return _dark_layout(fig, f"Radius of Gyration (Trend: {data.get('trend', 'N/A')})",
                        "Time (ps)", "Rg (Å)")


def _plot_secondary_structure(data):
    fig = go.Figure()
    n_frames = len(data.get("helix_fraction", []))
    frames = list(range(n_frames))
    fig.add_trace(go.Scatter(x=frames, y=data["helix_fraction"], mode='lines',
                             name='α-Helix', line=dict(color='#ff6b6b', width=1.5)))
    fig.add_trace(go.Scatter(x=frames, y=data["sheet_fraction"], mode='lines',
                             name='β-Sheet', line=dict(color='#4ecdc4', width=1.5)))
    fig.add_trace(go.Scatter(x=frames, y=data["coil_fraction"], mode='lines',
                             name='Coil', line=dict(color='#ffd93d', width=1.5)))
    return _dark_layout(fig, "Secondary Structure Evolution", "Frame", "Fraction")


def _plot_hbonds(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["n_hbonds"],
        mode='lines', name='H-bonds',
        line=dict(color='#6c5ce7', width=1.5),
        fill='tozeroy', fillcolor='rgba(108,92,231,0.1)',
    ))
    return _dark_layout(fig, f"Hydrogen Bonds (Mean: {data.get('mean_hbonds', 0):.1f})",
                        "Time (ps)", "Number of H-bonds")


def _plot_salt_bridges(data):
    pairs = data.get("pairs", [])
    time = data.get("time", [])
    total_per_frame = data.get("total_per_frame", [])
    if not pairs and not total_per_frame:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Salt Bridges Over Time", "Top Salt Bridge Pairs (Occupancy)"),
        row_heights=[0.5, 0.5])
    if time and total_per_frame:
        fig.add_trace(go.Scatter(
            x=time, y=total_per_frame,
            mode='lines', name='Salt Bridges/Frame',
            line=dict(color='#ffd93d', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,217,61,0.1)',
        ), row=1, col=1)
    if pairs:
        labels = [f"{p['positive']}-{p['negative']}" for p in pairs[:20]]
        occupancies = [p['occupancy'] for p in pairs[:20]]
        fig.add_trace(go.Bar(
            x=labels, y=occupancies,
            marker_color='#fdcb6e', name='Occupancy',
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text=f"Salt Bridges (Mean: {data.get('mean_salt_bridges', 0):.1f}/frame)",
                   font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_contact_map(data):
    cmap = data.get("contact_map", [])
    resids = data.get("resids", [])
    if not cmap or not resids:  # item 62: validation
        return None
    fig = go.Figure(data=go.Heatmap(
        z=cmap,
        x=resids,
        y=resids,
        colorscale="Viridis",
        colorbar=dict(title="Contact Freq"),
    ))
    return _dark_layout(fig, "Average Contact Map", "Residue ID", "Residue ID")


def _plot_pca(data):
    proj = np.array(data["projections"])
    if proj.shape[1] < 2:
        return None

    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=("PCA Projection", "Explained Variance"))
    fig.add_trace(go.Scatter(
        x=proj[:, 0], y=proj[:, 1],
        mode='markers', name='Conformations',
        marker=dict(color=list(range(len(proj))), colorscale='Plasma',
                   size=4, colorbar=dict(title="Frame", x=0.45)),
    ), row=1, col=1)

    cum_var = data["cumulative_variance"]
    fig.add_trace(go.Bar(
        x=list(range(1, len(cum_var)+1)), y=cum_var,
        marker_color='#00d4ff', name='Cumulative Variance',
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Principal Component Analysis", font=dict(size=18, color="#E0E0E0")),
    )
    return fig


def _plot_dccm(data):
    dccm = data.get("dccm", [])
    resids = data.get("resids", [])
    if not dccm or not resids:  # item 62: validation
        return None
    fig = go.Figure(data=go.Heatmap(
        z=dccm,
        x=resids,
        y=resids,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Correlation"),
    ))
    return _dark_layout(fig, "Dynamic Cross-Correlation Matrix", "Residue ID", "Residue ID")


def _plot_free_energy(data):
    fig = go.Figure(data=go.Contour(
        z=data["fel"],
        x=data["pc1_edges"][:-1],
        y=data["pc2_edges"][:-1],
        colorscale="Magma_r",
        colorbar=dict(title="ΔG (kJ/mol)"),
        contours=dict(showlabels=True),
    ))
    # Mark minima
    minima = data.get("minima", [])
    if minima:
        fig.add_trace(go.Scatter(
            x=[m["pc1"] for m in minima],
            y=[m["pc2"] for m in minima],
            mode='markers+text',
            text=[f'Min {i+1}' for i in range(len(minima))],
            marker=dict(color='#ff6b6b', size=12, symbol='star'),
            textposition="top center",
            textfont=dict(color="#ff6b6b"),
            name='Minima',
        ))
    return _dark_layout(fig, "Free Energy Landscape", "PC1", "PC2")


def _plot_clustering(data):
    fig = go.Figure()
    labels = data["labels"]
    populations = data.get("populations", {})

    fig.add_trace(go.Scatter(
        x=list(range(len(labels))), y=labels,
        mode='markers', name='Cluster',
        marker=dict(color=labels, colorscale='Set1', size=3),
    ))

    return _dark_layout(fig, f"Conformational Clusters (k={data['n_clusters']}, "
                        f"silhouette={data.get('silhouette_score', 0):.2f})",
                        "Frame", "Cluster ID")


def _plot_sasa(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time"], y=data["total_sasa"],
        mode='lines', name='Total SASA',
        line=dict(color='#fd79a8', width=1.5),
        fill='tozeroy', fillcolor='rgba(253,121,168,0.1)',
    ))
    return _dark_layout(fig, "Solvent Accessible Surface Area", "Time (ps)", "SASA (nm²)")


def _plot_dimensionality(data):
    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=("PCA 2D", "UMAP 2D", "t-SNE 2D"))

    for idx, (key, title) in enumerate([("pca_2d", "PCA"), ("umap_2d", "UMAP"), ("tsne_2d", "t-SNE")]):
        proj = data.get(key, [])
        if proj and len(proj) > 0:
            arr = np.array(proj)
            fig.add_trace(go.Scatter(
                x=arr[:, 0], y=arr[:, 1],
                mode='markers', name=title,
                marker=dict(color=list(range(len(arr))), colorscale='Plasma', size=3),
            ), row=1, col=idx+1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=40, r=20, t=60, b=40),
        title=dict(text="Dimensionality Reduction", font=dict(size=18, color="#E0E0E0")),
        showlegend=False,
    )
    return fig


def _plot_gnn(data):
    fig = go.Figure()
    resids = data.get("resids", [])
    importance = data.get("residue_importance", [])
    if resids and importance:
        fig.add_trace(go.Bar(
            x=resids, y=importance,
            marker_color='#e17055', name='GNN Importance',
        ))
    return _dark_layout(fig, "GNN Residue Importance Scores", "Residue ID", "Importance")


def _plot_transformer(data):
    fig = go.Figure()
    temporal = data.get("temporal_importance", [])
    if temporal:
        fig.add_trace(go.Scatter(
            x=list(range(len(temporal))), y=temporal,
            mode='lines', name='Frame Importance',
            line=dict(color='#00cec9', width=1.5),
            fill='tozeroy', fillcolor='rgba(0,206,201,0.1)',
        ))
        # Mark transitions
        transitions = data.get("transition_frames", [])
        if transitions:
            fig.add_trace(go.Scatter(
                x=[t["frame"] for t in transitions],
                y=[t["frame_importance"] for t in transitions],
                mode='markers', name='Transitions',
                marker=dict(color='#ff6b6b', size=10, symbol='diamond'),
            ))
    return _dark_layout(fig, "Transformer: Temporal Importance & Transitions",
                        "Frame", "Importance")


def _plot_msm(data):
    T = np.array(data.get("transition_matrix", []))
    if T.size == 0:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=T,
        colorscale="Blues",
        colorbar=dict(title="Probability"),
    ))
    return _dark_layout(fig, "MSM Transition Probability Matrix", "To State", "From State")


def _plot_tica(data):
    projections = data.get("projections", [])
    timescales = data.get("timescales", [])
    if not projections:
        return None
    arr = np.array(projections)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "tICA Projection (tIC1 vs tIC2)", "Implied Timescales"))
    if arr.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=arr[:, 0], y=arr[:, 1],
            mode='markers', name='Frames',
            marker=dict(color=list(range(len(arr))), colorscale='Viridis',
                       size=4, colorbar=dict(title="Frame", x=0.45)),
        ), row=1, col=1)
    if timescales:
        valid_ts = [(i + 1, t) for i, t in enumerate(timescales) if t != float('inf') and t > 0]
        if valid_ts:
            fig.add_trace(go.Bar(
                x=[f"tIC{i}" for i, _ in valid_ts],
                y=[t for _, t in valid_ts],
                marker_color='#55efc4', name='Timescale',
            ), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text=f"tICA Analysis (lag={data.get('lag_time', '?')})",
                   font=dict(size=18, color="#E0E0E0")),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Part A — New Plot Generators
# ──────────────────────────────────────────────────────────────

def _plot_water_bridges(data):
    bridges = data.get("bridges", [])
    if not bridges:
        return None
    fig = go.Figure()
    labels = [f"{b['resid_1']}-{b['resid_2']}" for b in bridges[:30]]
    occupancies = [b["occupancy"] for b in bridges[:30]]
    fig.add_trace(go.Bar(
        x=labels, y=occupancies,
        marker_color='#74b9ff', name='Water Bridge Occupancy',
    ))
    return _dark_layout(fig, "Top Water-Mediated Bridges", "Residue Pair", "Occupancy")


def _plot_energy_decomposition(data):
    resids = data.get("resids", [])
    total = data.get("total_energy", [])
    if not resids or not total:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Per-Residue Total Interaction Energy", "VdW vs Electrostatic"))
    fig.add_trace(go.Bar(
        x=resids, y=total, marker_color='#e17055', name='Total',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=resids, y=data.get("vdw_energy", []),
        marker_color='#00cec9', name='VdW',
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=resids, y=data.get("elec_energy", []),
        marker_color='#fdcb6e', name='Electrostatic',
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Per-Residue Energy Decomposition", font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_prs(data):
    matrix = data.get("response_matrix", [])
    if not matrix:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "PRS Response Matrix", "Effector vs Sensor Scores"),
        column_widths=[0.6, 0.4])
    fig.add_trace(go.Heatmap(
        z=matrix, colorscale="Hot", colorbar=dict(title="Response", x=0.45),
    ), row=1, col=1)
    resids = data.get("resids", [])
    fig.add_trace(go.Scatter(
        x=resids, y=data.get("effector_scores", []),
        mode='lines', name='Effector', line=dict(color='#ff6b6b', width=1.5),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=resids, y=data.get("sensor_scores", []),
        mode='lines', name='Sensor', line=dict(color='#74b9ff', width=1.5),
    ), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Perturbation Response Scanning", font=dict(size=18, color="#E0E0E0")),
        height=500,
    )
    return fig


def _plot_nma(data):
    resids = data.get("resids", [])
    bfactors = data.get("bfactors", [])
    if not resids or not bfactors:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "ANM Predicted B-factors (Mobility)", "Mode Collectivity"))
    fig.add_trace(go.Bar(
        x=resids, y=bfactors,
        marker_color='#a29bfe', name='B-factor',
    ), row=1, col=1)
    collectivity = data.get("mode_collectivity", [])
    if collectivity:
        fig.add_trace(go.Bar(
            x=list(range(1, len(collectivity) + 1)), y=collectivity,
            marker_color='#55efc4', name='Collectivity',
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Normal Mode Analysis", font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_entropy(data):
    resids = data.get("resids", [])
    per_res = data.get("per_residue_entropy", [])
    if not resids or not per_res:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Per-Residue Configurational Entropy", "Entropy Convergence"))
    fig.add_trace(go.Bar(
        x=resids, y=per_res,
        marker_color='#fd79a8', name='Entropy',
    ), row=1, col=1)
    conv = data.get("entropy_convergence", [])
    if conv:
        fig.add_trace(go.Scatter(
            x=[c["fraction"] * 100 for c in conv],
            y=[c["entropy_J_mol_K"] for c in conv],
            mode='lines+markers', name='Convergence',
            line=dict(color='#ffeaa7', width=2),
            marker=dict(size=8),
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text=f"Entropy Estimation (Total: {data.get('total_entropy_kJ_mol_K', 0):.2f} kJ/mol/K)",
                   font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_ifp(data):
    resids = data.get("resids", [])
    consensus = data.get("consensus_fingerprint", [])
    if not resids or not consensus:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=resids, y=consensus,
        marker_color='#e84393', name='Interaction Density',
    ))
    return _dark_layout(fig, f"Interaction Fingerprint Consensus (Mean: {data.get('mean_interactions_per_frame', 0):.0f}/frame)",
                        "Residue ID", "Interaction Score")


def _plot_tunnels(data):
    volumes = data.get("cavity_volume_per_frame", [])
    times = data.get("time", [])
    if not volumes or not times:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Cavity Volume Over Time", "Bottleneck Residues"))
    fig.add_trace(go.Scatter(
        x=times, y=volumes, mode='lines', name='Volume',
        line=dict(color='#00b894', width=1.5),
        fill='tozeroy', fillcolor='rgba(0,184,148,0.1)',
    ), row=1, col=1)
    bottleneck = data.get("bottleneck_residues", [])
    if bottleneck:
        fig.add_trace(go.Bar(
            x=[b["resid"] for b in bottleneck],
            y=[b["cavity_frequency"] for b in bottleneck],
            marker_color='#00cec9', name='Cavity Lining Freq',
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text=f"Tunnel / Cavity Detection (Mean: {data.get('mean_cavity_volume', 0):.0f} ų)",
                   font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_vae(data):
    latent = data.get("latent_coords", [])
    if not latent or len(latent) == 0:
        return None
    arr = np.array(latent)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "VAE Latent Space", "Training Loss"))
    fig.add_trace(go.Scatter(
        x=arr[:, 0], y=arr[:, 1],
        mode='markers', name='Conformations',
        marker=dict(color=list(range(len(arr))), colorscale='Turbo',
                   size=4, colorbar=dict(title="Frame", x=0.45)),
    ), row=1, col=1)
    recon_loss = data.get("reconstruction_loss", [])
    kl_loss = data.get("kl_loss", [])
    if recon_loss:
        epochs = list(range(1, len(recon_loss) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=recon_loss, mode='lines', name='Recon Loss',
            line=dict(color='#ff6b6b', width=1.5),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=epochs, y=kl_loss, mode='lines', name='KL Loss',
            line=dict(color='#74b9ff', width=1.5),
        ), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Variational Autoencoder Latent Space", font=dict(size=18, color="#E0E0E0")),
    )
    return fig


def _plot_dynamic_network(data):
    comm_evo = data.get("community_evolution", [])
    resids = data.get("resids", [])
    if not comm_evo or not resids:
        return None
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Community Evolution Over Time Windows", "Community Stability Per Residue"))
    fig.add_trace(go.Heatmap(
        z=comm_evo, x=resids,
        y=[f"Window {i+1}" for i in range(len(comm_evo))],
        colorscale="Set3", colorbar=dict(title="Community", x=0.95),
    ), row=1, col=1)
    stability = data.get("community_stability", [])
    if stability:
        fig.add_trace(go.Bar(
            x=resids, y=stability,
            marker_color='#6c5ce7', name='Stability',
        ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Dynamic Network Analysis", font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Phase 4 — New Plot Generators
# ──────────────────────────────────────────────────────────────

def _plot_dimensionality_3d(data):
    """3D PCA/UMAP scatter plots (item 57)."""
    pca_3d = data.get("pca_3d", [])
    umap_3d = data.get("umap_3d", [])
    tsne_3d = data.get("tsne_3d", [])

    has_any = any(len(d) > 0 for d in [pca_3d, umap_3d, tsne_3d])
    if not has_any:
        return None

    from plotly.subplots import make_subplots
    n_cols = sum(1 for d in [pca_3d, umap_3d, tsne_3d] if len(d) > 0)
    if n_cols == 0:
        return None

    titles = []
    datasets = []
    if pca_3d:
        titles.append("PCA 3D")
        datasets.append(np.array(pca_3d))
    if umap_3d:
        titles.append("UMAP 3D")
        datasets.append(np.array(umap_3d))
    if tsne_3d:
        titles.append("t-SNE 3D")
        datasets.append(np.array(tsne_3d))

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=titles,
        specs=[[{"type": "scatter3d"}] * n_cols],
    )

    for idx, (arr, title) in enumerate(zip(datasets, titles)):
        fig.add_trace(go.Scatter3d(
            x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
            mode='markers', name=title,
            marker=dict(color=list(range(len(arr))), colorscale='Plasma',
                       size=2, opacity=0.7),
        ), row=1, col=idx + 1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e",
        font=dict(color="#E0E0E0"), margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text="3D Dimensionality Reduction", font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_convergence(data):
    """Convergence assessment plot (item 47)."""
    if not data:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Block Average SEM", "Autocorrelation (RMSD)",
        "Cosine Content by PC", f"Convergence Score: {data.get('convergence_score', 'N/A')}"))

    # Block average SEM
    blocks = data.get("rmsd_block_averages", [])
    if blocks:
        fig.add_trace(go.Scatter(
            x=[b["n_blocks"] for b in blocks],
            y=[b["sem"] for b in blocks],
            mode='lines+markers', name='RMSD SEM',
            line=dict(color='#00d4ff', width=2),
            marker=dict(size=8),
        ), row=1, col=1)

    # Autocorrelation
    acf = data.get("autocorrelation_rmsd", [])
    if acf:
        fig.add_trace(go.Scatter(
            x=[a["lag"] for a in acf],
            y=[a["acf"] for a in acf],
            mode='lines', name='ACF',
            line=dict(color='#a29bfe', width=2),
        ), row=1, col=2)
        fig.add_hline(y=0, row=1, col=2, line_dash="dot", line_color="#666")

    # Cosine content
    cc = data.get("cosine_content", [])
    if cc:
        fig.add_trace(go.Bar(
            x=[f"PC{c['pc']}" for c in cc],
            y=[c["cosine_content"] for c in cc],
            marker_color=['#55efc4' if c["converged"] else '#ff6b6b' for c in cc],
            name='Cosine Content',
        ), row=2, col=1)
        fig.add_hline(y=0.5, row=2, col=1, line_dash="dash", line_color="#ffd93d",
                     annotation_text="Threshold")

    # Recommendations
    recs = data.get("recommendations", [])
    if recs:
        text = "<br>".join(f"• {r}" for r in recs)
        fig.add_annotation(
            text=text, xref="x4 domain", yref="y4 domain",
            x=0.5, y=0.5, showarrow=False, font=dict(size=11, color="#E0E0E0"),
            align="left",
        )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Convergence Assessment", font=dict(size=18, color="#E0E0E0")),
        height=600, showlegend=False,
    )
    return fig


def _plot_binding_kinetics(data):
    """Binding kinetics plot (item 48)."""
    if not data:
        return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "COM Distance Over Time", "Contact Survival Function",
        "Per-Residue Contact Occupancy",
        f"kon={data.get('kon_estimate_per_ps', 0):.2e}/ps  koff={data.get('koff_estimate_per_ps', 0):.2e}/ps"))

    # COM distance
    times = data.get("time", [])
    com = data.get("com_distance", [])
    if times and com and len(times) == len(com):
        fig.add_trace(go.Scatter(
            x=times, y=com, mode='lines', name='COM dist',
            line=dict(color='#00d4ff', width=1.5),
        ), row=1, col=1)

    # Survival function
    survival = data.get("contact_survival", [])
    if survival:
        fig.add_trace(go.Scatter(
            x=[s["lag_frames"] for s in survival],
            y=[s["survival"] for s in survival],
            mode='lines', name='S(t)',
            line=dict(color='#fd79a8', width=2),
        ), row=1, col=2)

    # Per-residue contact occupancy
    res_contacts = data.get("per_residue_contact_time", [])
    if res_contacts:
        fig.add_trace(go.Bar(
            x=[r["resid"] for r in res_contacts[:30]],
            y=[r["occupancy"] for r in res_contacts[:30]],
            marker_color='#55efc4', name='Occupancy',
        ), row=2, col=1)

    # Binding events timeline
    events = data.get("binding_events", [])
    bind_times = [e["time_ps"] for e in events if e["type"] == "bind"]
    unbind_times = [e["time_ps"] for e in events if e["type"] == "unbind"]
    if bind_times:
        fig.add_trace(go.Scatter(
            x=bind_times, y=[1] * len(bind_times),
            mode='markers', name='Bind',
            marker=dict(color='#55efc4', size=8, symbol='triangle-up'),
        ), row=2, col=2)
    if unbind_times:
        fig.add_trace(go.Scatter(
            x=unbind_times, y=[0] * len(unbind_times),
            mode='markers', name='Unbind',
            marker=dict(color='#ff6b6b', size=8, symbol='triangle-down'),
        ), row=2, col=2)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=60, r=30, t=60, b=50),
        title=dict(text="Binding Kinetics Analysis", font=dict(size=18, color="#E0E0E0")),
        height=600,
    )
    return fig


def _plot_allosteric_network(data):
    """Interactive allosteric network graph (item 59)."""
    if not data:
        return None
    edges = data.get("network_edges", [])
    communities = data.get("communities", [])
    hubs = data.get("hub_residues", [])
    resids = data.get("resids", [])

    if not edges or not resids:
        return None

    import networkx as nx

    # Build networkx graph for layout
    G = nx.Graph()
    for r in resids:
        G.add_node(r)
    for e in edges[:200]:
        G.add_edge(e["res_i"], e["res_j"], weight=abs(e["correlation"]))

    # Spring layout
    try:
        pos = nx.spring_layout(G, k=2.0/np.sqrt(len(resids)), iterations=50, seed=42)
    except Exception:
        return None

    # Color by community
    comm_color = {}
    colors_palette = ['#55efc4', '#a29bfe', '#ff6b6b', '#ffd93d', '#fd79a8',
                      '#00d4ff', '#e17055', '#74b9ff', '#00cec9', '#6c5ce7']
    for comm in communities:
        c_idx = comm["id"] % len(colors_palette)
        for r in comm["residues"]:
            comm_color[r] = colors_palette[c_idx]

    hub_resids = {h["resid"] for h in hubs}

    # Edge traces
    edge_x, edge_y = [], []
    for e in edges[:200]:
        if e["res_i"] in pos and e["res_j"] in pos:
            x0, y0 = pos[e["res_i"]]
            x1, y1 = pos[e["res_j"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(100,100,150,0.3)', width=0.5),
        hoverinfo='none', name='Edges',
    ))

    # Node traces
    node_x = [pos[r][0] for r in resids if r in pos]
    node_y = [pos[r][1] for r in resids if r in pos]
    node_color = [comm_color.get(r, '#666680') for r in resids if r in pos]
    node_size = [12 if r in hub_resids else 5 for r in resids if r in pos]
    node_text = [f"Res {r}" + (" (HUB)" if r in hub_resids else "") for r in resids if r in pos]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(color=node_color, size=node_size,
                   line=dict(width=1, color='rgba(255,255,255,0.3)')),
        text=node_text, hoverinfo='text', name='Residues',
    ))

    modularity = data.get("modularity_score", 0)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#E0E0E0"), margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f"Allosteric Network (Modularity: {modularity:.3f}, {len(edges)} edges)",
                   font=dict(size=18, color="#E0E0E0")),
        showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    return fig


def _plot_training_losses(data):
    """Combined training loss curves for GNN, Transformer, VAE (item 53)."""
    if not data:
        return None

    gnn = data.get("gnn", {})
    transformer = data.get("transformer", {})
    vae = data.get("vae", {})

    has_any = False
    fig = go.Figure()

    gnn_losses = gnn.get("training_losses", []) if isinstance(gnn, dict) else []
    trans_losses = transformer.get("training_losses", []) if isinstance(transformer, dict) else []
    vae_losses = vae.get("total_loss", []) if isinstance(vae, dict) else []

    if gnn_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(gnn_losses) + 1)), y=gnn_losses,
            mode='lines', name='GNN',
            line=dict(color='#e17055', width=2),
        ))
        has_any = True

    if trans_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(trans_losses) + 1)), y=trans_losses,
            mode='lines', name='Transformer',
            line=dict(color='#00cec9', width=2),
        ))
        has_any = True

    if vae_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vae_losses) + 1)), y=vae_losses,
            mode='lines', name='VAE',
            line=dict(color='#a29bfe', width=2),
        ))
        has_any = True

    if not has_any:
        return None

    return _dark_layout(fig, "Deep Learning Training Loss Curves", "Epoch", "Loss")

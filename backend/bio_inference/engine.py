"""
Biological Inference Engine.

Converts raw computational metrics into biologically meaningful explanations
using structural biology reasoning.  Each detector method returns a list of
insight dicts with schema::

    {
        "type": str,
        "residues": list[int],
        "description": str,
        "confidence": float,
        "evidence": list[str],
        "category": str,   # structural | dynamic | allosteric | binding | transition
    }
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

logger = logging.getLogger("md_ai_analyzer")


class BiologicalInferenceEngine:
    """Produce human-readable biological interpretations from analysis results.

    All detector methods are called via :meth:`interpret`, which catches and
    logs exceptions per-detector so that a failure in one does not prevent
    the others from running.
    """

    # Ordered list of detector method names.  New detectors should be
    # appended here rather than adding another try/except block.
    _DETECTORS: list[str] = [
        # Core structural / dynamic
        "_detect_hinge_residues",
        "_detect_flexible_loops",
        "_detect_stable_core",
        "_detect_allosteric_communication",
        "_detect_binding_pocket_dynamics",
        "_detect_conformational_transitions",
        "_detect_domain_motions",
        "_assess_protein_stability",
        # ML / DL interpretation
        "_interpret_gnn_results",
        "_interpret_transformer_results",
        # Part A — per-module interpretation
        "_interpret_water_bridges",
        "_interpret_energy_hotspots",
        "_interpret_prs",
        "_interpret_nma",
        "_interpret_entropy",
        "_interpret_tunnels",
        "_interpret_dynamic_network",
        "_interpret_vae",
        "_interpret_ifp",
        # Part B — advanced biological inferences
        "_detect_breathing_motions",
        "_detect_cracking_events",
        "_detect_cryptic_binding_sites",
        "_score_druggability",
        "_predict_ptm_sites",
        "_detect_ppi_hotspots",
        "_detect_interface_dynamics",
        "_infer_protonation_dynamics",
        "_detect_electrostatic_funnels",
        "_detect_aggregation_prone_regions",
        "_detect_folding_intermediates",
        "_classify_functional_motions",
        "_correlate_motions_to_function",
        "_detect_hbond_network_rewiring",
        "_identify_structural_waters",
        "_map_local_stiffness",
        "_detect_force_propagation",
        "_predict_mutation_sensitivity",
        "_predict_stability_changes",
    ]

    def interpret(self, result: Any) -> List[Dict[str, Any]]:
        """Generate biological insights from all analysis results.

        Parameters
        ----------
        result : AnalysisResult
            The fully populated analysis result object.

        Returns
        -------
        list[dict]
            Insight dicts sorted by descending confidence.
        """
        insights: List[Dict[str, Any]] = []

        for method_name in self._DETECTORS:
            method: Callable = getattr(self, method_name, None)  # type: ignore[assignment]
            if method is None:
                logger.warning("Detector method not found: %s", method_name)
                continue
            try:
                new_insights = method(result)
                if new_insights:
                    insights.extend(new_insights)
            except Exception:
                logger.debug(
                    "Detector '%s' raised an exception", method_name,
                    exc_info=True,
                )

        # Sort by confidence descending
        insights.sort(key=lambda x: -x.get("confidence", 0))
        return insights

    def _detect_hinge_residues(self, result) -> List[Dict]:
        """
        Hinge residues: high RMSF flanked by low-RMSF domains,
        high betweenness centrality in the interaction network.
        """
        insights = []
        rmsf_data = result.rmsf
        allosteric_data = result.allosteric

        if isinstance(rmsf_data, dict) and "rmsf" in rmsf_data and not rmsf_data.get("error"):
            rmsf = np.array(rmsf_data["rmsf"])
            resids = rmsf_data.get("resids", list(range(len(rmsf))))
            mean_rmsf = np.mean(rmsf)
            std_rmsf = np.std(rmsf)

            # A hinge: high RMSF residue flanked by low-RMSF regions on both sides
            for i in range(3, len(rmsf) - 3):
                left_avg = np.mean(rmsf[max(0, i-3):i])
                right_avg = np.mean(rmsf[i+1:min(len(rmsf), i+4)])
                center = rmsf[i]

                if (center > mean_rmsf + 0.5 * std_rmsf and
                    left_avg < mean_rmsf - 0.2 * std_rmsf and
                    right_avg < mean_rmsf - 0.2 * std_rmsf):

                    confidence = min(0.95, 0.5 + 0.3 * (center - mean_rmsf) / (std_rmsf + 1e-8))
                    evidence = [
                        f"Residue {resids[i]} has RMSF={rmsf[i]:.2f}Å (mean={mean_rmsf:.2f}Å)",
                        f"Flanking regions have low RMSF: left={left_avg:.2f}Å, right={right_avg:.2f}Å",
                    ]

                    # Check if also a network hub
                    if isinstance(allosteric_data, dict) and "hub_residues" in allosteric_data:
                        hub_resids = [h["resid"] for h in allosteric_data["hub_residues"]]
                        if resids[i] in hub_resids:
                            confidence = min(0.98, confidence + 0.15)
                            evidence.append(f"Residue {resids[i]} is also a network hub (high betweenness centrality)")

                    insights.append({
                        "type": "hinge_residue",
                        "residues": [int(resids[i])],
                        "description": f"Residue {resids[i]} acts as a hinge point between two rigid domains. "
                                      f"It shows elevated flexibility (RMSF={rmsf[i]:.2f}Å) while neighboring "
                                      f"regions remain rigid, characteristic of a mechanical hinge enabling domain motion.",
                        "confidence": round(confidence, 2),
                        "evidence": evidence,
                        "category": "structural",
                    })

        return insights[:5]

    def _detect_flexible_loops(self, result) -> List[Dict]:
        """Identify flexible loops: contiguous high-RMSF segments with coil SS."""
        insights = []
        rmsf_data = result.rmsf
        ss_data = result.secondary_structure

        if isinstance(rmsf_data, dict) and "flexible_segments" in rmsf_data:
            segments = rmsf_data["flexible_segments"]
            for seg in segments[:5]:
                start, end = seg["start"], seg["end"]
                length = seg["length"]

                evidence = [f"Contiguous flexible segment: residues {start}-{end} ({length} residues)"]

                # Check SS
                is_coil = True
                if isinstance(ss_data, dict) and "per_residue_dominant_ss" in ss_data:
                    ss_resids = ss_data.get("resids", [])
                    dom_ss = ss_data.get("per_residue_dominant_ss", [])
                    coil_count = 0
                    total = 0
                    for r in range(start, end + 1):
                        if r in ss_resids:
                            idx = ss_resids.index(r)
                            if idx < len(dom_ss):
                                total += 1
                                if dom_ss[idx] == 'C':
                                    coil_count += 1
                    if total > 0:
                        coil_frac = coil_count / total
                        is_coil = coil_frac > 0.5
                        evidence.append(f"Coil content: {coil_frac*100:.0f}%")

                insights.append({
                    "type": "flexible_loop",
                    "residues": list(range(start, end + 1)),
                    "description": f"Flexible loop spanning residues {start}-{end}. "
                                  f"This {length}-residue segment shows elevated dynamics, "
                                  f"suggesting it may be involved in substrate recognition, "
                                  f"protein-protein interaction, or conformational sampling.",
                    "confidence": round(min(0.9, 0.6 + 0.05 * length), 2),
                    "evidence": evidence,
                    "category": "structural",
                })

        return insights

    def _detect_stable_core(self, result) -> List[Dict]:
        """Identify the structural core: low RMSF + high contacts + regular SS."""
        insights = []
        rmsf_data = result.rmsf
        contact_data = result.contacts

        if isinstance(rmsf_data, dict) and "low_flexibility_residues" in rmsf_data:
            low_flex = rmsf_data["low_flexibility_residues"]
            if len(low_flex) > 5:
                evidence = [f"{len(low_flex)} residues with below-average flexibility"]

                if isinstance(contact_data, dict) and "persistent_contacts" in contact_data:
                    n_persistent = len(contact_data["persistent_contacts"])
                    evidence.append(f"{n_persistent} persistent inter-residue contacts maintain structural integrity")

                insights.append({
                    "type": "stable_core",
                    "residues": low_flex[:30],
                    "description": f"Identified a stable structural core comprising {len(low_flex)} residues. "
                                  f"These residues show minimal fluctuation and high packing density, "
                                  f"forming the backbone of the protein's fold stability. "
                                  f"Mutations in these regions are likely destabilizing.",
                    "confidence": 0.85,
                    "evidence": evidence,
                    "category": "structural",
                })

        return insights

    def _detect_allosteric_communication(self, result) -> List[Dict]:
        """Detect allosteric communication pathways from network analysis."""
        insights = []
        allosteric_data = result.allosteric

        if isinstance(allosteric_data, dict) and "shortest_paths" in allosteric_data:
            paths = allosteric_data["shortest_paths"]
            for path_info in paths[:3]:
                from_r = path_info["from_resid"]
                to_r = path_info["to_resid"]
                pathway = path_info["path"]
                corr = path_info["correlation"]

                insights.append({
                    "type": "allosteric_pathway",
                    "residues": pathway,
                    "description": f"Allosteric communication pathway detected from residue {from_r} to "
                                  f"residue {to_r} (correlation={corr:.2f}). The signal propagates through "
                                  f"{len(pathway)} intermediate residues: {' → '.join(map(str, pathway))}. "
                                  f"This pathway may transmit conformational changes across the protein, "
                                  f"suggesting functional coupling between these distant sites.",
                    "confidence": round(min(0.9, abs(corr)), 2),
                    "evidence": [
                        f"Strong dynamic correlation (r={corr:.2f}) between sequence-distant residues",
                        f"Shortest communication path through residue interaction network: {len(pathway)} nodes",
                    ],
                    "category": "allosteric",
                })

            # Hub residues
            if "hub_residues" in allosteric_data and allosteric_data["hub_residues"]:
                hub_list = allosteric_data["hub_residues"][:5]
                hub_resids = [h["resid"] for h in hub_list]
                insights.append({
                    "type": "communication_hub",
                    "residues": hub_resids,
                    "description": f"Residues {', '.join(map(str, hub_resids))} serve as communication hubs "
                                  f"in the protein's dynamic interaction network. They have high betweenness "
                                  f"centrality, meaning many allosteric signal pathways pass through them. "
                                  f"These residues are critical for long-range conformational signaling.",
                    "confidence": 0.8,
                    "evidence": [f"Hub {h['resid']}: betweenness={h['betweenness']:.4f}" for h in hub_list],
                    "category": "allosteric",
                })

        return insights

    def _detect_binding_pocket_dynamics(self, result) -> List[Dict]:
        """Analyze ligand binding pocket dynamics."""
        insights = []
        ligand_data = result.ligand

        if isinstance(ligand_data, dict) and "key_binding_residues" in ligand_data:
            key_res = ligand_data["key_binding_residues"]
            stability = ligand_data.get("binding_stability", "unknown")

            if key_res:
                resids = [r["resid"] for r in key_res]
                insights.append({
                    "type": "binding_pocket",
                    "residues": resids,
                    "description": f"Key ligand-binding residues identified: "
                                  f"{', '.join(r['resname'] + str(r['resid']) for r in key_res[:10])}. "
                                  f"These residues maintain >50% contact frequency with the ligand. "
                                  f"Binding stability assessment: {stability}. "
                                  f"{'The stable binding suggests a well-defined binding mode.' if stability == 'stable' else 'Dynamic binding suggests the ligand samples multiple poses.'}",
                    "confidence": 0.85,
                    "evidence": [f"{r['resname']}{r['resid']}: {r['frequency']*100:.0f}% contact" for r in key_res[:10]],
                    "category": "binding",
                })

        return insights

    def _detect_conformational_transitions(self, result) -> List[Dict]:
        """Detect conformational state transitions from clustering and MSM."""
        insights = []
        cluster_data = result.clustering
        msm_data = result.msm

        if isinstance(cluster_data, dict) and "n_clusters" in cluster_data:
            n_clusters = cluster_data["n_clusters"]
            populations = cluster_data.get("populations", {})
            n_transitions = cluster_data.get("n_transitions", 0)

            if n_clusters > 1:
                pop_str = ", ".join([f"State {k}: {v*100:.1f}%" for k, v in sorted(populations.items())])
                insights.append({
                    "type": "conformational_states",
                    "residues": [],
                    "description": f"The protein samples {n_clusters} distinct conformational states "
                                  f"during the simulation. State populations: {pop_str}. "
                                  f"{n_transitions} inter-state transitions were observed. "
                                  f"{'Multiple transitions suggest reversible conformational dynamics.' if n_transitions > 5 else 'Few transitions suggest the protein is kinetically trapped or the states are well-separated.'}",
                    "confidence": round(min(0.9, cluster_data.get("silhouette_score", 0.5) + 0.3), 2),
                    "evidence": [
                        f"Silhouette score: {cluster_data.get('silhouette_score', 'N/A')}",
                        f"{n_transitions} state transitions detected",
                    ],
                    "category": "transition",
                })

        if isinstance(msm_data, dict) and "metastable_states" in msm_data:
            meta_states = msm_data["metastable_states"]
            timescales = msm_data.get("implied_timescales", [])

            if meta_states and timescales:
                most_stable = meta_states[0]
                insights.append({
                    "type": "metastable_kinetics",
                    "residues": [],
                    "description": f"Markov State Model identifies {len(meta_states)} kinetic states. "
                                  f"The most metastable state (self-transition probability: "
                                  f"{most_stable['self_transition']:.2f}) has population "
                                  f"{most_stable['population']*100:.1f}%. "
                                  f"Slowest implied timescale: {timescales[0]:.1f} frames, "
                                  f"indicating the dominant relaxation process.",
                    "confidence": 0.75,
                    "evidence": [
                        f"State {s['state']}: P_self={s['self_transition']:.2f}, pop={s['population']*100:.1f}%"
                        for s in meta_states[:5]
                    ],
                    "category": "transition",
                })

        return insights

    def _detect_domain_motions(self, result) -> List[Dict]:
        """Interpret dynamic domain detection results."""
        insights = []
        domain_data = result.domains

        if isinstance(domain_data, dict) and "domain_info" in domain_data:
            domains = domain_data["domain_info"]
            inter_corr = domain_data.get("inter_domain_correlations", [])

            if len(domains) >= 2:
                domain_desc = []
                for d in domains:
                    domain_desc.append(
                        f"Domain {d['domain_id']} (residues {d['start_resid']}-{d['end_resid']}, "
                        f"size={d['size']}, internal corr={d['mean_internal_correlation']:.2f})"
                    )

                insights.append({
                    "type": "domain_motion",
                    "residues": [],
                    "description": f"Spectral clustering identifies {len(domains)} dynamic domains: "
                                  f"{'; '.join(domain_desc)}. Residues within each domain move as "
                                  f"quasi-rigid bodies, while inter-domain motions represent the "
                                  f"functional dynamics of the protein.",
                    "confidence": 0.8,
                    "evidence": [d_str for d_str in domain_desc],
                    "category": "dynamic",
                })

        return insights

    def _assess_protein_stability(self, result) -> List[Dict]:
        """Assess overall protein stability from multiple metrics."""
        insights = []
        rmsd_data = result.rmsd
        rg_data = result.rg
        hbond_data = result.hbonds

        evidence = []
        stability_score = 0.5

        if isinstance(rmsd_data, dict) and "mean_rmsd" in rmsd_data:
            mean_rmsd = rmsd_data["mean_rmsd"]
            std_rmsd = rmsd_data["std_rmsd"]
            equil = rmsd_data.get("equilibration_frame", 0)

            if mean_rmsd < 2.0 and std_rmsd < 0.5:
                stability_score += 0.2
                evidence.append(f"Low RMSD (mean={mean_rmsd:.2f}Å, std={std_rmsd:.2f}Å) indicates structural stability")
            elif mean_rmsd > 4.0:
                stability_score -= 0.2
                evidence.append(f"High RMSD (mean={mean_rmsd:.2f}Å) suggests significant structural changes")

        if isinstance(rg_data, dict) and "trend" in rg_data:
            trend = rg_data["trend"]
            if trend == "stable":
                stability_score += 0.1
                evidence.append(f"Stable radius of gyration indicates maintained compactness")
            elif trend == "expanding":
                stability_score -= 0.15
                evidence.append(f"Expanding Rg suggests partial unfolding")

        if isinstance(hbond_data, dict) and "mean_hbonds" in hbond_data:
            n_persistent = len(hbond_data.get("persistent_hbonds", []))
            evidence.append(f"{n_persistent} persistent hydrogen bonds maintain structural integrity")

        if evidence:
            assessment = "highly stable" if stability_score > 0.7 else \
                         "moderately stable" if stability_score > 0.5 else \
                         "showing dynamic fluctuations" if stability_score > 0.3 else \
                         "potentially unstable"
            insights.append({
                "type": "stability_assessment",
                "residues": [],
                "description": f"Overall structural stability assessment: the protein is {assessment} "
                              f"during the simulation. {' '.join(evidence)}.",
                "confidence": round(min(0.9, max(0.4, stability_score)), 2),
                "evidence": evidence,
                "category": "structural",
            })

        return insights

    def _interpret_gnn_results(self, result) -> List[Dict]:
        """Interpret GNN-identified residue importance."""
        insights = []
        gnn = result.gnn_results

        if isinstance(gnn, dict) and "top_residues" in gnn:
            top = gnn["top_residues"][:10]
            if top:
                resids = [r["resid"] for r in top]
                insights.append({
                    "type": "gnn_key_residues",
                    "residues": resids,
                    "description": f"Graph Neural Network analysis identifies residues "
                                  f"{', '.join(map(str, resids[:5]))} as structurally most important based on "
                                  f"their graph connectivity patterns and dynamic properties. These residues "
                                  f"are predicted to be critical for maintaining protein structure and function.",
                    "confidence": 0.7,
                    "evidence": [
                        f"Res {r['resid']}: GNN importance={r['importance']:.3f}, RMSF={r['rmsf']:.2f}Å, contacts={r['contacts']:.1f}"
                        for r in top[:5]
                    ],
                    "category": "dynamic",
                })

        return insights

    def _interpret_transformer_results(self, result) -> List[Dict]:
        """Interpret Transformer-detected transitions."""
        insights = []
        trans = result.transformer_results

        if isinstance(trans, dict) and "transition_frames" in trans:
            transitions = trans["transition_frames"]
            if transitions:
                insights.append({
                    "type": "transformer_transitions",
                    "residues": [],
                    "description": f"Transformer temporal analysis detects {len(transitions)} significant "
                                  f"structural transition events in the trajectory. The most prominent occurs "
                                  f"at frame {transitions[0]['frame']} (magnitude={transitions[0]['magnitude']:.3f}). "
                                  f"These transitions represent major conformational rearrangements detected by the "
                                  f"self-attention mechanism of the neural network.",
                    "confidence": 0.65,
                    "evidence": [
                        f"Frame {t['frame']}: transition magnitude={t['magnitude']:.3f}"
                        for t in transitions[:5]
                    ],
                    "category": "transition",
                })

        return insights

    # ──────────────────────────────────────────────────────────────
    # Part A — New Interpretation Methods
    # ──────────────────────────────────────────────────────────────

    def _interpret_water_bridges(self, result) -> List[Dict]:
        """Interpret water-mediated contact sites."""
        insights = []
        wb = result.water_bridges
        if isinstance(wb, dict) and not wb.get("error"):
            bridges = wb.get("bridges", [])
            if bridges:
                top = bridges[:5]
                pairs = [f"{b['resid_1']}-{b['resid_2']} ({b['occupancy']*100:.0f}%)" for b in top]
                resids = []
                for b in top:
                    resids.extend([b["resid_1"], b["resid_2"]])
                insights.append({
                    "type": "water_bridge_sites",
                    "residues": list(set(resids)),
                    "description": f"Identified {len(bridges)} persistent water-mediated bridges. "
                                  f"Top sites: {', '.join(pairs)}. Water bridges are critical for "
                                  f"stabilizing protein-protein interfaces and can contribute "
                                  f"significantly to binding free energy.",
                    "confidence": 0.75,
                    "evidence": [f"Bridge {b['resid_1']}-{b['resid_2']}: {b['occupancy']*100:.0f}% occupancy" for b in top],
                    "category": "structural",
                })
        return insights

    def _interpret_energy_hotspots(self, result) -> List[Dict]:
        """Interpret energetically important residues."""
        insights = []
        ed = result.energy_decomposition
        if isinstance(ed, dict) and not ed.get("error"):
            top_pairs = ed.get("top_pairs", [])
            if top_pairs:
                strong = top_pairs[:5]
                pair_strs = [f"{p['resid_i']}-{p['resid_j']} ({p['energy_kj']:.1f} kJ/mol)" for p in strong]
                resids = []
                for p in strong:
                    resids.extend([p["resid_i"], p["resid_j"]])
                insights.append({
                    "type": "energy_hotspot",
                    "residues": list(set(resids)),
                    "description": f"Energy decomposition identifies key interaction hotspots: "
                                  f"{', '.join(pair_strs)}. These residue pairs contribute most "
                                  f"to the protein's non-bonded energy, making them potential targets "
                                  f"for stability engineering or drug design.",
                    "confidence": 0.7,
                    "evidence": [f"Pair {p['resid_i']}-{p['resid_j']}: {p['energy_kj']:.1f} kJ/mol" for p in strong],
                    "category": "structural",
                })
        return insights

    def _interpret_prs(self, result) -> List[Dict]:
        """Interpret perturbation response effector/sensor roles."""
        insights = []
        prs = result.prs
        if isinstance(prs, dict) and not prs.get("error"):
            effectors = prs.get("top_effectors", [])
            sensors = prs.get("top_sensors", [])
            if effectors and sensors:
                eff_resids = [e["resid"] for e in effectors[:5]]
                sens_resids = [s["resid"] for s in sensors[:5]]
                insights.append({
                    "type": "prs_effectors_sensors",
                    "residues": list(set(eff_resids + sens_resids)),
                    "description": f"PRS identifies key effector residues {', '.join(map(str, eff_resids))} "
                                  f"(perturbing these propagates displacement throughout the protein) and "
                                  f"sensor residues {', '.join(map(str, sens_resids))} (most responsive to "
                                  f"perturbation). Effectors are prime mutation candidates; sensors are "
                                  f"likely functional sites sensitive to allosteric regulation.",
                    "confidence": 0.75,
                    "evidence": [f"Effector {e['resid']}: score={e['score']:.3f}" for e in effectors[:3]]
                              + [f"Sensor {s['resid']}: score={s['score']:.3f}" for s in sensors[:3]],
                    "category": "allosteric",
                })
        return insights

    def _interpret_nma(self, result) -> List[Dict]:
        """Interpret dominant collective motions from NMA."""
        insights = []
        nma = result.nma
        if isinstance(nma, dict) and not nma.get("error"):
            collectivity = nma.get("mode_collectivity", [])
            if collectivity:
                max_coll = max(collectivity)
                max_idx = collectivity.index(max_coll)
                insights.append({
                    "type": "nma_collective_motion",
                    "residues": [],
                    "description": f"Normal mode analysis reveals {nma.get('n_modes_computed', 0)} collective modes. "
                                  f"Mode {max_idx+1} is the most collective (κ={max_coll:.3f}), involving "
                                  f"coordinated motion of ~{max_coll*100:.0f}% of residues. "
                                  f"{'High collectivity indicates global domain motions (hinge-bending, clamshell closure).' if max_coll > 0.3 else 'Moderate collectivity suggests localised deformations.'}",
                    "confidence": 0.7,
                    "evidence": [f"Mode {i+1}: collectivity κ={c:.3f}" for i, c in enumerate(collectivity[:5])],
                    "category": "dynamic",
                })
        return insights

    def _interpret_entropy(self, result) -> List[Dict]:
        """Interpret configurational entropy estimation."""
        insights = []
        ent = result.entropy
        if isinstance(ent, dict) and not ent.get("error"):
            total = ent.get("total_entropy_kJ_mol_K", 0)
            convergence = ent.get("entropy_convergence", [])
            if total and convergence:
                converged = False
                if len(convergence) >= 2:
                    last = convergence[-1]["entropy_J_mol_K"]
                    prev = convergence[-2]["entropy_J_mol_K"]
                    if prev > 0:
                        change = abs(last - prev) / prev
                        converged = change < 0.05
                insights.append({
                    "type": "entropy_estimate",
                    "residues": [],
                    "description": f"Configurational entropy estimated at {total:.2f} kJ/(mol·K) using "
                                  f"Schlitter's method. {'The estimate is well-converged.' if converged else 'The estimate has not fully converged; longer simulation may be needed.'} "
                                  f"High-entropy residues represent conformationally flexible regions that "
                                  f"may contribute to binding entropy penalties or entropic stabilization.",
                    "confidence": 0.65 if converged else 0.5,
                    "evidence": [f"{c['fraction']*100:.0f}% trajectory: S={c['entropy_J_mol_K']:.0f} J/(mol·K)" for c in convergence],
                    "category": "dynamic",
                })
        return insights

    def _interpret_tunnels(self, result) -> List[Dict]:
        """Interpret tunnel/cavity detection results."""
        insights = []
        tun = result.tunnels
        if isinstance(tun, dict) and not tun.get("error"):
            bottleneck = tun.get("bottleneck_residues", [])
            mean_vol = tun.get("mean_cavity_volume", 0)
            if bottleneck and mean_vol > 0:
                bn_resids = [b["resid"] for b in bottleneck[:5]]
                insights.append({
                    "type": "cavity_channels",
                    "residues": bn_resids,
                    "description": f"Cavity analysis detects an average internal volume of {mean_vol:.0f} ų. "
                                  f"Key bottleneck residues lining the cavities: {', '.join(map(str, bn_resids))}. "
                                  f"These residues gate access to internal tunnels or binding pockets "
                                  f"and are prime targets for mutagenesis to modulate substrate access.",
                    "confidence": 0.7,
                    "evidence": [f"Residue {b['resid']}: cavity-lining frequency={b['cavity_frequency']*100:.0f}%" for b in bottleneck[:5]],
                    "category": "structural",
                })
        return insights

    def _interpret_dynamic_network(self, result) -> List[Dict]:
        """Interpret time-resolved network community evolution."""
        insights = []
        dn = result.dynamic_network
        if isinstance(dn, dict) and not dn.get("error"):
            stability = dn.get("community_stability", [])
            resids = dn.get("resids", [])
            if stability and resids:
                stability_arr = np.array(stability)
                mean_stab = float(stability_arr.mean())
                # Identify residues that switch communities frequently
                unstable_idx = np.where(stability_arr < 0.5)[0]
                unstable_resids = [int(resids[i]) for i in unstable_idx[:10]]
                hub_evo = dn.get("hub_evolution", [])
                persistent_hubs = []
                if len(hub_evo) >= 2:
                    hub_sets = [set(h["resid"] for h in window) for window in hub_evo if window]
                    if hub_sets:
                        persistent_hubs = list(set.intersection(*hub_sets)) if len(hub_sets) > 1 else []

                evidence = [f"Mean community stability: {mean_stab:.2f}"]
                if unstable_resids:
                    evidence.append(f"Community-switching residues: {', '.join(map(str, unstable_resids))}")
                if persistent_hubs:
                    evidence.append(f"Persistent hub residues across all windows: {', '.join(map(str, persistent_hubs[:5]))}")

                insights.append({
                    "type": "dynamic_network_evolution",
                    "residues": unstable_resids + persistent_hubs[:5],
                    "description": f"Dynamic network analysis over {dn.get('n_windows', '?')} time windows "
                                  f"shows mean community stability of {mean_stab:.2f}. "
                                  f"{'Residues ' + ', '.join(map(str, unstable_resids[:5])) + ' frequently switch communities, suggesting they lie at domain interfaces or allosteric pathways. ' if unstable_resids else ''}"
                                  f"{'Persistent hubs (' + ', '.join(map(str, persistent_hubs[:5])) + ') maintain high betweenness centrality throughout, indicating robust allosteric conduits.' if persistent_hubs else ''}",
                    "confidence": 0.7,
                    "evidence": evidence,
                    "category": "allosteric",
                })
        return insights

    def _interpret_vae(self, result) -> List[Dict]:
        """Interpret VAE latent space structure."""
        insights = []
        vae = result.vae
        if isinstance(vae, dict) and not vae.get("error"):
            recon_error = vae.get("reconstruction_error", None)
            latent_var = vae.get("latent_variance", [])
            latent_dim = vae.get("latent_dim", 2)
            if recon_error is not None:
                evidence = [f"Reconstruction MSE: {recon_error:.4f}"]
                if latent_var:
                    evidence.append(f"Latent variance per dim: {', '.join(f'{v:.3f}' for v in latent_var)}")

                quality = "good" if recon_error < 0.5 else "moderate" if recon_error < 1.0 else "poor"
                insights.append({
                    "type": "vae_conformational_landscape",
                    "residues": [],
                    "description": f"VAE with {latent_dim}D latent space achieves {quality} reconstruction "
                                  f"(MSE={recon_error:.4f}). "
                                  f"{'Well-separated clusters in latent space suggest distinct conformational states. ' if latent_var and max(latent_var) > 0.5 else ''}"
                                  f"The learned latent representation captures the dominant conformational "
                                  f"degrees of freedom in a continuous, differentiable manifold.",
                    "confidence": 0.65 if quality == "good" else 0.5,
                    "evidence": evidence,
                    "category": "transition",
                })
        return insights

    def _interpret_ifp(self, result) -> List[Dict]:
        """Interpret interaction fingerprint patterns."""
        insights = []
        ifp = result.interaction_fingerprints
        if isinstance(ifp, dict) and not ifp.get("error"):
            top = ifp.get("top_interactions", [])
            if top:
                salt_pairs = [t for t in top if t.get("salt_bridge", 0) > 0.3]
                hydro_pairs = [t for t in top if t.get("hydrophobic", 0) > 0.3]
                aro_pairs = [t for t in top if t.get("aromatic", 0) > 0.3]

                resids = []
                evidence = []
                for t in top[:5]:
                    resids.extend([t["resid_1"], t["resid_2"]])
                    evidence.append(
                        f"{t['resid_1']}-{t['resid_2']}: total={t['total_occupancy']:.2f}, "
                        f"hydrophobic={t['hydrophobic']:.2f}, salt={t['salt_bridge']:.2f}, "
                        f"aromatic={t['aromatic']:.2f}"
                    )

                desc_parts = [f"Interaction fingerprint analysis identified {len(top)} significant residue pair interactions."]
                if salt_pairs:
                    desc_parts.append(f"{len(salt_pairs)} persistent salt bridges provide electrostatic stabilization.")
                if hydro_pairs:
                    desc_parts.append(f"{len(hydro_pairs)} hydrophobic contact pairs contribute to the hydrophobic core.")
                if aro_pairs:
                    desc_parts.append(f"{len(aro_pairs)} aromatic stacking interactions detected.")

                insights.append({
                    "type": "interaction_fingerprint",
                    "residues": list(set(resids)),
                    "description": " ".join(desc_parts),
                    "confidence": 0.7,
                    "evidence": evidence,
                    "category": "structural",
                })
        return insights

    # ──────────────────────────────────────────────────────────────
    # Part B — Advanced Biological Inferences
    # ──────────────────────────────────────────────────────────────

    # Amino acid property lookup tables used by several methods below.
    _HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"}
    _AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
    _CHARGED_POSITIVE = {"ARG", "LYS"}
    _CHARGED_NEGATIVE = {"ASP", "GLU"}
    _TITRATABLE_RESIDUES = {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"}
    _PTM_PHOSPHO_RESIDUES = {"SER", "THR", "TYR"}
    _PTM_GLYCO_RESIDUES = {"ASN"}  # N-linked glycosylation Asn-X-Ser/Thr motif
    _PTM_UBIQ_RESIDUES = {"LYS"}

    # Residue-level aggregation propensity scale (Pawar et al., simplified)
    _AGGREGATION_PROPENSITY = {
        "ILE": 0.88, "VAL": 0.86, "LEU": 0.82, "PHE": 0.81, "TYR": 0.65,
        "TRP": 0.60, "MET": 0.56, "ALA": 0.42, "GLY": 0.10, "THR": 0.08,
        "SER": 0.06, "CYS": 0.30, "PRO": -0.30, "GLN": -0.10, "ASN": -0.10,
        "HIS": -0.12, "ASP": -0.50, "GLU": -0.55, "LYS": -0.60, "ARG": -0.65,
    }

    def _detect_breathing_motions(self, result) -> List[Dict]:
        """Detect periodic Rg oscillations or PC1 oscillations indicating breathing/gating."""
        insights = []
        rg_data = result.rg
        sasa_data = result.sasa
        pca_data = result.pca

        # Strategy: look for periodic oscillations in Rg or PC1 projection
        signal = None
        signal_label = None

        if isinstance(rg_data, dict) and "rg" in rg_data and not rg_data.get("error"):
            rg_series = np.array(rg_data["rg"])
            if len(rg_series) > 50:
                signal = rg_series
                signal_label = "Rg"

        if signal is None and isinstance(pca_data, dict) and "projections" in pca_data:
            proj = pca_data["projections"]
            if isinstance(proj, list) and len(proj) > 50:
                signal = np.array([p[0] for p in proj])
                signal_label = "PC1"

        if signal is not None and len(signal) > 50:
            # Detrend
            kernel = min(20, len(signal) // 5)
            if kernel < 1:
                kernel = 1
            signal_detrended = signal - np.convolve(signal, np.ones(kernel) / kernel, mode='same')
            # Autocorrelation to detect periodicity
            n = len(signal_detrended)
            sig_norm = signal_detrended - np.mean(signal_detrended)
            var = np.var(sig_norm)
            if var > 1e-10:
                acf = np.correlate(sig_norm, sig_norm, mode='full')[n-1:]
                acf = acf / (var * n)
                # Find first positive peak after initial decay
                peaks = []
                for i in range(2, min(len(acf) - 1, n // 2)):
                    if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.15:
                        peaks.append((i, acf[i]))
                        break

                if peaks:
                    period_frames = peaks[0][0]
                    acf_val = peaks[0][1]
                    confidence = min(0.9, 0.5 + 0.4 * acf_val)
                    evidence = [
                        f"Periodic oscillation detected in {signal_label} with period ~{period_frames} frames",
                        f"Autocorrelation at period: {acf_val:.3f}",
                    ]

                    # Check SASA correlation with Rg oscillations
                    if isinstance(sasa_data, dict) and "sasa" in sasa_data:
                        sasa_series = np.array(sasa_data["sasa"])
                        min_len = min(len(signal), len(sasa_series))
                        if min_len > 30:
                            corr = np.corrcoef(signal[:min_len], sasa_series[:min_len])[0, 1]
                            if abs(corr) > 0.4:
                                evidence.append(f"SASA correlates with {signal_label} (r={corr:.2f}), confirming solvent exposure cycling")
                                confidence = min(0.92, confidence + 0.1)

                    insights.append({
                        "type": "breathing_motion",
                        "residues": [],
                        "description": f"Breathing motion detected via periodic {signal_label} oscillations "
                                      f"with period ~{period_frames} frames (autocorrelation={acf_val:.2f}). "
                                      f"This opening-closing motion suggests the protein undergoes "
                                      f"cyclic conformational changes, potentially linked to substrate "
                                      f"access, product release, or allosteric gating.",
                        "confidence": round(confidence, 2),
                        "evidence": evidence,
                        "category": "dynamic",
                    })

        return insights

    def _detect_cracking_events(self, result) -> List[Dict]:
        """Detect local unfolding (cracking) from transient secondary structure loss."""
        insights = []
        ss_data = result.secondary_structure
        rmsf_data = result.rmsf

        if not isinstance(ss_data, dict) or ss_data.get("error"):
            return insights
        if "per_residue_dominant_ss" not in ss_data or "ss_fractions" not in ss_data:
            return insights

        frac = ss_data["ss_fractions"]
        helix_frac = frac.get("helix", [])
        sheet_frac = frac.get("sheet", [])

        if not helix_frac and not sheet_frac:
            return insights

        # Look for frames where secondary structure drops significantly
        cracking_events = []

        for label, series in [("helix", helix_frac), ("sheet", sheet_frac)]:
            arr = np.array(series)
            if len(arr) < 20:
                continue
            mean_frac = np.mean(arr)
            std_frac = np.std(arr)
            if mean_frac < 0.05 or std_frac < 0.005:
                continue
            # Frames where fraction drops > 2 std below mean
            threshold = mean_frac - 2 * std_frac
            dip_frames = np.where(arr < threshold)[0]
            if len(dip_frames) > 0:
                # Group contiguous dip frames into events
                events = []
                start = dip_frames[0]
                for j in range(1, len(dip_frames)):
                    if dip_frames[j] - dip_frames[j-1] > 3:
                        events.append((start, dip_frames[j-1]))
                        start = dip_frames[j]
                events.append((start, dip_frames[-1]))

                for ev_start, ev_end in events[:3]:
                    depth = mean_frac - float(np.min(arr[ev_start:ev_end+1]))
                    cracking_events.append({
                        "ss_type": label,
                        "start_frame": int(ev_start),
                        "end_frame": int(ev_end),
                        "depth": depth,
                        "mean_frac": mean_frac,
                    })

        if cracking_events:
            cracking_events.sort(key=lambda x: -x["depth"])
            top = cracking_events[:3]
            evidence = []
            for ev in top:
                evidence.append(
                    f"{ev['ss_type'].capitalize()} content drops by {ev['depth']*100:.1f}% "
                    f"(frames {ev['start_frame']}-{ev['end_frame']})"
                )

            # Cross-reference with RMSF to find which residues are involved
            cracking_residues = []
            if isinstance(rmsf_data, dict) and "flexible_segments" in rmsf_data:
                for seg in rmsf_data["flexible_segments"][:5]:
                    cracking_residues.extend(range(seg["start"], seg["end"] + 1))

            confidence = min(0.85, 0.5 + 0.3 * top[0]["depth"] / (top[0]["mean_frac"] + 1e-8))
            insights.append({
                "type": "cracking_event",
                "residues": cracking_residues[:20],
                "description": f"Local unfolding (cracking) events detected: {len(cracking_events)} episodes "
                              f"of transient secondary structure loss. The most significant event shows "
                              f"{top[0]['ss_type']} content dropping by {top[0]['depth']*100:.1f}% during "
                              f"frames {top[0]['start_frame']}-{top[0]['end_frame']}. Cracking is associated "
                              f"with allosteric signal propagation and can expose transient binding sites "
                              f"or facilitate conformational switching.",
                "confidence": round(confidence, 2),
                "evidence": evidence,
                "category": "dynamic",
            })

        return insights

    def _detect_cryptic_binding_sites(self, result) -> List[Dict]:
        """Detect transient pockets from SASA variance + tunnel data."""
        insights = []
        tunnel_data = result.tunnels
        rmsf_data = result.rmsf

        if not isinstance(tunnel_data, dict) or tunnel_data.get("error"):
            return insights

        bottleneck = tunnel_data.get("bottleneck_residues", [])
        cavity_volumes = tunnel_data.get("cavity_volumes", [])

        if not bottleneck:
            return insights

        # Check for volume fluctuation indicating transient pocket opening
        evidence = []
        vol_arr = np.array(cavity_volumes) if cavity_volumes else np.array([])
        vol_cv = 0.0
        if len(vol_arr) > 10:
            vol_cv = np.std(vol_arr) / (np.mean(vol_arr) + 1e-8)
            if vol_cv > 0.3:
                evidence.append(
                    f"Cavity volume coefficient of variation: {vol_cv:.2f} "
                    f"(high fluctuation indicates transient pocket opening)"
                )

        # Identify bottleneck residues with high RMSF (gating residues)
        gating_residues = []
        if isinstance(rmsf_data, dict) and "rmsf" in rmsf_data:
            rmsf = np.array(rmsf_data["rmsf"])
            resids = rmsf_data.get("resids", list(range(len(rmsf))))
            mean_rmsf = np.mean(rmsf)
            rmsf_lookup = dict(zip(resids, rmsf))
            for bn in bottleneck:
                rid = bn["resid"]
                if rid in rmsf_lookup and rmsf_lookup[rid] > mean_rmsf * 1.2:
                    gating_residues.append(rid)

        if evidence or gating_residues:
            bn_resids = [b["resid"] for b in bottleneck[:10]]
            confidence = 0.6
            if gating_residues:
                evidence.append(
                    f"Gating residues (high RMSF + cavity-lining): {', '.join(map(str, gating_residues[:5]))}"
                )
                confidence += 0.1
            if vol_cv > 0.3:
                confidence += 0.1

            insights.append({
                "type": "cryptic_binding_site",
                "residues": list(set(bn_resids + gating_residues))[:15],
                "description": f"Potential cryptic binding site detected. Cavity volume fluctuations "
                              f"and flexible bottleneck residues suggest a transient pocket that opens "
                              f"during simulation but is absent in the static structure. "
                              f"{'Gating residues ' + ', '.join(map(str, gating_residues[:5])) + ' control access to this pocket. ' if gating_residues else ''}"
                              f"Cryptic sites are high-value drug targets accessible only through "
                              f"conformational dynamics.",
                "confidence": round(min(0.85, confidence), 2),
                "evidence": evidence,
                "category": "binding",
            })

        return insights

    def _score_druggability(self, result) -> List[Dict]:
        """Score detected pockets for druggability based on hydrophobicity, enclosure, and volume."""
        insights = []
        tunnel_data = result.tunnels
        ifp_data = result.interaction_fingerprints
        energy_data = result.energy_decomposition

        if not isinstance(tunnel_data, dict) or tunnel_data.get("error"):
            return insights

        bottleneck = tunnel_data.get("bottleneck_residues", [])
        cavity_volumes = tunnel_data.get("cavity_volumes", [])

        if not bottleneck or not cavity_volumes:
            return insights

        mean_vol = np.mean(cavity_volumes)
        bn_resids = [b["resid"] for b in bottleneck[:15]]

        # Druggability scoring heuristics
        score = 0.0
        evidence = []

        # Volume: ideal drug-binding pocket is 200-1000 A^3
        if 200 <= mean_vol <= 1000:
            score += 0.3
            evidence.append(f"Mean cavity volume {mean_vol:.0f} A^3 is in the ideal druggable range (200-1000 A^3)")
        elif 100 <= mean_vol < 200 or 1000 < mean_vol <= 1500:
            score += 0.15
            evidence.append(f"Mean cavity volume {mean_vol:.0f} A^3 is marginally druggable")
        else:
            evidence.append(f"Mean cavity volume {mean_vol:.0f} A^3 is outside typical druggable range")

        # Hydrophobic character of lining residues from IFP
        if isinstance(ifp_data, dict) and "per_residue_types" in ifp_data:
            per_res = ifp_data["per_residue_types"]
            hydrophobic_count = 0
            total_bn = 0
            for rid in bn_resids:
                rid_str = str(rid)
                if rid_str in per_res:
                    total_bn += 1
                    if per_res[rid_str].get("hydrophobic", 0) > 0.3:
                        hydrophobic_count += 1
            if total_bn > 0:
                hydro_frac = hydrophobic_count / total_bn
                if hydro_frac > 0.4:
                    score += 0.25
                    evidence.append(f"{hydro_frac*100:.0f}% of cavity-lining residues are hydrophobic (favourable for drug binding)")
                elif hydro_frac > 0.2:
                    score += 0.1
                    evidence.append(f"{hydro_frac*100:.0f}% of cavity-lining residues are hydrophobic")

        # Energy: strong interactions at pocket residues
        if isinstance(energy_data, dict) and "per_residue_energy" in energy_data:
            per_res_e = energy_data["per_residue_energy"]
            pocket_energy = 0.0
            counted = 0
            for rid in bn_resids:
                rid_str = str(rid)
                if rid_str in per_res_e:
                    pocket_energy += abs(per_res_e[rid_str])
                    counted += 1
            if counted > 0:
                avg_pocket_e = pocket_energy / counted
                if avg_pocket_e > 50:
                    score += 0.2
                    evidence.append(f"Average pocket residue interaction energy: {avg_pocket_e:.1f} kJ/mol (strong)")
                else:
                    score += 0.05
                    evidence.append(f"Average pocket residue interaction energy: {avg_pocket_e:.1f} kJ/mol")

        # Volume stability (enclosure)
        vol_cv = np.std(cavity_volumes) / (mean_vol + 1e-8)
        if vol_cv < 0.2:
            score += 0.15
            evidence.append(f"Stable pocket volume (CV={vol_cv:.2f}): well-defined, enclosed binding site")
        elif vol_cv < 0.4:
            score += 0.05
            evidence.append(f"Moderately stable pocket volume (CV={vol_cv:.2f})")

        drug_label = "highly druggable" if score > 0.7 else "moderately druggable" if score > 0.4 else "low druggability"
        insights.append({
            "type": "druggability_score",
            "residues": bn_resids,
            "description": f"Druggability assessment: the primary detected cavity is {drug_label} "
                          f"(score={score:.2f}/1.0). "
                          f"Assessment considers pocket volume ({mean_vol:.0f} A^3), "
                          f"hydrophobic character, interaction energy, and volume stability. "
                          f"{'This pocket is a strong candidate for structure-based drug design.' if score > 0.6 else 'Consider experimental validation (e.g., fragment screening) to confirm binding potential.'}",
            "confidence": round(min(0.85, 0.5 + score * 0.4), 2),
            "evidence": evidence,
            "category": "binding",
        })

        return insights

    def _predict_ptm_sites(self, result) -> List[Dict]:
        """Predict post-translational modification site accessibility from SASA + RMSF + sequence."""
        insights = []
        rmsf_data = result.rmsf
        ss_data = result.secondary_structure

        if not isinstance(rmsf_data, dict) or "rmsf" not in rmsf_data:
            return insights

        rmsf = np.array(rmsf_data["rmsf"])
        resids = rmsf_data.get("resids", list(range(len(rmsf))))
        resnames = rmsf_data.get("resnames", [])

        if not resnames:
            return insights

        mean_rmsf = np.mean(rmsf)
        rmsf_lookup = dict(zip(resids, rmsf))
        resname_lookup = dict(zip(resids, resnames))

        # Gather per-residue SS info
        coil_residues = set()
        if isinstance(ss_data, dict) and "per_residue_dominant_ss" in ss_data:
            ss_resids = ss_data.get("resids", [])
            dom_ss = ss_data["per_residue_dominant_ss"]
            for rid, ss in zip(ss_resids, dom_ss):
                if ss == "C":
                    coil_residues.add(rid)

        # Identify PTM-accessible sites
        ptm_candidates = []
        for i, rid in enumerate(resids):
            rn = resname_lookup.get(rid, "")
            rn3 = rn[:3].upper()

            ptm_type = None
            if rn3 in self._PTM_PHOSPHO_RESIDUES:
                ptm_type = "phosphorylation"
            elif rn3 in self._PTM_GLYCO_RESIDUES:
                # Check for N-X-S/T sequon
                if i + 2 < len(resids):
                    next2 = resname_lookup.get(resids[i+2], "")[:3].upper()
                    if next2 in ("SER", "THR"):
                        ptm_type = "N-glycosylation"
            elif rn3 in self._PTM_UBIQ_RESIDUES:
                ptm_type = "ubiquitination/acetylation"

            if ptm_type is None:
                continue

            # Accessibility: high RMSF (flexible) + in coil region
            flexibility = rmsf_lookup.get(rid, 0)
            is_flexible = flexibility > mean_rmsf * 0.8
            is_coil = rid in coil_residues
            accessibility_score = 0.0
            if is_flexible:
                accessibility_score += 0.4
            if is_coil:
                accessibility_score += 0.3
            # Surface exposure bonus
            if flexibility > mean_rmsf * 1.3:
                accessibility_score += 0.2

            if accessibility_score >= 0.5:
                ptm_candidates.append({
                    "resid": int(rid),
                    "resname": rn3,
                    "ptm_type": ptm_type,
                    "score": accessibility_score,
                    "flexibility": flexibility,
                    "is_coil": is_coil,
                })

        if ptm_candidates:
            ptm_candidates.sort(key=lambda x: -x["score"])
            top = ptm_candidates[:10]
            resid_list = [c["resid"] for c in top]
            evidence = []
            for c in top[:5]:
                evidence.append(
                    f"{c['resname']}{c['resid']}: {c['ptm_type']} candidate "
                    f"(accessibility={c['score']:.2f}, RMSF={c['flexibility']:.2f}A, "
                    f"{'coil' if c['is_coil'] else 'structured'})"
                )

            # Group by PTM type
            type_counts = {}
            for c in top:
                type_counts[c["ptm_type"]] = type_counts.get(c["ptm_type"], 0) + 1
            type_desc = ", ".join(f"{v} {k}" for k, v in type_counts.items())

            insights.append({
                "type": "ptm_site_prediction",
                "residues": resid_list,
                "description": f"Identified {len(ptm_candidates)} potential post-translational modification "
                              f"sites accessible during MD simulation (top {len(top)} shown). "
                              f"Types: {type_desc}. These residues are surface-exposed, dynamically "
                              f"flexible, and in loop/coil regions — structural prerequisites for "
                              f"enzymatic PTM. Phosphorylation at these sites could modulate protein "
                              f"dynamics and allosteric signaling.",
                "confidence": round(min(0.8, 0.5 + 0.05 * len(ptm_candidates)), 2),
                "evidence": evidence,
                "category": "structural",
            })

        return insights

    def _detect_ppi_hotspots(self, result) -> List[Dict]:
        """Detect protein-protein interaction hotspot patches on the surface."""
        insights = []
        rmsf_data = result.rmsf
        contact_data = result.contacts
        energy_data = result.energy_decomposition

        if not isinstance(rmsf_data, dict) or "rmsf" not in rmsf_data:
            return insights

        rmsf = np.array(rmsf_data["rmsf"])
        resids = list(rmsf_data.get("resids", range(len(rmsf))))
        resnames = rmsf_data.get("resnames", [])

        if not resnames:
            return insights

        mean_rmsf = np.mean(rmsf)
        resname_lookup = dict(zip(resids, resnames))

        # PPI hotspots: surface residues with hydrophobic/aromatic character + contacts + energy
        surface_candidates = []
        for i, rid in enumerate(resids):
            rn = resname_lookup.get(rid, "")[:3].upper()
            f_val = float(rmsf[i])
            # Surface: moderate flexibility
            if 0.5 * mean_rmsf < f_val < 2.0 * mean_rmsf:
                is_hydrophobic = rn in self._HYDROPHOBIC_RESIDUES
                is_aromatic = rn in self._AROMATIC_RESIDUES
                surface_candidates.append({
                    "resid": int(rid),
                    "resname": rn,
                    "rmsf": f_val,
                    "hydrophobic": is_hydrophobic,
                    "aromatic": is_aromatic,
                    "ppi_score": 0.0,
                })

        if not surface_candidates:
            return insights

        sc_lookup = {c["resid"]: c for c in surface_candidates}
        for c in surface_candidates:
            if c["hydrophobic"]:
                c["ppi_score"] += 0.3
            if c["aromatic"]:
                c["ppi_score"] += 0.2

        # Boost for persistent contacts with other surface residues
        if isinstance(contact_data, dict) and "persistent_contacts" in contact_data:
            for pc in contact_data["persistent_contacts"]:
                r1, r2 = pc.get("resid_1"), pc.get("resid_2")
                if r1 in sc_lookup and r2 in sc_lookup:
                    sc_lookup[r1]["ppi_score"] += 0.1
                    sc_lookup[r2]["ppi_score"] += 0.1

        # Boost for energy hotspots
        if isinstance(energy_data, dict) and "top_pairs" in energy_data:
            for p in energy_data["top_pairs"][:20]:
                for rid in [p.get("resid_i"), p.get("resid_j")]:
                    if rid in sc_lookup:
                        sc_lookup[rid]["ppi_score"] += 0.1

        scored = sorted(surface_candidates, key=lambda x: -x["ppi_score"])
        top_hotspots = [c for c in scored if c["ppi_score"] >= 0.4][:15]

        if top_hotspots:
            resid_list = [c["resid"] for c in top_hotspots]
            evidence = [
                f"{c['resname']}{c['resid']}: PPI score={c['ppi_score']:.2f} ({'hydrophobic' if c['hydrophobic'] else 'polar'})"
                for c in top_hotspots[:5]
            ]

            insights.append({
                "type": "ppi_interface_hotspot",
                "residues": resid_list,
                "description": f"Predicted protein-protein interaction hotspots: {len(top_hotspots)} "
                              f"surface residues with complementary properties for PPI interfaces. "
                              f"Top candidates: {', '.join(c['resname'] + str(c['resid']) for c in top_hotspots[:5])}. "
                              f"These residues combine moderate flexibility, surface exposure, and "
                              f"hydrophobic/aromatic character — hallmarks of PPI interface patches.",
                "confidence": round(min(0.75, 0.45 + 0.03 * len(top_hotspots)), 2),
                "evidence": evidence,
                "category": "binding",
            })

        return insights

    def _detect_interface_dynamics(self, result) -> List[Dict]:
        """Detect surface regions switching between ordered and disordered states (conformational selection)."""
        insights = []
        ss_data = result.secondary_structure
        rmsf_data = result.rmsf

        if not isinstance(ss_data, dict) or ss_data.get("error"):
            return insights
        if "per_residue_dominant_ss" not in ss_data:
            return insights

        ss_resids = ss_data.get("resids", [])
        dom_ss = ss_data.get("per_residue_dominant_ss", [])

        if not isinstance(rmsf_data, dict) or "rmsf" not in rmsf_data:
            return insights

        rmsf = np.array(rmsf_data["rmsf"])
        rmsf_resids = rmsf_data.get("resids", list(range(len(rmsf))))
        mean_rmsf = np.mean(rmsf)
        rmsf_lookup = dict(zip(rmsf_resids, rmsf))

        # Look for residues with non-dominant SS fraction (disorder-order switching)
        switching_residues = []

        per_res_ss = ss_data.get("per_residue_ss_fractions", {})
        if per_res_ss:
            for rid_str, fracs in per_res_ss.items():
                rid = int(rid_str)
                vals = list(fracs.values()) if isinstance(fracs, dict) else []
                if not vals:
                    continue
                max_frac = max(vals)
                # Order-disorder switching: dominant fraction < 0.7
                if max_frac < 0.7 and rmsf_lookup.get(rid, 0) > mean_rmsf * 0.8:
                    switching_residues.append({
                        "resid": rid,
                        "max_ss_frac": max_frac,
                        "rmsf": rmsf_lookup.get(rid, 0),
                    })
        else:
            # Fallback: high RMSF + coil dominant
            for rid, ss in zip(ss_resids, dom_ss):
                if ss == "C" and rmsf_lookup.get(rid, 0) > mean_rmsf * 1.3:
                    switching_residues.append({
                        "resid": rid,
                        "max_ss_frac": 0.5,
                        "rmsf": rmsf_lookup.get(rid, 0),
                    })

        if switching_residues:
            switching_residues.sort(key=lambda x: x["max_ss_frac"])
            top = switching_residues[:10]
            resid_list = [s["resid"] for s in top]
            evidence = [
                f"Residue {s['resid']}: dominant SS fraction={s['max_ss_frac']:.2f}, RMSF={s['rmsf']:.2f}A"
                for s in top[:5]
            ]

            insights.append({
                "type": "interface_conformational_selection",
                "residues": resid_list,
                "description": f"Identified {len(switching_residues)} residues undergoing order-disorder transitions, "
                              f"characteristic of conformational selection mechanisms. These residues "
                              f"alternate between structured and disordered states, potentially "
                              f"becoming ordered upon binding a partner protein (folding-upon-binding). "
                              f"Key switching residues: {', '.join(map(str, resid_list[:5]))}.",
                "confidence": round(min(0.75, 0.5 + 0.03 * len(switching_residues)), 2),
                "evidence": evidence,
                "category": "binding",
            })

        return insights

    def _infer_protonation_dynamics(self, result) -> List[Dict]:
        """Infer pH-dependent behavior from salt bridge dynamics of titratable residues."""
        insights = []
        sb_data = result.salt_bridges
        rmsf_data = result.rmsf

        if not isinstance(sb_data, dict) or sb_data.get("error"):
            return insights

        pairs = sb_data.get("pairs", [])
        if not pairs:
            return insights

        resnames = {}
        if isinstance(rmsf_data, dict) and "resnames" in rmsf_data:
            resnames = dict(zip(
                rmsf_data.get("resids", []),
                rmsf_data.get("resnames", [])
            ))

        # Find salt bridges involving titratable residues with partial occupancy
        dynamic_titratable = []
        for pair in pairs:
            r1, r2 = pair.get("resid_1"), pair.get("resid_2")
            occ = pair.get("occupancy", 0)
            rn1 = resnames.get(r1, "")[:3].upper()
            rn2 = resnames.get(r2, "")[:3].upper()

            titratable_involved = []
            if rn1 in self._TITRATABLE_RESIDUES:
                titratable_involved.append((r1, rn1))
            if rn2 in self._TITRATABLE_RESIDUES:
                titratable_involved.append((r2, rn2))

            # Partial occupancy suggests pH-sensitivity
            if titratable_involved and 0.15 < occ < 0.85:
                for rid, rn in titratable_involved:
                    dynamic_titratable.append({
                        "resid": rid,
                        "resname": rn,
                        "partner": r2 if rid == r1 else r1,
                        "occupancy": occ,
                    })

        if dynamic_titratable:
            seen = set()
            unique = []
            for d in dynamic_titratable:
                if d["resid"] not in seen:
                    seen.add(d["resid"])
                    unique.append(d)
            unique.sort(key=lambda x: abs(x["occupancy"] - 0.5))

            top = unique[:8]
            resid_list = [d["resid"] for d in top]
            evidence = [
                f"{d['resname']}{d['resid']}-{d['partner']}: salt bridge occupancy={d['occupancy']*100:.0f}%"
                for d in top[:5]
            ]

            insights.append({
                "type": "protonation_dynamics",
                "residues": resid_list,
                "description": f"Identified {len(unique)} titratable residues with dynamic salt bridge "
                              f"behavior (partial occupancy). Residues: "
                              f"{', '.join(d['resname'] + str(d['resid']) for d in top[:5])}. "
                              f"Partial salt bridge occupancy suggests these residues operate near "
                              f"their pKa and may switch protonation states under physiological pH "
                              f"changes, making them pH-dependent conformational switches.",
                "confidence": round(min(0.75, 0.5 + 0.04 * len(unique)), 2),
                "evidence": evidence,
                "category": "structural",
            })

        return insights

    def _detect_electrostatic_funnels(self, result) -> List[Dict]:
        """Detect charged residue clusters forming electrostatic funnels for substrate guidance."""
        insights = []
        rmsf_data = result.rmsf

        if not isinstance(rmsf_data, dict) or "resnames" not in rmsf_data:
            return insights

        resids = rmsf_data.get("resids", [])
        resnames = rmsf_data.get("resnames", [])
        rmsf = np.array(rmsf_data.get("rmsf", []))

        if not resnames or len(rmsf) == 0:
            return insights

        resname_lookup = dict(zip(resids, resnames))
        rmsf_lookup = dict(zip(resids, rmsf))
        mean_rmsf = np.mean(rmsf)

        # Find surface-exposed charged residues
        positive_surface = []
        negative_surface = []
        for rid in resids:
            rn = resname_lookup.get(rid, "")[:3].upper()
            f_val = rmsf_lookup.get(rid, 0)
            if f_val > mean_rmsf * 0.7:
                if rn in self._CHARGED_POSITIVE:
                    positive_surface.append(int(rid))
                elif rn in self._CHARGED_NEGATIVE:
                    negative_surface.append(int(rid))

        def find_clusters(resid_list, gap=4):
            if not resid_list:
                return []
            sorted_r = sorted(resid_list)
            clusters = [[sorted_r[0]]]
            for r in sorted_r[1:]:
                if r - clusters[-1][-1] <= gap:
                    clusters[-1].append(r)
                else:
                    clusters.append([r])
            return [c for c in clusters if len(c) >= 3]

        pos_clusters = find_clusters(positive_surface)
        neg_clusters = find_clusters(negative_surface)

        funnels = []
        for cluster in pos_clusters:
            funnels.append({"charge": "positive", "residues": cluster})
        for cluster in neg_clusters:
            funnels.append({"charge": "negative", "residues": cluster})

        if funnels:
            all_resids = []
            evidence = []
            for f in funnels[:3]:
                all_resids.extend(f["residues"])
                evidence.append(
                    f"{f['charge'].capitalize()} cluster: residues {', '.join(map(str, f['residues']))}"
                )

            insights.append({
                "type": "electrostatic_funnel",
                "residues": all_resids[:20],
                "description": f"Detected {len(funnels)} electrostatic surface clusters that may "
                              f"form substrate-guiding funnels. {len(pos_clusters)} positive and "
                              f"{len(neg_clusters)} negative charge clusters observed on the protein "
                              f"surface. These charged patches can act as electrostatic funnels, "
                              f"accelerating substrate/ligand association by steering charged "
                              f"molecules toward the active site via long-range electrostatic interactions.",
                "confidence": round(min(0.7, 0.45 + 0.08 * len(funnels)), 2),
                "evidence": evidence,
                "category": "structural",
            })

        return insights

    def _detect_aggregation_prone_regions(self, result) -> List[Dict]:
        """Identify exposed hydrophobic patches prone to amyloid-like aggregation."""
        insights = []
        rmsf_data = result.rmsf

        if not isinstance(rmsf_data, dict) or "rmsf" not in rmsf_data:
            return insights

        rmsf = np.array(rmsf_data["rmsf"])
        resids = rmsf_data.get("resids", list(range(len(rmsf))))
        resnames = rmsf_data.get("resnames", [])

        if not resnames:
            return insights

        mean_rmsf = np.mean(rmsf)

        # Compute per-residue aggregation score
        agg_scores = []
        for i, rid in enumerate(resids):
            rn = resnames[i][:3].upper() if i < len(resnames) else ""
            base_prop = self._AGGREGATION_PROPENSITY.get(rn, 0.0)
            flexibility = float(rmsf[i])

            # Surface-exposed hydrophobic = aggregation risk
            if base_prop > 0.3 and flexibility > mean_rmsf * 0.8:
                exposure_factor = min(1.0, flexibility / (mean_rmsf * 2))
                score = base_prop * (0.5 + 0.5 * exposure_factor)
                agg_scores.append({
                    "resid": int(rid),
                    "resname": rn,
                    "score": score,
                    "rmsf": flexibility,
                })

        if not agg_scores:
            return insights

        # Find contiguous aggregation-prone stretches (APRs)
        agg_scores.sort(key=lambda x: x["resid"])

        stretches = []
        current = [agg_scores[0]]
        for a in agg_scores[1:]:
            if a["resid"] - current[-1]["resid"] <= 2:
                current.append(a)
            else:
                if len(current) >= 3:
                    stretches.append(current)
                current = [a]
        if len(current) >= 3:
            stretches.append(current)

        if stretches:
            stretches.sort(key=lambda s: -np.mean([a["score"] for a in s]))
            top = stretches[:3]
            all_resids = []
            evidence = []
            for s in top:
                rids = [a["resid"] for a in s]
                avg_score = np.mean([a["score"] for a in s])
                all_resids.extend(rids)
                evidence.append(
                    f"APR: residues {rids[0]}-{rids[-1]} (length={len(s)}, "
                    f"mean aggregation score={avg_score:.2f})"
                )

            insights.append({
                "type": "aggregation_prone_region",
                "residues": all_resids,
                "description": f"Identified {len(stretches)} aggregation-prone regions (APRs) with "
                              f"exposed hydrophobic stretches. Most significant: residues "
                              f"{top[0][0]['resid']}-{top[0][-1]['resid']}. "
                              f"These stretches combine hydrophobic residues with surface exposure, "
                              f"resembling sequences known to nucleate amyloid-like aggregation. "
                              f"Consider protective mutations (e.g., gatekeeper charges) or "
                              f"formulation strategies to mitigate aggregation risk.",
                "confidence": round(min(0.75, 0.5 + 0.08 * len(stretches)), 2),
                "evidence": evidence,
                "category": "structural",
            })

        return insights

    def _detect_folding_intermediates(self, result) -> List[Dict]:
        """Identify metastable states with partial secondary structure as folding intermediates."""
        insights = []
        msm_data = result.msm
        ss_data = result.secondary_structure
        cluster_data = result.clustering

        if not isinstance(cluster_data, dict) or "n_clusters" not in cluster_data:
            return insights

        n_clusters = cluster_data["n_clusters"]
        if n_clusters < 2:
            return insights

        ss_frac = ss_data if isinstance(ss_data, dict) and not ss_data.get("error") else None

        evidence = []
        intermediate_detected = False

        if ss_frac and "ss_fractions" in ss_frac:
            helix = np.array(ss_frac["ss_fractions"].get("helix", []))
            sheet = np.array(ss_frac["ss_fractions"].get("sheet", []))

            if len(helix) > 0 or len(sheet) > 0:
                min_len = max(len(helix), len(sheet))
                total_structure = np.zeros(min_len)
                if len(helix) == min_len:
                    total_structure += helix
                if len(sheet) == min_len:
                    total_structure += sheet[:min_len]

                if len(total_structure) > 10:
                    mean_struct = np.mean(total_structure)
                    min_struct = np.min(total_structure)
                    low_struct_frames = np.where(total_structure < mean_struct * 0.7)[0]
                    if len(low_struct_frames) > len(total_structure) * 0.05:
                        intermediate_detected = True
                        frac_low = len(low_struct_frames) / len(total_structure)
                        evidence.append(
                            f"{frac_low*100:.1f}% of frames show >30% reduction in secondary "
                            f"structure content (partially unfolded states)"
                        )
                        evidence.append(
                            f"Mean structured fraction: {mean_struct*100:.1f}%, "
                            f"minimum: {min_struct*100:.1f}%"
                        )

        # Check MSM for short-lived metastable states
        if isinstance(msm_data, dict) and "metastable_states" in msm_data:
            meta = msm_data["metastable_states"]
            low_pop = [s for s in meta if 0.01 < s.get("population", 0) < 0.15]
            if low_pop:
                intermediate_detected = True
                for s in low_pop[:3]:
                    evidence.append(
                        f"Metastable state {s['state']}: population={s['population']*100:.1f}%, "
                        f"P_self={s['self_transition']:.2f} (short-lived intermediate)"
                    )

        if intermediate_detected:
            insights.append({
                "type": "folding_intermediate",
                "residues": [],
                "description": f"Evidence for folding/unfolding intermediate states detected. "
                              f"The simulation samples partially structured conformations that "
                              f"likely represent on-pathway or off-pathway folding intermediates. "
                              f"These intermediates may serve as aggregation nucleation points or "
                              f"as kinetic traps affecting the folding rate. "
                              f"{'Multiple metastable states with low population confirm transient intermediates.' if len(evidence) > 2 else ''}",
                "confidence": round(min(0.75, 0.5 + 0.05 * len(evidence)), 2),
                "evidence": evidence,
                "category": "transition",
            })

        return insights

    def _classify_functional_motions(self, result) -> List[Dict]:
        """Classify dominant PCA/NMA modes into known motion types."""
        insights = []
        nma_data = result.nma
        pca_data = result.pca
        domain_data = result.domains

        if not isinstance(nma_data, dict) or nma_data.get("error"):
            return insights

        collectivity = nma_data.get("mode_collectivity", [])
        if not collectivity:
            return insights

        n_domains = 0
        if isinstance(domain_data, dict) and "domain_info" in domain_data:
            n_domains = len(domain_data["domain_info"])

        pca_variance = []
        if isinstance(pca_data, dict) and "explained_variance" in pca_data:
            pca_variance = pca_data["explained_variance"]

        classifications = []
        evidence = []

        coll_1 = collectivity[0]

        if coll_1 > 0.4:
            if n_domains >= 2:
                motion_type = "hinge-bending"
                motion_desc = (
                    f"global hinge-bending motion between {n_domains} domains. "
                    f"This is the dominant functional motion, involving coordinated "
                    f"opening/closing of a cleft or interdomain angle"
                )
            else:
                motion_type = "breathing"
                motion_desc = (
                    f"global breathing motion involving {coll_1*100:.0f}% of residues. "
                    f"The protein undergoes collective expansion/contraction"
                )
        elif coll_1 > 0.2:
            motion_type = "shear/twist"
            motion_desc = (
                f"partial shear or twisting motion involving {coll_1*100:.0f}% of residues. "
                f"Different regions of the protein move in opposing directions"
            )
        else:
            motion_type = "local deformation"
            motion_desc = (
                f"localized deformation (collectivity={coll_1:.2f}). "
                f"The dominant motion is concentrated in a small subset of residues"
            )

        classifications.append(motion_type)
        evidence.append(f"Mode 1: {motion_type} (collectivity k={coll_1:.3f})")

        for idx in range(1, min(3, len(collectivity))):
            c = collectivity[idx]
            if c > 0.3:
                label = "collective rearrangement"
            elif c > 0.15:
                label = "partial domain motion"
            else:
                label = "local fluctuation"
            classifications.append(label)
            evidence.append(f"Mode {idx+1}: {label} (k={c:.3f})")

        if pca_variance and len(pca_variance) >= 2:
            top2_var = sum(pca_variance[:2])
            evidence.append(f"PC1+PC2 explain {top2_var*100:.1f}% of variance")
            if top2_var > 0.6:
                evidence.append("Dominant motions are highly concentrated in 2 principal components")

        insights.append({
            "type": "functional_motion_classification",
            "residues": [],
            "description": f"Functional motion classification: the dominant mode is a {motion_desc}. "
                          f"{'Higher modes contribute ' + ', '.join(classifications[1:]) + '. ' if len(classifications) > 1 else ''}"
                          f"Understanding these motion types helps predict functional mechanisms — "
                          f"hinge-bending facilitates substrate access, breathing enables "
                          f"allosteric signaling, and shear motions support catalytic rearrangements.",
            "confidence": round(min(0.8, 0.55 + 0.1 * coll_1 / 0.5), 2),
            "evidence": evidence,
            "category": "dynamic",
        })

        return insights

    def _correlate_motions_to_function(self, result) -> List[Dict]:
        """Map which functionally important residues are dynamically active along principal components."""
        insights = []
        pca_data = result.pca
        rmsf_data = result.rmsf
        ligand_data = result.ligand
        energy_data = result.energy_decomposition

        if not isinstance(pca_data, dict) or "explained_variance" not in pca_data:
            return insights

        variance = pca_data.get("explained_variance", [])
        if not variance or variance[0] < 0.1:
            return insights

        # Identify functionally important residues
        functional_resids = set()

        if isinstance(ligand_data, dict) and "key_binding_residues" in ligand_data:
            for r in ligand_data["key_binding_residues"]:
                functional_resids.add(r["resid"])

        if isinstance(energy_data, dict) and "top_pairs" in energy_data:
            for p in energy_data["top_pairs"][:10]:
                functional_resids.add(p.get("resid_i"))
                functional_resids.add(p.get("resid_j"))

        gnn = result.gnn_results
        if isinstance(gnn, dict) and "top_residues" in gnn:
            for r in gnn["top_residues"][:5]:
                functional_resids.add(r["resid"])

        functional_resids.discard(None)

        if not functional_resids:
            return insights

        # Cross-reference with RMSF
        motionally_active_func_residues = []

        if isinstance(rmsf_data, dict) and "rmsf" in rmsf_data:
            rmsf = np.array(rmsf_data["rmsf"])
            resids = rmsf_data.get("resids", list(range(len(rmsf))))
            mean_rmsf = np.mean(rmsf)
            rmsf_lookup = dict(zip(resids, rmsf))

            for rid in functional_resids:
                f_val = rmsf_lookup.get(rid, 0)
                if f_val > mean_rmsf * 0.8:
                    motionally_active_func_residues.append({
                        "resid": int(rid),
                        "rmsf": f_val,
                        "rmsf_ratio": f_val / mean_rmsf,
                    })

        if motionally_active_func_residues:
            motionally_active_func_residues.sort(key=lambda x: -x["rmsf_ratio"])
            top = motionally_active_func_residues[:8]
            resid_list = [m["resid"] for m in top]
            evidence = [
                f"Functional residue {m['resid']}: RMSF={m['rmsf']:.2f}A ({m['rmsf_ratio']:.1f}x mean)"
                for m in top[:5]
            ]
            evidence.append(f"PC1 explains {variance[0]*100:.1f}% of total variance")

            insights.append({
                "type": "motion_function_coupling",
                "residues": resid_list,
                "description": f"Identified {len(motionally_active_func_residues)} functionally important "
                              f"residues that are dynamically active along dominant motions. "
                              f"Residues {', '.join(map(str, resid_list[:5]))} combine functional "
                              f"significance (binding, energy, or GNN importance) with high "
                              f"displacement in the dominant principal components. This coupling "
                              f"suggests these motions are catalytically or functionally relevant, "
                              f"not mere thermal fluctuations.",
                "confidence": round(min(0.8, 0.5 + 0.05 * len(motionally_active_func_residues)), 2),
                "evidence": evidence,
                "category": "dynamic",
            })

        return insights

    def _detect_hbond_network_rewiring(self, result) -> List[Dict]:
        """Detect cooperative H-bond switching events in the network topology."""
        insights = []
        hbond_data = result.hbonds
        clustering_data = result.clustering

        if not isinstance(hbond_data, dict) or hbond_data.get("error"):
            return insights

        persistent = hbond_data.get("persistent_hbonds", [])
        hbond_counts = hbond_data.get("hbond_counts", [])

        if not persistent or not hbond_counts:
            return insights

        hb_arr = np.array(hbond_counts)
        if len(hb_arr) < 20:
            return insights

        # Detect cooperative switching: large sudden changes in H-bond count
        diff = np.abs(np.diff(hb_arr))
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        threshold = mean_diff + 2 * std_diff
        cooperative_frames = np.where(diff > threshold)[0]

        evidence = []
        if len(cooperative_frames) > 0:
            n_events = len(cooperative_frames)
            max_change_idx = int(np.argmax(diff))
            max_change = float(diff[max_change_idx])
            evidence.append(f"{n_events} cooperative H-bond switching events detected")
            evidence.append(f"Largest event at frame {max_change_idx}: {max_change:.0f} H-bonds changed simultaneously")

            # Check if events correlate with conformational transitions
            if isinstance(clustering_data, dict) and "labels" in clustering_data:
                labels = np.array(clustering_data["labels"])
                if len(labels) > max_change_idx:
                    transition_count = 0
                    for cf in cooperative_frames:
                        if cf < len(labels) - 1 and labels[cf] != labels[cf + 1]:
                            transition_count += 1
                    if transition_count > 0:
                        evidence.append(
                            f"{transition_count}/{n_events} H-bond rewiring events coincide "
                            f"with conformational state transitions"
                        )

            # Identify residues in dynamic H-bonds
            partial_hbonds = [p for p in persistent if 0.3 < p.get("occupancy", 0) < 0.7]
            relay_resids = []
            for hb in partial_hbonds[:10]:
                donor = hb.get("donor_resid")
                acceptor = hb.get("acceptor_resid")
                if donor:
                    relay_resids.append(int(donor))
                if acceptor:
                    relay_resids.append(int(acceptor))

            if partial_hbonds:
                evidence.append(
                    f"{len(partial_hbonds)} H-bonds with partial occupancy (30-70%) "
                    f"form the dynamic backbone of the network"
                )

            insights.append({
                "type": "hbond_network_rewiring",
                "residues": list(set(relay_resids))[:15],
                "description": f"Detected {n_events} cooperative H-bond network rewiring events "
                              f"where multiple hydrogen bonds break and reform simultaneously. "
                              f"The largest event involves {max_change:.0f} H-bonds at frame "
                              f"{max_change_idx}. This cooperative switching is characteristic of "
                              f"proton relay chains and allosteric signal transmission through "
                              f"the H-bond network. "
                              f"{'Partial-occupancy H-bonds at residues ' + ', '.join(map(str, list(set(relay_resids))[:5])) + ' form the dynamic relay.' if relay_resids else ''}",
                "confidence": round(min(0.8, 0.5 + 0.03 * min(n_events, 10)), 2),
                "evidence": evidence,
                "category": "allosteric",
            })

        return insights

    def _identify_structural_waters(self, result) -> List[Dict]:
        """Distinguish structural waters from transient bulk using water bridge persistence."""
        insights = []
        wb_data = result.water_bridges

        if not isinstance(wb_data, dict) or wb_data.get("error"):
            return insights

        bridges = wb_data.get("bridges", [])
        if not bridges:
            return insights

        structural = [b for b in bridges if b.get("occupancy", 0) > 0.6]
        transient = [b for b in bridges if b.get("occupancy", 0) < 0.2]

        if not structural:
            return insights

        resids = set()
        evidence = []
        for b in structural[:8]:
            r1, r2 = b["resid_1"], b["resid_2"]
            resids.add(r1)
            resids.add(r2)
            evidence.append(
                f"Water bridge {r1}-{r2}: {b['occupancy']*100:.0f}% occupancy (structural)"
            )

        insights.append({
            "type": "structural_waters",
            "residues": list(resids)[:15],
            "description": f"Identified {len(structural)} structurally integral water-mediated "
                          f"contacts (>60% occupancy) vs {len(transient)} transient bulk water "
                          f"interactions (<20%). Structural waters at "
                          f"{', '.join(str(r) for r in list(resids)[:5])} are functionally "
                          f"important — they stabilize domain interfaces, mediate long-range "
                          f"interactions, and are often conserved across crystal structures. "
                          f"Displacement of these waters by drug molecules can contribute "
                          f"favourably to binding free energy (water displacement entropy gain).",
            "confidence": round(min(0.8, 0.55 + 0.05 * len(structural)), 2),
            "evidence": evidence,
            "category": "structural",
        })

        return insights

    def _map_local_stiffness(self, result) -> List[Dict]:
        """Compute per-residue stiffness from PRS response + NMA B-factors."""
        insights = []
        prs_data = result.prs
        nma_data = result.nma
        rmsf_data = result.rmsf

        b_factors = None
        response_scores = None

        if isinstance(nma_data, dict) and "b_factors" in nma_data:
            b_factors = np.array(nma_data["b_factors"])

        if isinstance(prs_data, dict) and "response_matrix" in prs_data:
            resp = np.array(prs_data["response_matrix"])
            if resp.ndim == 2:
                response_scores = np.mean(resp, axis=0)

        if b_factors is None and response_scores is None:
            return insights

        resids = []
        if isinstance(rmsf_data, dict):
            resids = rmsf_data.get("resids", [])

        stiffness = None
        evidence = []

        if b_factors is not None and len(b_factors) > 5:
            bf_norm = b_factors / (np.max(b_factors) + 1e-10)
            stiffness = 1.0 - bf_norm
            evidence.append(f"Stiffness derived from NMA B-factors ({len(b_factors)} residues)")

        if response_scores is not None and len(response_scores) > 5:
            rs_norm = response_scores / (np.max(response_scores) + 1e-10)
            prs_stiffness = 1.0 - rs_norm
            if stiffness is not None and len(stiffness) == len(prs_stiffness):
                stiffness = 0.5 * stiffness + 0.5 * prs_stiffness
                evidence.append("Combined with PRS response scores")
            else:
                stiffness = prs_stiffness
                evidence.append(f"Stiffness derived from PRS response ({len(response_scores)} residues)")

        if stiffness is None:
            return insights

        n_res = len(stiffness)
        if not resids:
            resids = list(range(n_res))
        resids = resids[:n_res]

        mean_stiff = np.mean(stiffness)
        std_stiff = np.std(stiffness)

        stiff_resids = [int(resids[i]) for i in range(n_res) if stiffness[i] > mean_stiff + std_stiff]
        soft_resids = [int(resids[i]) for i in range(n_res) if stiffness[i] < mean_stiff - std_stiff]

        evidence.append(f"Rigid scaffold: {len(stiff_resids)} residues (stiffness > mean+std)")
        evidence.append(f"Compliant regions: {len(soft_resids)} residues (stiffness < mean-std)")

        insights.append({
            "type": "local_stiffness_map",
            "residues": stiff_resids[:15],
            "description": f"Local stiffness mapping identifies {len(stiff_resids)} rigid load-bearing "
                          f"residues and {len(soft_resids)} compliant regions. Rigid residues "
                          f"({', '.join(map(str, stiff_resids[:5]))}) form the mechanical scaffold "
                          f"of the protein — mutations here risk structural collapse. "
                          f"Compliant residues ({', '.join(map(str, soft_resids[:5]))}) absorb "
                          f"mechanical stress and enable functional deformations.",
            "confidence": 0.7,
            "evidence": evidence,
            "category": "dynamic",
        })

        return insights

    def _detect_force_propagation(self, result) -> List[Dict]:
        """Map how forces propagate through the structure using PRS perturbation-response."""
        insights = []
        prs_data = result.prs
        allosteric_data = result.allosteric

        if not isinstance(prs_data, dict) or prs_data.get("error"):
            return insights

        effectors = prs_data.get("top_effectors", [])
        sensors = prs_data.get("top_sensors", [])

        if not effectors or not sensors:
            return insights

        evidence = []
        resp_matrix = prs_data.get("response_matrix")

        force_paths = []
        if resp_matrix is not None:
            resp = np.array(resp_matrix)
            if resp.ndim == 2:
                n = resp.shape[0]
                for eff in effectors[:5]:
                    eff_idx = eff.get("index", eff.get("resid", 0))
                    if isinstance(eff_idx, int) and eff_idx < n:
                        row = resp[eff_idx]
                        top_sensors_idx = np.argsort(row)[-3:][::-1]
                        for s_idx in top_sensors_idx:
                            if s_idx != eff_idx:
                                force_paths.append({
                                    "effector": eff["resid"],
                                    "sensor_idx": int(s_idx),
                                    "strength": float(row[s_idx]),
                                })

        if force_paths:
            force_paths.sort(key=lambda x: -x["strength"])
            for fp in force_paths[:3]:
                evidence.append(
                    f"Effector {fp['effector']} -> sensor index {fp['sensor_idx']}: "
                    f"response strength={fp['strength']:.4f}"
                )

        if isinstance(allosteric_data, dict) and "shortest_paths" in allosteric_data:
            paths = allosteric_data["shortest_paths"]
            if paths:
                evidence.append(
                    f"Allosteric network confirms {len(paths)} signal propagation corridors"
                )

        eff_resids = [e["resid"] for e in effectors[:5]]
        sens_resids = [s["resid"] for s in sensors[:5]]

        insights.append({
            "type": "force_propagation_pathway",
            "residues": list(set(eff_resids + sens_resids)),
            "description": f"Force propagation analysis maps how mechanical perturbations "
                          f"transmit through the structure. Top effectors "
                          f"({', '.join(map(str, eff_resids[:3]))}) propagate displacement "
                          f"to distant sensors ({', '.join(map(str, sens_resids[:3]))}). "
                          f"These mechano-transduction pathways reveal how the protein "
                          f"transmits force from binding events or PTMs to distant functional "
                          f"sites, providing mechanistic insight into allosteric regulation.",
            "confidence": round(min(0.8, 0.55 + 0.05 * len(force_paths)), 2),
            "evidence": evidence,
            "category": "allosteric",
        })

        return insights

    def _predict_mutation_sensitivity(self, result) -> List[Dict]:
        """Predict residues most sensitive to mutation by combining multiple importance metrics."""
        insights = []
        gnn_data = result.gnn_results
        allosteric_data = result.allosteric
        hbond_data = result.hbonds
        rmsf_data = result.rmsf
        contact_data = result.contacts

        # Collect per-residue importance from all sources
        scores = {}  # resid -> {total, sources}

        # GNN importance
        if isinstance(gnn_data, dict) and "top_residues" in gnn_data:
            max_imp = max((r["importance"] for r in gnn_data["top_residues"]), default=1)
            for r in gnn_data["top_residues"][:30]:
                rid = r["resid"]
                if rid not in scores:
                    scores[rid] = {"total": 0, "sources": []}
                norm = r["importance"] / (max_imp + 1e-10)
                scores[rid]["total"] += norm * 0.25
                scores[rid]["sources"].append(f"GNN importance={r['importance']:.3f}")

        # Allosteric hub centrality
        if isinstance(allosteric_data, dict) and "hub_residues" in allosteric_data:
            hubs = allosteric_data["hub_residues"]
            max_btw = max((h["betweenness"] for h in hubs), default=1)
            for h in hubs[:30]:
                rid = h["resid"]
                if rid not in scores:
                    scores[rid] = {"total": 0, "sources": []}
                norm = h["betweenness"] / (max_btw + 1e-10)
                scores[rid]["total"] += norm * 0.25
                scores[rid]["sources"].append(f"Hub betweenness={h['betweenness']:.4f}")

        # H-bond participation
        if isinstance(hbond_data, dict) and "persistent_hbonds" in hbond_data:
            hb_count = {}
            for hb in hbond_data["persistent_hbonds"]:
                for key in ("donor_resid", "acceptor_resid"):
                    rid = hb.get(key)
                    if rid is not None:
                        hb_count[rid] = hb_count.get(rid, 0) + 1
            if hb_count:
                max_hb = max(hb_count.values())
                for rid, cnt in hb_count.items():
                    if rid not in scores:
                        scores[rid] = {"total": 0, "sources": []}
                    norm = cnt / (max_hb + 1e-10)
                    scores[rid]["total"] += norm * 0.2
                    scores[rid]["sources"].append(f"H-bonds={cnt}")

        # Burial (low RMSF = buried)
        if isinstance(rmsf_data, dict) and "rmsf" in rmsf_data:
            rmsf = np.array(rmsf_data["rmsf"])
            resids_rmsf = rmsf_data.get("resids", list(range(len(rmsf))))
            max_rmsf = np.max(rmsf) if len(rmsf) > 0 else 1
            for i, rid in enumerate(resids_rmsf):
                if rid not in scores:
                    scores[rid] = {"total": 0, "sources": []}
                burial_score = 1.0 - (rmsf[i] / (max_rmsf + 1e-10))
                scores[rid]["total"] += burial_score * 0.15

        # Contact density
        if isinstance(contact_data, dict) and "persistent_contacts" in contact_data:
            contact_count = {}
            for pc in contact_data["persistent_contacts"]:
                for key in ("resid_1", "resid_2"):
                    rid = pc.get(key)
                    if rid is not None:
                        contact_count[rid] = contact_count.get(rid, 0) + 1
            if contact_count:
                max_cc = max(contact_count.values())
                for rid, cnt in contact_count.items():
                    if rid not in scores:
                        scores[rid] = {"total": 0, "sources": []}
                    norm = cnt / (max_cc + 1e-10)
                    scores[rid]["total"] += norm * 0.15
                    if norm > 0.5:
                        scores[rid]["sources"].append(f"Contacts={cnt}")

        if not scores:
            return insights

        ranked = sorted(scores.items(), key=lambda x: -x[1]["total"])
        top = ranked[:10]

        resid_list = [int(r) for r, _ in top]
        evidence = [
            f"Residue {r}: sensitivity={s['total']:.3f} ({'; '.join(s['sources'][:3])})"
            for r, s in top[:5]
        ]

        insights.append({
            "type": "mutation_sensitivity",
            "residues": resid_list,
            "description": f"Mutation sensitivity prediction identifies {len(resid_list)} residues "
                          f"most likely to cause functional disruption if mutated. "
                          f"Top candidates: {', '.join(map(str, resid_list[:5]))}. "
                          f"These residues score highly across multiple metrics: structural "
                          f"centrality (GNN), allosteric importance (network hubs), hydrogen bond "
                          f"participation, burial depth, and contact density. Mutations at these "
                          f"positions are predicted to significantly impact stability and/or function.",
            "confidence": round(min(0.85, 0.55 + 0.03 * len([r for r, s in top if len(s["sources"]) >= 2])), 2),
            "evidence": evidence,
            "category": "structural",
        })

        return insights

    def _predict_stability_changes(self, result) -> List[Dict]:
        """Estimate which mutations would destabilize the protein using energy + contacts + burial."""
        insights = []
        energy_data = result.energy_decomposition
        contact_data = result.contacts
        rmsf_data = result.rmsf
        hbond_data = result.hbonds

        if not isinstance(energy_data, dict) or energy_data.get("error"):
            return insights

        per_res_energy = energy_data.get("per_residue_energy", {})
        if not per_res_energy:
            return insights

        resnames = {}
        rmsf_lookup = {}
        mean_rmsf = 1.0
        if isinstance(rmsf_data, dict) and "resnames" in rmsf_data:
            resnames = dict(zip(
                rmsf_data.get("resids", []),
                rmsf_data.get("resnames", [])
            ))
            rmsf_arr = np.array(rmsf_data.get("rmsf", []))
            rmsf_resids = rmsf_data.get("resids", [])
            rmsf_lookup = dict(zip(rmsf_resids, rmsf_arr))
            mean_rmsf = float(np.mean(rmsf_arr)) if len(rmsf_arr) > 0 else 1.0

        # H-bonds per residue
        hb_count = {}
        if isinstance(hbond_data, dict) and "persistent_hbonds" in hbond_data:
            for hb in hbond_data["persistent_hbonds"]:
                for key in ("donor_resid", "acceptor_resid"):
                    rid = hb.get(key)
                    if rid is not None:
                        hb_count[rid] = hb_count.get(rid, 0) + 1

        # Contacts per residue
        contact_count = {}
        if isinstance(contact_data, dict) and "persistent_contacts" in contact_data:
            for pc in contact_data["persistent_contacts"]:
                for key in ("resid_1", "resid_2"):
                    rid = pc.get(key)
                    if rid is not None:
                        contact_count[rid] = contact_count.get(rid, 0) + 1

        max_abs_e = max((abs(e) for e in per_res_energy.values()), default=1)

        destab_candidates = []
        for rid_str, energy in per_res_energy.items():
            rid = int(rid_str)
            rn = resnames.get(rid, "")[:3].upper()
            f_val = rmsf_lookup.get(rid, 0)

            # Skip surface residues
            if rmsf_lookup and f_val > mean_rmsf * 1.5:
                continue

            n_hb = hb_count.get(rid, 0)
            n_contacts = contact_count.get(rid, 0)

            burial = 1.0 - (f_val / (mean_rmsf * 2 + 1e-10)) if rmsf_lookup else 0.5
            burial = max(0, min(1, burial))

            destab_score = (
                0.3 * (abs(energy) / (max_abs_e + 1e-10)) +
                0.25 * burial +
                0.25 * min(1.0, n_contacts / 10.0) +
                0.2 * min(1.0, n_hb / 5.0)
            )

            if destab_score > 0.4:
                destab_candidates.append({
                    "resid": rid,
                    "resname": rn,
                    "score": destab_score,
                    "energy": energy,
                    "contacts": n_contacts,
                    "hbonds": n_hb,
                    "burial": burial,
                })

        if not destab_candidates:
            return insights

        destab_candidates.sort(key=lambda x: -x["score"])
        top = destab_candidates[:10]
        resid_list = [c["resid"] for c in top]

        evidence = [
            f"{c['resname']}{c['resid']}: ddG-proxy={c['score']:.2f} "
            f"(E={c['energy']:.1f} kJ/mol, contacts={c['contacts']}, "
            f"H-bonds={c['hbonds']}, burial={c['burial']:.2f})"
            for c in top[:5]
        ]

        core_hydrophobic = [c for c in top if c["resname"] in self._HYDROPHOBIC_RESIDUES and c["burial"] > 0.6]

        desc_extra = ""
        if core_hydrophobic:
            desc_extra = (
                f" Notably, buried hydrophobic residues "
                f"{', '.join(c['resname'] + str(c['resid']) for c in core_hydrophobic[:3])} "
                f"are core-packing residues — mutations to polar/charged amino acids at these "
                f"positions would severely destabilize the fold."
            )

        insights.append({
            "type": "stability_change_prediction",
            "residues": resid_list,
            "description": f"Stability change prediction identifies {len(destab_candidates)} residues "
                          f"where mutations are most likely to destabilize the protein (ddG > 0). "
                          f"Highest-risk positions: {', '.join(c['resname'] + str(c['resid']) for c in top[:5])}. "
                          f"These residues are deeply buried, form extensive contacts and hydrogen "
                          f"bonds, and contribute significantly to the total interaction energy.{desc_extra}",
            "confidence": round(min(0.8, 0.5 + 0.05 * len([c for c in top if c["score"] > 0.5])), 2),
            "evidence": evidence,
            "category": "structural",
        })

        return insights

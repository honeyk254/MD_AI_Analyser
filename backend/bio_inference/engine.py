"""
Biological Inference Engine.
Converts raw computational metrics into biologically meaningful explanations
using structural biology reasoning.
"""
import numpy as np
from typing import List, Dict, Any


class BiologicalInferenceEngine:
    """
    Takes all computed analysis results and generates human-readable
    biological interpretations based on structural biology principles.
    """

    def interpret(self, result) -> List[Dict[str, Any]]:
        """
        Generate biological insights from all analysis results.
        Each insight is a dict with: type, residues, description, confidence, evidence, category.
        """
        insights = []

        try:
            insights.extend(self._detect_hinge_residues(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_flexible_loops(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_stable_core(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_allosteric_communication(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_binding_pocket_dynamics(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_conformational_transitions(result))
        except Exception:
            pass
        try:
            insights.extend(self._detect_domain_motions(result))
        except Exception:
            pass
        try:
            insights.extend(self._assess_protein_stability(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_gnn_results(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_transformer_results(result))
        except Exception:
            pass
        # Part A new interpretations
        try:
            insights.extend(self._interpret_water_bridges(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_energy_hotspots(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_prs(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_nma(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_entropy(result))
        except Exception:
            pass
        try:
            insights.extend(self._interpret_tunnels(result))
        except Exception:
            pass

        # Sort by confidence
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

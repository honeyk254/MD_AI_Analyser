/**
 * MD AI Analyzer — Frontend Application
 * Handles file upload, analysis control, results rendering, and 3D viewer.
 */

// ===== State =====
let currentJobId = null;
let analysisResults = null;
let viewer3d = null;
let structureData = null;
let eventSource = null;

// ===== Init =====
document.addEventListener('DOMContentLoaded', async () => {
    initNavigation();
    initDragDrop();
    await checkSystemInfo();
});

// ===== Navigation =====
function initNavigation() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`section-${tab.dataset.section}`).classList.add('active');

            if (tab.dataset.section === 'viewer' && structureData && !viewer3d) {
                setTimeout(() => initViewer(), 100);
            }
        });
    });
}

function switchSection(name) {
    document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.section === name);
    });
    document.querySelectorAll('.section').forEach(s => {
        s.classList.toggle('active', s.id === `section-${name}`);
    });
}

// ===== System Info =====
async function checkSystemInfo() {
    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        if (data.system?.gpu_available) {
            const badge = document.getElementById('gpu-badge');
            badge.textContent = `GPU: ${data.system.gpu_name}`;
            badge.style.display = 'inline-flex';
        }
    } catch (e) {
        console.log('Health check failed:', e);
    }
}

// ===== Drag & Drop =====
function initDragDrop() {
    const zone = document.getElementById('upload-zone');
    ['dragenter', 'dragover'].forEach(evt => {
        zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.remove('dragover'); });
    });
    zone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        for (const f of files) {
            const ext = f.name.split('.').pop().toLowerCase();
            if (['xtc', 'trr'].includes(ext)) document.getElementById('file-trajectory').files = createFileList(f);
            else if (ext === 'tpr') document.getElementById('file-topology').files = createFileList(f);
            else if (['pdb', 'gro'].includes(ext)) document.getElementById('file-structure').files = createFileList(f);
        }
    });
}

function createFileList(file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    return dt.files;
}

// ===== Upload & Analyze =====
async function uploadAndAnalyze() {
    const btn = document.getElementById('btn-upload');
    btn.disabled = true;
    btn.textContent = '⏳ Uploading...';

    const formData = new FormData();
    const trajectory = document.getElementById('file-trajectory').files[0];
    const topology = document.getElementById('file-topology').files[0];
    const structure = document.getElementById('file-structure').files[0];
    const reference = document.getElementById('file-reference').files[0];

    if (!structure) {
        alert('Please provide at least a structure file (.pdb or .gro)');
        btn.disabled = false;
        btn.textContent = '🚀 Upload & Analyze';
        return;
    }

    if (trajectory) formData.append('trajectory', trajectory);
    if (topology) formData.append('topology', topology);
    if (structure) formData.append('structure', structure);
    if (reference) formData.append('reference', reference);

    try {
        // Upload
        const uploadResp = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!uploadResp.ok) {
            const err = await uploadResp.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const uploadData = await uploadResp.json();
        currentJobId = uploadData.job_id;

        // Switch to progress
        switchSection('progress');
        addLog('Files uploaded successfully');

        // Start analysis — include configurable params
        const startFrameVal = document.getElementById('opt-start-frame')?.value;
        const endFrameVal = document.getElementById('opt-end-frame')?.value;

        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: currentJobId,
                stride: 1,
                run_gnn: document.getElementById('opt-gnn').checked,
                run_transformer: document.getElementById('opt-transformer').checked,
                run_msm: document.getElementById('opt-msm').checked,
                ligand_selection: document.getElementById('opt-ligand').value || null,
                start_frame: startFrameVal ? parseInt(startFrameVal) : null,
                end_frame: endFrameVal ? parseInt(endFrameVal) : null,
                hbond_cutoff: parseFloat(document.getElementById('opt-hbond-cutoff')?.value) || 3.5,
                contact_cutoff: parseFloat(document.getElementById('opt-contact-cutoff')?.value) || 8.0,
                salt_bridge_cutoff: parseFloat(document.getElementById('opt-salt-cutoff')?.value) || 4.0,
                fel_bins: parseInt(document.getElementById('opt-fel-bins')?.value) || 50,
                temperature: parseFloat(document.getElementById('opt-temperature')?.value) || 300.0,
                msm_lag_time: parseInt(document.getElementById('opt-msm-lag')?.value) || 5,
                grid_spacing: parseFloat(document.getElementById('opt-grid-spacing')?.value) || 2.0,
                correlation_threshold: parseFloat(document.getElementById('opt-corr-threshold')?.value) || 0.5,
                vae_latent_dim: parseInt(document.getElementById('opt-vae-dim')?.value) || 2,
            }),
        });

        if (!analyzeResp.ok) throw new Error('Failed to start analysis');
        addLog('Analysis pipeline started');

        // Connect SSE
        connectSSE(currentJobId);

    } catch (e) {
        alert('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = '🚀 Upload & Analyze';
    }
}

// ===== SSE Progress =====
function connectSSE(jobId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/api/progress/${jobId}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.heartbeat) return;

        const pct = data.progress_percent || 0;
        document.getElementById('progress-bar').style.width = `${pct}%`;
        document.getElementById('progress-percent').textContent = `${Math.round(pct)}%`;
        document.getElementById('progress-module').textContent = data.current_module || '';
        document.getElementById('progress-message').textContent = data.message || '';

        addLog(`[${data.current_module}] ${data.message}`);

        if (data.status === 'completed') {
            eventSource.close();
            addLog('✅ Analysis complete!');
            fetchResults(jobId);
        } else if (data.status === 'failed') {
            eventSource.close();
            addLog('❌ Analysis failed: ' + data.message);
        }
    };

    eventSource.onerror = () => {
        addLog('SSE connection lost, polling for results...');
        eventSource.close();
        setTimeout(() => pollResults(jobId), 2000);
    };
}

function addLog(message) {
    const log = document.getElementById('progress-log');
    const time = new Date().toLocaleTimeString();
    log.innerHTML += `<div class="log-entry"><span class="timestamp">${time}</span>${message}</div>`;
    log.scrollTop = log.scrollHeight;
}

async function pollResults(jobId) {
    try {
        const resp = await fetch(`/api/results/${jobId}`);
        const data = await resp.json();
        if (data.status === 'completed') {
            analysisResults = data;
            renderResults();
        } else if (data.status === 'running') {
            setTimeout(() => pollResults(jobId), 3000);
        }
    } catch (e) {
        console.error('Poll failed:', e);
    }
}

// ===== Fetch & Render Results =====
async function fetchResults(jobId) {
    try {
        const resp = await fetch(`/api/results/${jobId}`);
        analysisResults = await resp.json();
        renderResults();

        // Also fetch structure for 3D viewer
        try {
            const structResp = await fetch(`/api/structure/${jobId}`);
            structureData = await structResp.text();
        } catch (e) {
            console.log('No structure file for viewer');
        }
    } catch (e) {
        console.error('Failed to fetch results:', e);
    }
}

function renderResults() {
    renderStats();
    renderPlots();
    renderInsights();
    switchSection('results');
}

// ===== Stats Grid =====
function renderStats() {
    const info = analysisResults.trajectory_info || {};
    const rmsd = analysisResults.rmsd || {};
    const rmsf = analysisResults.rmsf || {};
    const rg = analysisResults.rg || {};
    const clustering = analysisResults.clustering || {};

    const convergence = analysisResults.convergence || {};
    const bk = analysisResults.binding_kinetics || {};

    const stats = [
        { value: info.n_frames || 'N/A', label: 'Frames' },
        { value: info.n_atoms || 'N/A', label: 'Atoms' },
        { value: info.n_residues || 'N/A', label: 'Residues' },
        { value: info.total_time_ns ? `${info.total_time_ns.toFixed(1)} ns` : 'N/A', label: 'Sim Time' },
        { value: rmsd.mean_rmsd ? `${rmsd.mean_rmsd.toFixed(2)} Å` : 'N/A', label: 'Mean RMSD' },
        { value: rmsf.mean_rmsf ? `${rmsf.mean_rmsf.toFixed(2)} Å` : 'N/A', label: 'Mean RMSF' },
        { value: rg.mean_rg ? `${rg.mean_rg.toFixed(1)} Å` : 'N/A', label: 'Mean Rg' },
        { value: clustering.n_clusters || 'N/A', label: 'Clusters' },
        { value: convergence.convergence_score != null ? `${(convergence.convergence_score * 100).toFixed(0)}%` : 'N/A', label: 'Convergence' },
        { value: bk.total_contact_fraction != null ? `${(bk.total_contact_fraction * 100).toFixed(0)}%` : 'N/A', label: 'Contact Frac.' },
    ];

    document.getElementById('stats-grid').innerHTML = stats.map(s =>
        `<div class="stat-card"><div class="stat-value">${s.value}</div><div class="stat-label">${s.label}</div></div>`
    ).join('');
}

// ===== Plots =====
const plotNameMap = {
    rmsd_plot: 'RMSD', rmsf_plot: 'RMSF', rg_plot: 'Rg', ss_plot: 'Sec. Struct.',
    hbond_plot: 'H-Bonds', salt_bridges_plot: 'Salt Bridges', contact_map: 'Contacts', pca_plot: 'PCA',
    dccm_plot: 'DCCM', fel_plot: 'Free Energy', clustering_plot: 'Clusters',
    sasa_plot: 'SASA', dimensionality_plot: 'Dim. Reduction',
    dimensionality_3d_plot: '3D Projections',
    gnn_plot: 'GNN', transformer_plot: 'Transformer', msm_plot: 'MSM',
    // Part A
    tica_plot: 'tICA', water_bridges_plot: 'Water Bridges', energy_plot: 'Energy Decomp.',
    prs_plot: 'PRS', nma_plot: 'Normal Modes', entropy_plot: 'Entropy',
    ifp_plot: 'Interaction FP', tunnel_plot: 'Tunnels/Cavities',
    vae_plot: 'VAE Latent', dynamic_network_plot: 'Dynamic Network',
    // Phase 4
    convergence_plot: 'Convergence', binding_kinetics_plot: 'Binding Kinetics',
    network_graph_plot: 'Allosteric Network', training_loss_plot: 'Training Losses',
};

function renderPlots() {
    const plots = analysisResults.plots || {};
    const plotKeys = Object.keys(plots).filter(k => !['report_html', 'csv_metrics'].includes(k));

    const tabsEl = document.getElementById('plot-tabs');
    tabsEl.innerHTML = plotKeys.map((key, i) =>
        `<button class="plot-tab ${i === 0 ? 'active' : ''}" data-plot="${key}">${plotNameMap[key] || key}</button>`
    ).join('');

    tabsEl.querySelectorAll('.plot-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            tabsEl.querySelectorAll('.plot-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            showPlot(tab.dataset.plot);
        });
    });

    if (plotKeys.length > 0) showPlot(plotKeys[0]);
}

function showPlot(key) {
    const container = document.getElementById('plot-container');
    const plots = analysisResults.plots || {};
    const jsonStr = plots[key];
    if (!jsonStr) {
        container.innerHTML = '<p style="padding:40px;text-align:center;color:#666;">No data for this plot.</p>';
        return;
    }
    try {
        const fig = JSON.parse(jsonStr);
        Plotly.newPlot(container, fig.data, {
            ...fig.layout,
            autosize: true,
            height: 450,
        }, { responsive: true });
    } catch (e) {
        container.innerHTML = `<p style="padding:40px;color:#ff6b6b;">Error rendering plot: ${e.message}</p>`;
    }
}

// ===== Insights =====
const categoryColors = {
    structural: '#55efc4', dynamic: '#a29bfe', allosteric: '#ff6b6b',
    binding: '#ffd93d', transition: '#fd79a8',
};

const categoryLabels = {
    structural: 'Structural', dynamic: 'Dynamic', allosteric: 'Allosteric',
    binding: 'Binding', transition: 'Transition',
};

const insightTypeLabels = {
    // Original
    hinge_residue: 'Hinge Residue', flexible_loop: 'Flexible Loop', stable_core: 'Stable Core',
    allosteric_pathway: 'Allosteric Pathway', communication_hub: 'Communication Hub',
    binding_pocket: 'Binding Pocket', conformational_states: 'Conformational States',
    metastable_kinetics: 'Metastable Kinetics', domain_motion: 'Domain Motion',
    stability_assessment: 'Stability Assessment', gnn_key_residues: 'GNN Key Residues',
    transformer_transitions: 'Transformer Transitions',
    // Part A
    water_bridge_sites: 'Water Bridge Sites', energy_hotspot: 'Energy Hotspot',
    prs_effectors_sensors: 'PRS Effectors/Sensors', nma_collective_motion: 'NMA Collective Motion',
    entropy_estimate: 'Entropy Estimate', cavity_channels: 'Cavity Channels',
    dynamic_network_evolution: 'Dynamic Network', vae_conformational_landscape: 'VAE Landscape',
    interaction_fingerprint: 'Interaction Fingerprint',
    // Part B — New
    breathing_motion: 'Breathing Motion', cracking_event: 'Cracking / Local Unfolding',
    cryptic_binding_site: 'Cryptic Binding Site', druggability_score: 'Druggability Score',
    ptm_site_prediction: 'PTM Site Prediction', ppi_interface_hotspot: 'PPI Interface Hotspot',
    interface_conformational_selection: 'Conformational Selection',
    protonation_dynamics: 'Protonation Dynamics', electrostatic_funnel: 'Electrostatic Funnel',
    aggregation_prone_region: 'Aggregation-Prone Region', folding_intermediate: 'Folding Intermediate',
    functional_motion_classification: 'Functional Motion Type', motion_function_coupling: 'Motion-Function Coupling',
    hbond_network_rewiring: 'H-Bond Network Rewiring', structural_waters: 'Structural Waters',
    local_stiffness_map: 'Local Stiffness Map', force_propagation_pathway: 'Force Propagation',
    mutation_sensitivity: 'Mutation Sensitivity', stability_change_prediction: 'Stability Change (ddG)',
};

let activeInsightFilter = 'all';

function renderInsights() {
    const insights = analysisResults.biological_insights || [];
    const wrapper = document.getElementById('insights-list');

    if (insights.length === 0) {
        wrapper.innerHTML = '<p style="color:#666;padding:20px;">No biological insights generated. This may indicate the trajectory is too short or lacks protein atoms.</p>';
        return;
    }

    // Count per category
    const counts = { all: insights.length };
    insights.forEach(ins => {
        const cat = ins.category || 'structural';
        counts[cat] = (counts[cat] || 0) + 1;
    });

    // Build filter bar + cards container
    const categories = ['all', 'structural', 'dynamic', 'allosteric', 'binding', 'transition'];
    const filterBar = categories
        .filter(c => c === 'all' || counts[c])
        .map(c => {
            const label = c === 'all' ? 'All' : categoryLabels[c] || c;
            const count = counts[c] || 0;
            const color = c === 'all' ? 'var(--accent-cyan)' : categoryColors[c] || '#00d4ff';
            return `<button class="insight-filter-btn ${activeInsightFilter === c ? 'active' : ''}" data-filter="${c}" style="--filter-color:${color}">${label} <span class="filter-count">${count}</span></button>`;
        }).join('');

    const cards = insights.map(ins => {
        const cat = ins.category || 'structural';
        const color = categoryColors[cat] || '#00d4ff';
        const confPct = Math.round((ins.confidence || 0) * 100);
        const evidenceItems = (ins.evidence || []).map(e => `<li>${e}</li>`).join('');
        const residueStr = (ins.residues || []).slice(0, 15).join(', ');
        const displayType = insightTypeLabels[ins.type] || (ins.type || '').replace(/_/g, ' ');
        const hidden = activeInsightFilter !== 'all' && cat !== activeInsightFilter;

        return `
        <div class="insight-card cat-${cat}" data-category="${cat}" ${hidden ? 'style="display:none;"' : ''}>
            <div class="insight-header">
                <span class="insight-type">${displayType}</span>
                <span class="confidence">${confPct}%</span>
            </div>
            <div class="confidence-bar-mini">
                <div class="confidence-fill-mini" style="width:${confPct}%;background:${color};"></div>
            </div>
            <p class="description">${ins.description || ''}</p>
            ${residueStr ? `<div class="residues">Residues: ${residueStr}</div>` : ''}
            ${evidenceItems ? `<details><summary>Evidence</summary><ul>${evidenceItems}</ul></details>` : ''}
        </div>`;
    }).join('');

    wrapper.innerHTML = `<div class="insight-filters">${filterBar}</div><div class="insight-cards">${cards}</div>`;

    // Bind filter clicks
    wrapper.querySelectorAll('.insight-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            activeInsightFilter = btn.dataset.filter;
            wrapper.querySelectorAll('.insight-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            wrapper.querySelectorAll('.insight-card').forEach(card => {
                if (activeInsightFilter === 'all' || card.dataset.category === activeInsightFilter) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
}

// ===== 3D Viewer =====
function initViewer() {
    if (!structureData) return;

    const container = document.getElementById('viewer-3d');
    container.innerHTML = '';

    viewer3d = $3Dmol.createViewer(container, {
        backgroundColor: '#0a0a14',
        antialias: true,
    });

    // Detect format from extension
    const format = structureData.includes('ATOM') || structureData.includes('HETATM') ? 'pdb' : 'gro';
    viewer3d.addModel(structureData, format);
    viewer3d.setStyle({}, { cartoon: { color: 'spectrum' } });
    viewer3d.zoomTo();
    viewer3d.render();
}

function updateViewerStyle() {
    if (!viewer3d) return;
    const style = document.getElementById('viewer-style').value;
    const colorScheme = document.getElementById('viewer-color').value;
    const colorObj = getColorObject(colorScheme);

    viewer3d.setStyle({}, { [style]: colorObj });
    viewer3d.render();
}

function updateViewerColor() {
    updateViewerStyle();
}

function getColorObject(scheme) {
    switch (scheme) {
        case 'spectrum': return { color: 'spectrum' };
        case 'chain': return { colorscheme: 'chain' };
        case 'ss': return { colorscheme: 'ssJmol' };
        case 'rmsf':
            if (analysisResults?.rmsf?.rmsf) {
                return { colorscheme: { prop: 'b', gradient: 'rwb', min: 0, max: 5 } };
            }
            return { color: 'spectrum' };
        default: return { color: 'spectrum' };
    }
}

function updateViewerBg() {
    if (!viewer3d) return;
    const bg = document.getElementById('viewer-bg').value;
    const colors = { black: '#000000', dark: '#0a0a14', white: '#ffffff' };
    viewer3d.setBackgroundColor(colors[bg] || '#0a0a14');
    viewer3d.render();
}

function highlightResidues() {
    if (!viewer3d || !analysisResults) return;

    const option = document.getElementById('viewer-highlight').value;
    const style = document.getElementById('viewer-style').value;
    const colorObj = getColorObject(document.getElementById('viewer-color').value);

    // Reset
    viewer3d.setStyle({}, { [style]: colorObj });

    let residues = [];
    let highlightColor = '#ff6b6b';

    switch (option) {
        case 'flexible':
            residues = analysisResults.rmsf?.high_flexibility_residues || [];
            highlightColor = '#ff6b6b';
            break;
        case 'hinge':
            residues = getInsightResidues('hinge_residue');
            highlightColor = '#e17055';
            break;
        case 'stable_core':
            residues = getInsightResidues('stable_core');
            highlightColor = '#55efc4';
            break;
        case 'stiffness':
            residues = getInsightResidues('local_stiffness_map');
            highlightColor = '#00b894';
            break;
        case 'hubs':
            residues = (analysisResults.allosteric?.hub_residues || []).map(h => h.resid);
            highlightColor = '#ff6b6b';
            break;
        case 'gnn':
            residues = (analysisResults.gnn_results?.top_residues || []).map(r => r.resid);
            highlightColor = '#a29bfe';
            break;
        case 'mutation':
            residues = getInsightResidues('mutation_sensitivity');
            highlightColor = '#d63031';
            break;
        case 'ptm':
            residues = getInsightResidues('ptm_site_prediction');
            highlightColor = '#fdcb6e';
            break;
        case 'cryptic':
            residues = getInsightResidues('cryptic_binding_site');
            highlightColor = '#ffd93d';
            break;
        case 'ppi':
            residues = getInsightResidues('ppi_interface_hotspot');
            highlightColor = '#fab1a0';
            break;
        case 'druggability':
            residues = getInsightResidues('druggability_score');
            highlightColor = '#00cec9';
            break;
        case 'aggregation':
            residues = getInsightResidues('aggregation_prone_region');
            highlightColor = '#e17055';
            break;
        case 'electrostatic':
            residues = getInsightResidues('electrostatic_funnel');
            highlightColor = '#74b9ff';
            break;
    }

    if (residues.length > 0) {
        viewer3d.setStyle({ resi: residues }, {
            [style]: colorObj,
            sphere: { radius: 0.8, color: highlightColor, opacity: 0.7 },
        });
    }

    viewer3d.render();
}

function getInsightResidues(insightType) {
    const insights = analysisResults.biological_insights || [];
    return insights
        .filter(i => i.type === insightType)
        .flatMap(i => i.residues || []);
}

// ===== Downloads =====
function downloadReport(type) {
    if (!currentJobId) {
        alert('No analysis results available');
        return;
    }

    switch (type) {
        case 'html':
            window.open(`/api/report/${currentJobId}`, '_blank');
            break;
        case 'csv':
            window.open(`/api/csv/${currentJobId}`, '_blank');
            break;
        case 'pdf':
            window.open(`/api/pdf/${currentJobId}`, '_blank');
            break;
        case 'json':
            if (analysisResults) {
                const blob = new Blob([JSON.stringify(analysisResults, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `md_analysis_${currentJobId}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }
            break;
    }
}

/**
 * MD AI Analyzer — Frontend Application v2
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
    initFileInputListeners();
    injectProgressGradient();
    await checkSystemInfo();
    loadHistory();
});

// ===== Toast Notification System =====
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');

    const icons = {
        success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>',
        error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        warning: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `${icons[type] || icons.info}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        toast.addEventListener('animationend', () => toast.remove());
    }, duration);
}

// ===== SVG gradient for progress ring =====
function injectProgressGradient() {
    const svg = document.querySelector('.progress-ring');
    if (!svg) return;
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', 'progress-gradient');
    gradient.setAttribute('x1', '0%');
    gradient.setAttribute('y1', '0%');
    gradient.setAttribute('x2', '100%');
    gradient.setAttribute('y2', '100%');

    const stops = [
        { offset: '0%', color: '#00d4ff' },
        { offset: '50%', color: '#a78bfa' },
        { offset: '100%', color: '#f472b6' },
    ];

    stops.forEach(s => {
        const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop.setAttribute('offset', s.offset);
        stop.setAttribute('stop-color', s.color);
        gradient.appendChild(stop);
    });

    defs.appendChild(gradient);
    svg.insertBefore(defs, svg.firstChild);
}

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
            document.getElementById('gpu-badge-text').textContent = `GPU: ${data.system.gpu_name}`;
            badge.style.display = 'inline-flex';
        }
    } catch (e) {
        console.log('Health check failed:', e);
    }
}

// ===== File Input Listeners =====
function initFileInputListeners() {
    const fileInputs = [
        { id: 'file-trajectory', statusId: 'status-trajectory', groupId: 'fg-trajectory' },
        { id: 'file-topology', statusId: 'status-topology', groupId: 'fg-topology' },
        { id: 'file-structure', statusId: 'status-structure', groupId: 'fg-structure' },
        { id: 'file-reference', statusId: 'status-reference', groupId: 'fg-reference' },
    ];

    fileInputs.forEach(({ id, statusId, groupId }) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('change', () => {
            const status = document.getElementById(statusId);
            const group = document.getElementById(groupId);
            if (input.files.length > 0) {
                const name = input.files[0].name;
                const size = formatFileSize(input.files[0].size);
                status.textContent = `${name} (${size})`;
                group.classList.add('has-file');
            } else {
                status.textContent = '';
                group.classList.remove('has-file');
            }
        });
    });
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
    return (bytes / 1073741824).toFixed(2) + ' GB';
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
            if (['xtc', 'trr'].includes(ext)) assignFile('file-trajectory', f);
            else if (ext === 'tpr') assignFile('file-topology', f);
            else if (['pdb', 'gro'].includes(ext)) assignFile('file-structure', f);
        }
        showToast('Files detected and assigned', 'success');
    });
}

function assignFile(inputId, file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    const input = document.getElementById(inputId);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
}

// ===== Upload & Analyze =====
async function uploadAndAnalyze() {
    const btn = document.getElementById('btn-upload');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner" style="width:20px;height:20px;border-width:2px;margin:0;"></div> Uploading...';

    const formData = new FormData();
    const trajectory = document.getElementById('file-trajectory').files[0];
    const topology = document.getElementById('file-topology').files[0];
    const structure = document.getElementById('file-structure').files[0];
    const reference = document.getElementById('file-reference').files[0];

    if (!structure) {
        showToast('Please provide at least a structure file (.pdb or .gro)', 'warning');
        btn.disabled = false;
        btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2Z"/></svg> Upload & Analyze';
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
        showToast('Files uploaded successfully', 'success');
        addLog('Files uploaded successfully');

        // Start analysis
        const startFrameVal = document.getElementById('opt-start-frame')?.value;
        const endFrameVal = document.getElementById('opt-end-frame')?.value;
        const strideVal = document.getElementById('opt-stride')?.value;

        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                job_id: currentJobId,
                stride: strideVal ? parseInt(strideVal) : 1,
                run_gnn: document.getElementById('opt-gnn').checked,
                run_transformer: document.getElementById('opt-transformer').checked,
                run_msm: document.getElementById('opt-msm').checked,
                discard_equilibration: document.getElementById('opt-discard-equil').checked,
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
        showToast('Analysis pipeline started', 'info');
        addLog('Analysis pipeline started');

        connectSSE(currentJobId);

    } catch (e) {
        showToast('Error: ' + e.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2Z"/></svg> Upload & Analyze';
    }
}

// ===== Progress Ring Update =====
function updateProgressRing(pct) {
    const circle = document.getElementById('progress-ring-fill');
    const text = document.getElementById('progress-ring-text');
    if (!circle || !text) return;

    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (pct / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    text.textContent = `${Math.round(pct)}%`;
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
        document.getElementById('progress-module').textContent = data.current_module || '';
        document.getElementById('progress-message').textContent = data.message || '';
        updateProgressRing(pct);

        addLog(`[${data.current_module}] ${data.message}`);

        if (data.status === 'completed') {
            eventSource.close();
            showToast('Analysis complete!', 'success', 6000);
            addLog('Analysis complete!');
            fetchResults(jobId);
            loadHistory();
        } else if (data.status === 'failed') {
            eventSource.close();
            showToast('Analysis failed: ' + data.message, 'error', 8000);
            addLog('Analysis failed: ' + data.message);
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

        try {
            const structResp = await fetch(`/api/structure/${jobId}`);
            structureData = await structResp.text();
        } catch (e) {
            console.log('No structure file for viewer');
        }
    } catch (e) {
        console.error('Failed to fetch results:', e);
        showToast('Failed to fetch results', 'error');
    }
}

// ===== History =====
async function loadHistory() {
    try {
        const resp = await fetch('/api/jobs');
        const jobs = await resp.json();
        renderHistory(jobs);
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderHistory(jobs) {
    const list = document.getElementById('history-list');
    if (!jobs || jobs.length === 0) {
        list.innerHTML = '<div class="history-empty">No previous runs found.</div>';
        return;
    }
    list.innerHTML = jobs.map(job => {
        const date = job.created_at ? new Date(job.created_at * 1000).toLocaleString() : '—';
        const info = job.trajectory_info || {};
        const isLegacy = !Object.keys(job.files || {}).length && !info.n_frames;
        const statusClass = job.status === 'completed' ? 'history-status-ok'
            : job.status === 'failed' ? 'history-status-err' : 'history-status-run';
        const fileNames = Object.values(job.files || {}).filter(Boolean);
        const filesLabel = isLegacy ? '<em>report only</em>' : (fileNames.length ? fileNames.join(', ') : '—');
        const parts = [];
        if (info.n_frames) parts.push(`${info.n_frames} frames`);
        if (info.n_residues) parts.push(`${info.n_residues} residues`);
        if (info.total_time_ns) parts.push(`${Number(info.total_time_ns).toFixed(1)} ns`);
        const meta = parts.join(' · ');
        return `<div class="history-item" onclick="loadHistoryJob('${job.job_id}')">
            <div class="history-item-left">
                <span class="history-status ${statusClass}">${job.status}</span>
                <div class="history-item-id">Job ${job.job_id}</div>
                <div class="history-item-files">${filesLabel}</div>
            </div>
            <div class="history-item-right">
                ${meta ? `<div class="history-item-meta">${meta}</div>` : ''}
                <div class="history-item-date">${date}</div>
            </div>
        </div>`;
    }).join('');
}

async function loadHistoryJob(jobId) {
    currentJobId = jobId;
    showToast(`Loading run ${jobId}…`, 'info');

    try {
        const resp = await fetch(`/api/results/${jobId}`);
        const data = await resp.json();

        // Legacy stub — report exists on disk but no full result was persisted
        if (!data.plots || Object.keys(data.plots).length === 0) {
            showToast('Legacy run — full results unavailable. HTML report and CSV are still downloadable.', 'warning', 6000);
            switchSection('downloads');
            return;
        }

        analysisResults = data;
        renderResults();

        try {
            const structResp = await fetch(`/api/structure/${jobId}`);
            structureData = await structResp.text();
        } catch (_) {}

        switchSection('results');
    } catch (e) {
        console.error('Failed to load history job:', e);
        showToast('Failed to load run ' + jobId, 'error');
    }
}

function renderResults() {
    renderStats();
    renderCaveats();
    renderPlots();
    renderInsights();
    switchSection('results');
}

// ===== Methodology Caveats =====
function renderCaveats() {
    const existing = document.getElementById('caveats-section');
    if (existing) existing.remove();

    const notes = [];
    const bk = analysisResults.binding_kinetics || {};
    const ed = analysisResults.energy_decomposition || {};
    const msm = analysisResults.msm || {};
    const gnn = analysisResults.gnn_results || {};
    const transformer = analysisResults.transformer_results || {};
    const vae = analysisResults.vae || {};
    const tunnels = analysisResults.tunnels || {};

    if (bk.kinetics_caveat) notes.push({ label: 'Binding Kinetics', text: bk.kinetics_caveat });
    if (ed.caveat) notes.push({ label: 'Interaction Scores', text: ed.caveat });
    if (msm.caveat) notes.push({ label: 'MSM', text: msm.caveat });
    if (gnn.caveat) notes.push({ label: 'GNN', text: gnn.caveat });
    if (transformer.caveat) notes.push({ label: 'Transformer', text: transformer.caveat });
    if (vae.caveat) notes.push({ label: 'VAE', text: vae.caveat });
    if (tunnels.caveat) notes.push({ label: 'Cavities', text: tunnels.caveat });

    if (notes.length === 0) return;

    const section = document.createElement('div');
    section.id = 'caveats-section';
    section.className = 'card';
    section.style.cssText = 'border-color: rgba(251,191,36,0.3);';
    section.innerHTML = `
        <div class="card-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-yellow)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <h2 style="font-size:1rem;">Methodology Notes</h2>
        </div>
        <div style="display:flex;flex-direction:column;gap:10px;padding:0 4px 8px;">
            ${notes.map(n => `
                <div style="font-size:0.82rem;color:var(--text-muted);border-left:3px solid rgba(251,191,36,0.4);padding-left:12px;">
                    <strong style="color:var(--accent-yellow);font-size:0.8rem;">${n.label}:</strong> ${n.text}
                </div>
            `).join('')}
        </div>`;

    const statsCard = document.querySelector('#section-results .card');
    if (statsCard && statsCard.parentNode) {
        statsCard.parentNode.insertBefore(section, statsCard.nextSibling);
    }
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
    const msm = analysisResults.msm || {};
    const ck = msm.chapman_kolmogorov || {};
    const displayedTime = info.subset_applied ? info.analyzed_time_ns : info.total_time_ns;
    const timeLabel = info.subset_applied ? 'Analyzed Time' : 'Sim Time';

    const markovianVal = msm.is_markovian != null
        ? (msm.is_markovian ? 'Usable' : 'Exploratory')
        : 'N/A';

    const stats = [
        { value: info.n_frames || 'N/A', label: 'Frames' },
        { value: info.n_atoms || 'N/A', label: 'Atoms' },
        { value: info.n_residues || 'N/A', label: 'Residues' },
        { value: displayedTime ? `${displayedTime.toFixed(1)} ns` : 'N/A', label: timeLabel },
        { value: rmsd.mean_rmsd ? `${rmsd.mean_rmsd.toFixed(2)} \u00c5` : 'N/A', label: 'Mean RMSD' },
        { value: rmsf.mean_rmsf ? `${rmsf.mean_rmsf.toFixed(2)} \u00c5` : 'N/A', label: 'Mean RMSF' },
        { value: rg.mean_rg ? `${rg.mean_rg.toFixed(1)} \u00c5` : 'N/A', label: 'Mean Rg' },
        { value: clustering.n_clusters || 'N/A', label: 'Clusters' },
        { value: convergence.convergence_score != null ? `${(convergence.convergence_score * 100).toFixed(0)}%` : 'N/A', label: 'Convergence' },
        { value: bk.total_contact_fraction != null ? `${(bk.total_contact_fraction * 100).toFixed(0)}%` : 'N/A', label: 'Contact Frac.' },
        { value: markovianVal, label: 'MSM Quality' },
    ];

    const grid = document.getElementById('stats-grid');
    grid.innerHTML = stats.map((s, i) =>
        `<div class="stat-card" style="animation-delay:${i * 50}ms">
            <div class="stat-value">${s.value}</div>
            <div class="stat-label">${s.label}</div>
        </div>`
    ).join('');

    // Stagger animation
    grid.querySelectorAll('.stat-card').forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(8px)';
        setTimeout(() => {
            card.style.transition = 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, i * 60);
    });
}

// ===== Plots =====
const plotNameMap = {
    rmsd_plot: 'RMSD', rmsf_plot: 'RMSF', rg_plot: 'Rg', ss_plot: 'Sec. Struct.',
    hbond_plot: 'H-Bonds', salt_bridges_plot: 'Salt Bridges', contact_map: 'Contacts', pca_plot: 'PCA',
    dccm_plot: 'DCCM', fel_plot: 'Free Energy', clustering_plot: 'Clusters',
    sasa_plot: 'SASA', dimensionality_plot: 'Dim. Reduction',
    dimensionality_3d_plot: '3D Projections',
    gnn_plot: 'GNN Ranking', transformer_plot: 'Transformer', msm_plot: 'MSM',
    tica_plot: 'tICA', water_bridges_plot: 'Water Bridges', energy_plot: 'Interaction Scores',
    prs_plot: 'PRS', nma_plot: 'Normal Modes', entropy_plot: 'Entropy',
    ifp_plot: 'Interaction FP', tunnel_plot: 'Cavities/Voids',
    vae_plot: 'VAE Latent', dynamic_network_plot: 'Dynamic Network',
    convergence_plot: 'Convergence', binding_kinetics_plot: 'Binding Persistence',
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
        container.innerHTML = '<p style="padding:40px;text-align:center;color:var(--text-muted);">No data for this plot.</p>';
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
        container.innerHTML = `<p style="padding:40px;color:var(--accent-red);">Error rendering plot: ${e.message}</p>`;
    }
}

// ===== Insights =====
const categoryColors = {
    structural: '#34d399', dynamic: '#a78bfa', allosteric: '#f87171',
    binding: '#fbbf24', transition: '#f472b6',
};

const categoryLabels = {
    structural: 'Structural', dynamic: 'Dynamic', allosteric: 'Allosteric',
    binding: 'Binding', transition: 'Transition',
};

const insightTypeLabels = {
    hinge_residue: 'Hinge Residue', flexible_loop: 'Flexible Loop', stable_core: 'Stable Core',
    allosteric_pathway: 'Allosteric Pathway', communication_hub: 'Communication Hub',
    binding_pocket: 'Binding Pocket', conformational_states: 'Conformational States',
    metastable_kinetics: 'Exploratory MSM Kinetics', domain_motion: 'Domain Motion',
    stability_assessment: 'Stability Assessment', gnn_key_residues: 'GNN Topological Outliers',
    transformer_transitions: 'Temporal Change-Points',
    water_bridge_sites: 'Water Bridge Sites', energy_hotspot: 'Interaction Hotspot',
    prs_effectors_sensors: 'PRS Effectors/Sensors', nma_collective_motion: 'NMA Collective Motion',
    entropy_estimate: 'Entropy Estimate', cavity_channels: 'Cavity Channels',
    dynamic_network_evolution: 'Dynamic Network', vae_conformational_landscape: 'VAE Landscape',
    interaction_fingerprint: 'Interaction Fingerprint',
    breathing_motion: 'Breathing Motion', cracking_event: 'Cracking / Local Unfolding',
    cryptic_binding_site: 'Cryptic Binding Site', druggability_score: 'Druggability Heuristic',
    ptm_site_prediction: 'PTM Accessibility Screen', ppi_interface_hotspot: 'PPI Interface Hotspot',
    interface_conformational_selection: 'Conformational Selection',
    protonation_dynamics: 'Protonation Dynamics', electrostatic_funnel: 'Electrostatic Funnel',
    aggregation_prone_region: 'Aggregation-Prone Region', folding_intermediate: 'Folding Intermediate',
    functional_motion_classification: 'Functional Motion Type', motion_function_coupling: 'Motion-Function Coupling',
    hbond_network_rewiring: 'H-Bond Network Rewiring', structural_waters: 'Structural Waters',
    local_stiffness_map: 'Local Stiffness Map', force_propagation_pathway: 'Force Propagation',
    mutation_sensitivity: 'Mutation Risk Heuristic', stability_change_prediction: 'Stability Risk Heuristic',
};

let activeInsightFilter = 'all';

function renderInsights() {
    const insights = analysisResults.biological_insights || [];
    const wrapper = document.getElementById('insights-list');

    if (insights.length === 0) {
        wrapper.innerHTML = '<p style="color:var(--text-muted);padding:20px;">No biological insights generated. This may indicate the trajectory is too short or lacks protein atoms.</p>';
        return;
    }

    const counts = { all: insights.length };
    insights.forEach(ins => {
        const cat = ins.category || 'structural';
        counts[cat] = (counts[cat] || 0) + 1;
    });

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

    wrapper.innerHTML = `
        <div style="color:var(--text-muted);font-size:0.82rem;margin-bottom:12px;">
            Insight scores are heuristic detector scores, not calibrated probabilities.
        </div>
        <div class="insight-filters">${filterBar}</div>
        <div class="insight-cards">${cards}</div>
    `;

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
    const placeholder = document.getElementById('viewer-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    viewer3d = $3Dmol.createViewer(container, {
        backgroundColor: '#060610',
        antialias: true,
    });

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
    const colors = { black: '#000000', dark: '#060610', white: '#ffffff' };
    viewer3d.setBackgroundColor(colors[bg] || '#060610');
    viewer3d.render();
}

function highlightResidues() {
    if (!viewer3d || !analysisResults) return;

    const option = document.getElementById('viewer-highlight').value;
    const style = document.getElementById('viewer-style').value;
    const colorObj = getColorObject(document.getElementById('viewer-color').value);

    viewer3d.setStyle({}, { [style]: colorObj });

    let residues = [];
    let highlightColor = '#f87171';

    switch (option) {
        case 'flexible':
            residues = analysisResults.rmsf?.high_flexibility_residues || [];
            highlightColor = '#f87171';
            break;
        case 'hinge':
            residues = getInsightResidues('hinge_residue');
            highlightColor = '#fb923c';
            break;
        case 'stable_core':
            residues = getInsightResidues('stable_core');
            highlightColor = '#34d399';
            break;
        case 'stiffness':
            residues = getInsightResidues('local_stiffness_map');
            highlightColor = '#10b981';
            break;
        case 'hubs':
            residues = (analysisResults.allosteric?.hub_residues || []).map(h => h.resid);
            highlightColor = '#f87171';
            break;
        case 'gnn':
            residues = (analysisResults.gnn_results?.top_residues || []).map(r => r.resid);
            highlightColor = '#a78bfa';
            break;
        case 'mutation':
            residues = getInsightResidues('mutation_sensitivity');
            highlightColor = '#ef4444';
            break;
        case 'ptm':
            residues = getInsightResidues('ptm_site_prediction');
            highlightColor = '#fbbf24';
            break;
        case 'cryptic':
            residues = getInsightResidues('cryptic_binding_site');
            highlightColor = '#fbbf24';
            break;
        case 'ppi':
            residues = getInsightResidues('ppi_interface_hotspot');
            highlightColor = '#fca5a5';
            break;
        case 'druggability':
            residues = getInsightResidues('druggability_score');
            highlightColor = '#22d3ee';
            break;
        case 'aggregation':
            residues = getInsightResidues('aggregation_prone_region');
            highlightColor = '#fb923c';
            break;
        case 'electrostatic':
            residues = getInsightResidues('electrostatic_funnel');
            highlightColor = '#60a5fa';
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
        showToast('No analysis results available', 'warning');
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

    showToast(`Preparing ${type.toUpperCase()} download...`, 'info', 2000);
}

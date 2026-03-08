import requests, json, time

job_id = "95dd667d"
print(f"Monitoring job {job_id}...\n")

for i in range(120):
    r = requests.get(f"http://localhost:8000/api/results/{job_id}")
    data = r.json()
    status = data.get("status", "unknown")
    progress = data.get("progress", data.get("progress_percent", "?"))
    msg = data.get("message", "")

    if status == "running":
        print(f"  [{progress}%] {msg}")
        time.sleep(5)
    elif "job_id" in data and "rmsd" in data:
        print("=== ANALYSIS COMPLETED ===")
        print(f"Available result keys: {list(data.keys())}")
        ti = data.get("trajectory_info")
        if ti:
            print(f"\nTrajectory Info:")
            print(f"  Atoms: {ti.get('n_atoms')}")
            print(f"  Frames: {ti.get('n_frames')}")
            print(f"  Residues: {ti.get('n_residues')}")
        rmsd = data.get("rmsd")
        if rmsd:
            print(f"RMSD: mean={rmsd.get('mean_rmsd','?')} nm, max={rmsd.get('max_rmsd','?')} nm")
        rmsf = data.get("rmsf")
        if rmsf:
            print(f"RMSF: mean={rmsf.get('mean_rmsf','?')} nm")
        rg = data.get("rg")
        if rg:
            print(f"Rg: mean={rg.get('mean_rg','?')} nm")
        hb = data.get("hbonds")
        if hb:
            print(f"HBonds: mean={hb.get('mean_hbonds','?')}")
        sb = data.get("salt_bridges")
        if sb:
            print(f"Salt bridges: {sb.get('n_salt_bridges','?')} found")
        pca = data.get("pca")
        if pca:
            print(f"PCA: {pca.get('n_components','?')} components, variance={pca.get('explained_variance_ratio','?')}")
        bi = data.get("biological_insights")
        if bi and isinstance(bi, list):
            print(f"\nBiological Insights: {len(bi)} found")
            for insight in bi[:5]:
                if isinstance(insight, dict):
                    print(f"  - [{insight.get('type','')}] {insight.get('description','')[:100]}")
        plots = data.get("plots")
        if plots:
            print(f"\nPlots generated: {list(plots.keys())}")
        break
    elif status == "failed":
        print(f"FAILED: {msg}")
        print(json.dumps(data, indent=2)[:2000])
        break
    else:
        print(f"Status: {status} | progress: {progress} | {msg}")
        time.sleep(5)
else:
    print("Timed out waiting for results (10 min)")

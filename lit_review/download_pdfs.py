"""One-shot downloader for Phase-2 PDFs. Respects arxiv's 3s rate limit."""
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

# 17 papers from phase1_triage.md — 13 applicable + 2 methodology + 2 uncertain
TARGETS = [
    # APPLICABLE (13) + 2 flagged in main applicable list — see triage
    ("2410.09998v1", "SlimSeiz"),
    ("2407.19841v1", "RRAM"),
    ("2407.14876v1", "PreictalPeriodOptimization"),
    ("2402.09424v1", "SpikingConformer"),
    ("2306.08256v2", "DiffEEG"),
    ("2211.02679v1", "CNN_LSTM"),
    ("2209.11172v1", "TMC-ViT"),
    ("2206.09951v1", "MemristiveCNN"),
    ("2206.07518v1", "BSDCNN"),
    ("2108.07453v1", "EndToEndCNN_Xu"),
    ("2106.04510v1", "RandomForest_SPH5_SOP30"),
    ("2105.02823v1", "MultiScaleDilated3DCNN"),
    ("2012.00430v1", "DCGAN_CESP"),
    ("2012.00307v3", "EdgeDL_NeuralImplants"),
    ("2011.09581v1", "PatientIndependent_Dissanayake"),  # key cross-patient ref
    # METHODOLOGY (2)
    ("2302.10672v1", "Pale_MethodologicalChoices"),
    ("2306.12292v1", "Handa_DatasetsReport"),
    # UNCERTAIN (2)
    ("2403.03276v2", "ARNN"),
    ("2301.03465v2", "ShorterLatency"),
]

OUT_DIR = Path(__file__).parent / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch(paper_id: str, short: str):
    url = f"http://arxiv.org/pdf/{paper_id}"
    dst = OUT_DIR / f"{paper_id}__{short}.pdf"
    if dst.exists() and dst.stat().st_size > 10000:
        print(f"  SKIP {paper_id} ({dst.stat().st_size // 1024} KB already)")
        return True
    req = Request(url, headers={"User-Agent": "SzPredict-LitReview/0.1 (szpredict@hyperreal.com.au)"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        dst.write_bytes(data)
        print(f"  OK   {paper_id} -> {dst.name} ({len(data) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  FAIL {paper_id}: {e}")
        return False


def main():
    print(f"Downloading {len(TARGETS)} PDFs to {OUT_DIR}")
    ok = 0
    fail = 0
    for i, (pid, short) in enumerate(TARGETS):
        success = fetch(pid, short)
        if success:
            ok += 1
        else:
            fail += 1
        if i < len(TARGETS) - 1:
            time.sleep(3.2)  # arxiv rate-limit buffer
    total_mb = sum(p.stat().st_size for p in OUT_DIR.glob("*.pdf")) / (1024 * 1024)
    print(f"\nDone. ok={ok}, fail={fail}. Total on disk: {total_mb:.1f} MB in {OUT_DIR}")


if __name__ == "__main__":
    main()

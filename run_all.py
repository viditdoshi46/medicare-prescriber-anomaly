"""
End-to-end runner: build data -> clean -> features -> anomaly scoring.
Then launch the dashboard:  streamlit run app/streamlit_app.py

Usage:
    python run_all.py            # synthetic demo data (offline)
    python run_all.py --real     # download a real CMS sample first
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def run(mod: str):
    print(f"\n===== {mod} =====")
    subprocess.run([sys.executable, str(SRC / mod)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="download a real CMS Part D sample instead of synthetic")
    args = ap.parse_args()
    run("download_data.py" if args.real else "make_synthetic.py")
    run("clean.py")
    run("features.py")
    run("anomaly.py")
    print("\nDone. Launch the dashboard:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

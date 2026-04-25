#!/usr/bin/env bash
# SzPredict install script — handles Python dependencies + CHB-MIT dataset setup.
#
# Usage:
#   ./install.sh                  # Interactive: prompts for CHB-MIT handling
#   ./install.sh --skip-dataset   # Install Python deps only, skip CHB-MIT prompt
#
# Exit codes:
#   0 — success
#   1 — python / pip failure
#   2 — dataset validation failure
#   3 — user cancelled

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
NC=$'\033[0m'

echo "${BLUE}==============================================${NC}"
echo "${BLUE}  SzPredict — install${NC}"
echo "${BLUE}==============================================${NC}"
echo ""

# --- Python version check ---
PYTHON_CMD="${PYTHON:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "${RED}ERROR: python3 not found. Install Python 3.8+ first.${NC}"
    exit 1
fi

PY_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 8 ]]; }; then
    echo "${RED}ERROR: Python ${PY_VERSION} detected. SzPredict requires Python 3.8+.${NC}"
    exit 1
fi
echo "${GREEN}✓${NC} Python ${PY_VERSION} found"

# --- venv + pip install ---
# PEP 668 (Debian/Ubuntu/Pop!_OS modern Python) blocks system-wide pip,
# so we always install into a project-local venv at ./.venv
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo ""
    echo "Creating Python venv at $VENV_DIR ..."
    if ! "$PYTHON_CMD" -m venv "$VENV_DIR"; then
        echo "${RED}ERROR: venv creation failed. Install python3-venv (or python3-full on Debian).${NC}"
        echo "  sudo apt install python3-venv     # Debian/Ubuntu/Pop!_OS"
        exit 1
    fi
    echo "${GREEN}✓${NC} venv created"
fi

VENV_PY="$VENV_DIR/bin/python"

echo ""
echo "Installing Python dependencies into venv from requirements.txt ..."
if ! "$VENV_PY" -m pip install --upgrade pip >/dev/null; then
    echo "${YELLOW}Warning: failed to upgrade pip in venv (continuing).${NC}"
fi
if ! "$VENV_PY" -m pip install -r requirements.txt; then
    echo "${RED}ERROR: pip install failed. See output above.${NC}"
    exit 1
fi
echo "${GREEN}✓${NC} Python dependencies installed in $VENV_DIR"
echo ""
echo "${YELLOW}NOTE:${NC} Activate the venv before running scripts:"
echo "  source $VENV_DIR/bin/activate"
echo "  # ... or prefix commands with: $VENV_PY"

# --- Dataset handling ---
if [[ "${1:-}" == "--skip-dataset" ]]; then
    echo ""
    echo "${YELLOW}Skipping CHB-MIT setup (--skip-dataset flag)${NC}"
    echo "Use scripts/make_mock_labels.py to generate mock data for pipeline tests."
    echo ""
    echo "${GREEN}Install complete.${NC}"
    exit 0
fi

# --- Smoke test: mock pipeline end-to-end ---
# Confirms numpy, torch, metrics module, all 3 baselines, file I/O all work
# on this machine. Skip with --no-smoke-test.

SMOKE_TEST_FLAG=true
for arg in "$@"; do
    [[ "$arg" == "--no-smoke-test" ]] && SMOKE_TEST_FLAG=false
done

if $SMOKE_TEST_FLAG; then
    echo ""
    echo "${BLUE}==============================================${NC}"
    echo "${BLUE}  Smoke test — mock pipeline end-to-end${NC}"
    echo "${BLUE}==============================================${NC}"
    echo "Validates the install by running all 3 baselines on synthetic data."
    echo "Takes ~30-60 seconds. Skip with: ./install.sh --no-smoke-test"
    echo ""

    SMOKE_DIR="$SCRIPT_DIR/data/_smoke_test"
    rm -rf "$SMOKE_DIR"
    mkdir -p "$SMOKE_DIR" "$SCRIPT_DIR/results" "$SCRIPT_DIR/runs"

    SMOKE_FAIL=0

    echo "  [1/4] Generating mock data ..."
    if ! "$VENV_PY" scripts/make_mock_labels.py --out "$SMOKE_DIR" --n 2000 --events 5 --with-windows >/dev/null; then
        echo "${RED}    FAIL: mock data generation failed${NC}"
        SMOKE_FAIL=1
    else
        echo "${GREEN}    OK${NC}"
    fi

    if [[ $SMOKE_FAIL -eq 0 ]]; then
        echo "  [2/4] Running baseline_random ..."
        if ! "$VENV_PY" -m baselines.baseline_random --labels "$SMOKE_DIR/labels.npy" --event-ids "$SMOKE_DIR/event_ids.npy" --out "$SCRIPT_DIR/results/_smoke_random.json" >/dev/null; then
            echo "${RED}    FAIL: baseline_random failed${NC}"
            SMOKE_FAIL=1
        else
            echo "${GREEN}    OK${NC}"
        fi
    fi

    if [[ $SMOKE_FAIL -eq 0 ]]; then
        echo "  [3/4] Running baseline_majority ..."
        if ! "$VENV_PY" -m baselines.baseline_majority --labels "$SMOKE_DIR/labels.npy" --event-ids "$SMOKE_DIR/event_ids.npy" --out "$SCRIPT_DIR/results/_smoke_majority.json" >/dev/null; then
            echo "${RED}    FAIL: baseline_majority failed${NC}"
            SMOKE_FAIL=1
        else
            echo "${GREEN}    OK${NC}"
        fi
    fi

    if [[ $SMOKE_FAIL -eq 0 ]]; then
        echo "  [4/4] Running baseline_cnn (5 epochs, CPU OK) ..."
        if ! "$VENV_PY" -m baselines.baseline_cnn train \
            --train-x "$SMOKE_DIR/windows.npy" --train-y "$SMOKE_DIR/labels.npy" \
            --val-x "$SMOKE_DIR/windows.npy"   --val-y "$SMOKE_DIR/labels.npy" \
            --out "$SCRIPT_DIR/runs/_smoke_cnn" --epochs 5 --device cpu >/dev/null 2>&1; then
            echo "${RED}    FAIL: baseline_cnn train failed${NC}"
            SMOKE_FAIL=1
        else
            if ! "$VENV_PY" -m baselines.baseline_cnn eval \
                --ckpt "$SCRIPT_DIR/runs/_smoke_cnn/best.pt" \
                --test-x "$SMOKE_DIR/windows.npy" --test-y "$SMOKE_DIR/labels.npy" \
                --test-events "$SMOKE_DIR/event_ids.npy" \
                --out "$SCRIPT_DIR/results/_smoke_cnn.json" --device cpu >/dev/null 2>&1; then
                echo "${RED}    FAIL: baseline_cnn eval failed${NC}"
                SMOKE_FAIL=1
            else
                echo "${GREEN}    OK${NC}"
            fi
        fi
    fi

    echo ""
    if [[ $SMOKE_FAIL -eq 0 ]]; then
        echo "${GREEN}==============================================${NC}"
        echo "${GREEN}  Smoke test PASSED — pipeline is healthy.${NC}"
        echo "${GREEN}==============================================${NC}"
        echo "Output files in: results/_smoke_*.json (safe to delete)"
    else
        echo "${RED}==============================================${NC}"
        echo "${RED}  Smoke test FAILED — see errors above.${NC}"
        echo "${RED}==============================================${NC}"
        echo "Re-run with verbose output to debug:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  python -m baselines.baseline_random --labels $SMOKE_DIR/labels.npy --out /tmp/test.json"
    fi
fi


echo ""
echo "${BLUE}==============================================${NC}"
echo "${BLUE}  CHB-MIT Scalp EEG Database (~42 GB)${NC}"
echo "${BLUE}==============================================${NC}"
echo ""
echo "SzPredict's real evaluation runs against the CHB-MIT dataset from PhysioNet."
echo "How would you like to handle it?"
echo ""
echo "  ${GREEN}1${NC}) Download fresh from PhysioNet (~20–60 min depending on bandwidth)"
echo "  ${GREEN}2${NC}) Use existing local copy (we'll symlink and validate structure)"
echo "  ${GREEN}3${NC}) Skip (install code only; use mock data for pipeline testing)"
echo ""
read -r -p "Choice [1/2/3]: " DATASET_CHOICE

DATASET_DIR="$SCRIPT_DIR/data/chb-mit"

case "$DATASET_CHOICE" in
    1)
        echo ""
        echo "Downloading CHB-MIT from PhysioNet ..."
        if ! command -v wget >/dev/null 2>&1; then
            echo "${RED}ERROR: wget not found. Install wget or choose option 2 after manual download.${NC}"
            exit 2
        fi
        mkdir -p "$DATASET_DIR"
        cd "$DATASET_DIR"
        # PhysioNet wget-compatible mirror
        wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
        # Flatten: move contents from physionet.org/files/chbmit/1.0.0/ up
        if [[ -d "physionet.org/files/chbmit/1.0.0" ]]; then
            mv physionet.org/files/chbmit/1.0.0/* .
            rm -rf physionet.org
        fi
        cd "$SCRIPT_DIR"
        echo "${GREEN}✓${NC} CHB-MIT downloaded to $DATASET_DIR"
        ;;

    2)
        echo ""
        read -r -p "Path to existing CHB-MIT directory (should contain chb01/, chb02/, ...): " EXISTING_PATH
        EXISTING_PATH="${EXISTING_PATH/#\~/$HOME}"  # expand tilde
        if [[ ! -d "$EXISTING_PATH" ]]; then
            echo "${RED}ERROR: $EXISTING_PATH is not a directory.${NC}"
            exit 2
        fi
        # Validate structure: expect chb01 directory at minimum
        if [[ ! -d "$EXISTING_PATH/chb01" ]]; then
            echo "${RED}ERROR: $EXISTING_PATH does not contain a chb01/ subdirectory.${NC}"
            echo "        Expected layout: <path>/chb01/, <path>/chb02/, ..., <path>/chb24/"
            exit 2
        fi
        # Count subject directories
        N_SUBJECTS=$(find "$EXISTING_PATH" -maxdepth 1 -type d -name "chb*" | wc -l)
        if [[ "$N_SUBJECTS" -lt 20 ]]; then
            echo "${YELLOW}WARNING: Only $N_SUBJECTS chb* subdirectories found. Expected ~24.${NC}"
            read -r -p "Continue anyway? [y/N]: " CONTINUE
            if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
                echo "Cancelled."
                exit 3
            fi
        fi
        mkdir -p "$(dirname "$DATASET_DIR")"
        if [[ -e "$DATASET_DIR" ]]; then
            echo "${YELLOW}$DATASET_DIR already exists.${NC}"
            read -r -p "Remove and replace with symlink? [y/N]: " REPLACE
            if [[ "$REPLACE" == "y" || "$REPLACE" == "Y" ]]; then
                rm -rf "$DATASET_DIR"
            else
                echo "Cancelled."
                exit 3
            fi
        fi
        ln -s "$EXISTING_PATH" "$DATASET_DIR"
        echo "${GREEN}✓${NC} Symlinked $DATASET_DIR → $EXISTING_PATH ($N_SUBJECTS subjects)"
        ;;

    3)
        echo ""
        echo "${YELLOW}Skipping dataset setup.${NC}"
        echo "You can run baselines on mock data:"
        echo "    python scripts/make_mock_labels.py --out data/mock --n 10000 --events 15 --with-windows"
        ;;

    *)
        echo "${RED}Invalid choice. Cancelled.${NC}"
        exit 3
        ;;
esac


# --- Optional: end-to-end CHB-MIT benchmark run ---
# Only offer if the dataset symlink/download succeeded.
if [[ -d "$DATASET_DIR" ]] && [[ "${DATASET_CHOICE:-}" != "3" ]]; then
    echo ""
    echo "${BLUE}==============================================${NC}"
    echo "${BLUE}  Optional end-to-end CHB-MIT benchmark${NC}"
    echo "${BLUE}==============================================${NC}"
    echo "Runs benchmark_runner.py prepare on the test split, then"
    echo "trains baseline_cnn and scores it. ~30-90 min depending on hardware."
    echo "Validates the full pipeline (EDF parsing → labels → train → eval)."
    echo ""
    read -r -p "Run end-to-end benchmark now? [y/N]: " RUN_E2E

    if [[ "$RUN_E2E" == "y" || "$RUN_E2E" == "Y" ]]; then
        E2E_DIR="$SCRIPT_DIR/data/chbmit_p3/test"
        echo ""
        echo "  [1/3] Preparing CHB-MIT Protocol 3 test split (this loads EDF files) ..."
        if ! "$VENV_PY" scripts/benchmark_runner.py prepare \
            --chb-mit-dir "$DATASET_DIR" \
            --out "$E2E_DIR" \
            --protocol 3 --split test \
            --window-seconds 1 --include-windows; then
            echo "${RED}    FAIL: prepare step failed. Skipping rest of E2E.${NC}"
        else
            echo "${GREEN}    OK${NC}"
            echo "  [2/3] Training baseline_cnn on test split (using as both train+val for smoke purposes) ..."
            if ! "$VENV_PY" -m baselines.baseline_cnn train \
                --train-x "$E2E_DIR/windows.npy" --train-y "$E2E_DIR/labels.npy" \
                --val-x "$E2E_DIR/windows.npy" --val-y "$E2E_DIR/labels.npy" \
                --out "$SCRIPT_DIR/runs/cnn_e2e" --epochs 5; then
                echo "${RED}    FAIL: CNN training failed.${NC}"
            else
                echo "${GREEN}    OK${NC}"
                echo "  [3/3] Evaluating + scoring ..."
                if ! "$VENV_PY" -m baselines.baseline_cnn eval \
                    --ckpt "$SCRIPT_DIR/runs/cnn_e2e/best.pt" \
                    --test-x "$E2E_DIR/windows.npy" --test-y "$E2E_DIR/labels.npy" \
                    --test-events "$E2E_DIR/event_ids.npy" \
                    --out "$SCRIPT_DIR/results/baseline_cnn_chbmit_p3_test_e2e.json"; then
                    echo "${RED}    FAIL: eval failed.${NC}"
                else
                    echo "${GREEN}    OK${NC}"
                    echo ""
                    echo "${GREEN}End-to-end benchmark complete.${NC}"
                    echo "Result: $SCRIPT_DIR/results/baseline_cnn_chbmit_p3_test_e2e.json"
                fi
            fi
        fi
    else
        echo "Skipped. To run later:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  python scripts/benchmark_runner.py prepare --chb-mit-dir data/chb-mit \\"
        echo "    --out data/chbmit_p3/test --protocol 3 --split test --include-windows"
    fi
fi

echo ""
echo "${GREEN}==============================================${NC}"
echo "${GREEN}  Install complete.${NC}"
echo "${GREEN}==============================================${NC}"
echo ""
echo "Activate the venv before running scripts:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  • Full Protocol 3 prep: scripts/benchmark_runner.py prepare-all ..."
echo "  • Benchmark spec:       spec/BENCHMARK_SPEC.md"
echo "  • Submit results:       CONTRIBUTING.md"
echo ""

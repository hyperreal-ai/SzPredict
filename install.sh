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

# --- pip install ---
echo ""
echo "Installing Python dependencies from requirements.txt ..."
if ! "$PYTHON_CMD" -m pip install --upgrade -r requirements.txt; then
    echo "${RED}ERROR: pip install failed. See output above.${NC}"
    exit 1
fi
echo "${GREEN}✓${NC} Python dependencies installed"

# --- Dataset handling ---
if [[ "${1:-}" == "--skip-dataset" ]]; then
    echo ""
    echo "${YELLOW}Skipping CHB-MIT setup (--skip-dataset flag)${NC}"
    echo "Use scripts/make_mock_labels.py to generate mock data for pipeline tests."
    echo ""
    echo "${GREEN}Install complete.${NC}"
    exit 0
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

echo ""
echo "${GREEN}==============================================${NC}"
echo "${GREEN}  Install complete.${NC}"
echo "${GREEN}==============================================${NC}"
echo ""
echo "Next steps:"
echo "  • Mock pipeline test: see Quickstart in README.md"
echo "  • Benchmark spec:     spec/BENCHMARK_SPEC.md"
echo "  • Submit results:     docs/CONTRIBUTING.md"
echo ""

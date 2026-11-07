#!/usr/bin/env bash
# Build GKlib + METIS inside WSL (Ubuntu).
#
# Why WSL: the MSVC build of METIS 5.2.1 succeeds on this workstation but the
# resulting gpmetis.exe crashes with an access violation (0xC0000005) on every
# input, including METIS' own sample graphs, while graphchk.exe works -- i.e. a
# Win64/LLP64 portability problem in the library, not a build error.  Upstream
# tests on Linux, so we build there (see docs/decisions.md item D-7).
#
#   wsl -e bash /mnt/d/repos/phd_thesis/experiments/RGF/scripts/build_metis_wsl.sh
#
# Output: third_party/METIS/build-wsl/programs/gpmetis
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${REPO_ROOT}/third_party"
export PATH="${HOME}/.local/bin:${HOME}/opt/cmake/bin:${PATH}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "=== bootstrapping cmake ==="
  if ! python3 -m pip --version >/dev/null 2>&1; then
    curl -fsSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
      && python3 /tmp/get-pip.py --user --quiet || true
  fi
  if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --user --quiet cmake || true
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    mkdir -p "${HOME}/opt" && cd "${HOME}/opt"
    tag="$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/Kitware/CMake/releases/latest | sed -E 's#.*/tag/v##')"
    curl -fsSL -o cmake.tar.gz "https://github.com/Kitware/CMake/releases/download/v${tag}/cmake-${tag}-linux-x86_64.tar.gz"
    tar xzf cmake.tar.gz && mv "cmake-${tag}-linux-x86_64" cmake
    export PATH="${HOME}/opt/cmake/bin:${PATH}"
  fi
fi
cmake --version | head -1

[ -d "${TP}/GKlib" ] || git clone --depth 1 https://github.com/KarypisLab/GKlib.git "${TP}/GKlib"

echo "=== GKlib ==="
cmake -S "${TP}/GKlib" -B "${TP}/GKlib/build-wsl" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="${TP}/GKlib/install-wsl"
cmake --build "${TP}/GKlib/build-wsl" -j "$(nproc)" --target install

echo "=== prepare build/xinclude (widths are set by hand upstream) ==="
mkdir -p "${TP}/METIS/build/xinclude"
cp "${TP}/METIS/include/CMakeLists.txt" "${TP}/METIS/build/xinclude/CMakeLists.txt"
{
  printf '#define IDXTYPEWIDTH 32\n#define REALTYPEWIDTH 32\n'
  cat "${TP}/METIS/include/metis.h"
} > "${TP}/METIS/build/xinclude/metis.h"

echo "=== METIS ==="
# METIS_NO_IPO: the LTO link of the CLI programs fails here ("lto1: all warnings
# being treated as errors"); it is a build-system quirk, not an algorithm flag.
cmake -S "${TP}/METIS" -B "${TP}/METIS/build-wsl" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DMETIS_NO_IPO=ON -DGKLIB_PATH="${TP}/GKlib/install-wsl"
cmake --build "${TP}/METIS/build-wsl" -j "$(nproc)"

echo "=== gpmetis ==="
ls -l "${TP}/METIS/build-wsl/programs/gpmetis"
"${TP}/METIS/build-wsl/programs/gpmetis" "${TP}/METIS/graphs/4elt.graph" 4 | tail -5

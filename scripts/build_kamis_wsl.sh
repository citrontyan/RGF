#!/usr/bin/env bash
# Build KaMIS inside WSL (Ubuntu).  KaMIS is a Linux C++ project (CMake + KaHIP
# submodule) and its CMakeLists asks for CMake >= 3.10; the Ubuntu image that
# ships with this workstation has no cmake, so it is bootstrapped from pip into
# ~/.local when missing (no sudo needed).
#
#   wsl -e bash /mnt/d/repos/phd_thesis/experiments/RGF/scripts/build_kamis_wsl.sh
#
# Output: third_party/KaMIS/build-wsl/redumis (and the other KaMIS binaries).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAMIS="${REPO_ROOT}/third_party/KaMIS"

export PATH="${HOME}/.local/bin:${HOME}/opt/cmake/bin:${PATH}"
if ! command -v cmake >/dev/null 2>&1; then
  echo "=== bootstrapping cmake ==="
  # 1) pip --user (needs pip; bootstrap it when the image lacks python3-pip)
  if ! python3 -m pip --version >/dev/null 2>&1; then
    curl -fsSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
      && python3 /tmp/get-pip.py --user --quiet || true
  fi
  if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --user --quiet cmake && export PATH="${HOME}/.local/bin:${PATH}"
  fi
  # 2) fallback: prebuilt Linux tarball from the latest CMake release
  if ! command -v cmake >/dev/null 2>&1; then
    mkdir -p "${HOME}/opt" && cd "${HOME}/opt"
    tag="$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/Kitware/CMake/releases/latest | sed -E 's#.*/tag/v##')"
    echo "latest CMake release tag: ${tag}"
    curl -fsSL -o cmake.tar.gz "https://github.com/Kitware/CMake/releases/download/v${tag}/cmake-${tag}-linux-x86_64.tar.gz"
    tar xzf cmake.tar.gz && mv "cmake-${tag}-linux-x86_64" cmake
    export PATH="${HOME}/opt/cmake/bin:${PATH}"
  fi
fi
cmake --version | head -1

cd "${KAMIS}"
git submodule update --init --recursive || true

echo "=== configure ==="
# KaMIS ships sub-projects (wmis, mmwis) whose cmake_minimum_required is older
# than CMake 3.5, which modern CMake rejects unless the policy floor is lowered.
cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "=== build ==="
cmake --build build-wsl -j "$(nproc)"

echo "=== binaries ==="
find build-wsl -maxdepth 2 -type f -perm -u+x -name 'redumis*' -o -maxdepth 2 -type f -perm -u+x -name 'kamis*'

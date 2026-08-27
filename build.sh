#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
JOBS="$(nproc 2>/dev/null || echo 4)"

echo "==> Detecting package manager and checking dependencies..."

need_pkg() {
    # $1: command name (used to check if already installed)
    command -v "$1" >/dev/null 2>&1
}

install_arch() {
    echo "==> Detected Arch Linux (pacman)"
    local pkgs=()
    command -v cmake  >/dev/null 2>&1 || pkgs+=(cmake)
    command -v g++    >/dev/null 2>&1 || pkgs+=(gcc)
    command -v make   >/dev/null 2>&1 || pkgs+=(make)
    pacman -Qi opencv  >/dev/null 2>&1 || pkgs+=(opencv)
    pacman -Qi vtk     >/dev/null 2>&1 || pkgs+=(vtk)          # common OpenCV dependency
    pacman -Qi hdf5    >/dev/null 2>&1 || pkgs+=(hdf5)         # common OpenCV dependency
    pacman -Qi pkgconf >/dev/null 2>&1 || pkgs+=(pkgconf)

    if [ "${#pkgs[@]}" -gt 0 ]; then
        echo "==> Missing packages: ${pkgs[*]}"
        sudo pacman -Sy --needed --noconfirm "${pkgs[@]}"
    else
        echo "==> All dependencies already installed"
    fi
}

install_apt() {
    echo "==> Detected Debian/Ubuntu (apt)"
    local pkgs=()
    command -v cmake  >/dev/null 2>&1 || pkgs+=(cmake)
    command -v g++    >/dev/null 2>&1 || pkgs+=(build-essential)
    dpkg -s libopencv-dev >/dev/null 2>&1 || pkgs+=(libopencv-dev)
    dpkg -s pkg-config    >/dev/null 2>&1 || pkgs+=(pkg-config)

    if [ "${#pkgs[@]}" -gt 0 ]; then
        echo "==> Missing packages: ${pkgs[*]}"
        sudo apt-get update
        sudo apt-get install -y "${pkgs[@]}"
    else
        echo "==> All dependencies already installed"
    fi
}

if command -v pacman >/dev/null 2>&1; then
    install_arch
elif command -v apt-get >/dev/null 2>&1; then
    install_apt
else
    echo "!! Could not detect a supported package manager (not Arch/pacman or Debian/apt). Please install cmake / g++ / OpenCV manually."
    exit 1
fi

echo "==> Building project (${BUILD_DIR}/)"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"${JOBS}"

echo "==> Build complete! Binaries are located in ${BUILD_DIR}/"

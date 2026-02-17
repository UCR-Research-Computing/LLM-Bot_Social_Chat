#!/bin/bash

# Script to check if the local environment has all the necessary
# system-level dependencies to run the installer.

# --- Globals ---
OS_ID=""
PACKAGE_MANAGER=""
PYTHON_DEV_PKG=""
BUILD_ESSENTIALS_PKG=""
PYGAME_DEPS_PKG=""
MISSING_COUNT=0

# --- Helper Functions ---

show_help() {
    echo "Usage: ./check_environment.sh [OPTIONS]"
    echo ""
    echo "This script checks if the local environment has all the necessary"
    echo "system-level dependencies to run the installer."
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help    Show this help message and exit."
}

if [[ " $1 " == " -h " ]] || [[ " $1 " == " --help " ]]; then
    show_help
    exit 0
fi

print_header() {
    echo "================================================="
    echo " $1"
    echo "================================================="
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID=$ID
    else
        echo "Cannot determine OS. Exiting."
        exit 1
    fi

    case "$OS_ID" in
        ubuntu|debian|mint)
            PACKAGE_MANAGER="apt"
            PYTHON_DEV_PKG="python3-dev"
            BUILD_ESSENTIALS_PKG="build-essential"
            PYGAME_DEPS_PKG="libsdl2-dev"
            ;;
        fedora|centos|rhel)
            PACKAGE_MANAGER="dnf"
            PYTHON_DEV_PKG="python3-devel"
            BUILD_ESSENTIALS_PKG="Development Tools"
            PYGAME_DEPS_PKG="SDL2-devel"
            ;;
        arch)
            PACKAGE_MANAGER="pacman"
            PYTHON_DEV_PKG="python"
            BUILD_ESSENTIALS_PKG="base-devel"
            PYGAME_DEPS_PKG="sdl2"
            ;;
        *)
            echo "Unsupported Linux distribution: $OS_ID"
            exit 1
            ;;
    esac
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ '$1' is NOT installed."
        ((MISSING_COUNT++))
    else
        echo "✅ '$1' is installed."
    fi
}

check_package() {
    local pkg_name=$1
    local friendly_name=$2
    local is_installed=false

    case "$PACKAGE_MANAGER" in
        apt)
            if dpkg -l | grep -q "^ii  $pkg_name"; then is_installed=true; fi
            ;;
        dnf)
            if rpm -q "$pkg_name" &> /dev/null; then is_installed=true; fi
            ;;
        pacman)
            if pacman -Q "$pkg_name" &> /dev/null; then is_installed=true; fi
            ;;
    esac

    if $is_installed; then
        echo "✅ $friendly_name ($pkg_name) is installed."
    else
        echo "❌ $friendly_name ($pkg_name) is NOT installed."
        ((MISSING_COUNT++))
    fi
}

# --- Main Execution ---
print_header "Checking System Environment"
detect_os

echo ""
echo "--- Core Commands ---"
check_command "python3"
check_command "pip"
check_command "git"
check_command "uv"
check_command "sudo"

echo ""
echo "--- Build & Runtime Packages ---"
check_package "$PYTHON_DEV_PKG" "Python Development Headers"
check_package "$BUILD_ESSENTIALS_PKG" "Build Tools"
check_package "$PYGAME_DEPS_PKG" "Pygame Audio Dependencies"

echo ""
print_header "Result"
if [ $MISSING_COUNT -eq 0 ]; then
    echo "✅ Your system has all the required dependencies."
    exit 0
else
    echo "❌ Your system is missing $MISSING_COUNT required dependencies."
    echo "Please install them and run this check again."
    exit 1
fi

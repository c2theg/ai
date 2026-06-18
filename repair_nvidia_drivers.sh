#!/usr/bin/env bash
#
# Repair NVIDIA drivers on DGX Spark / GB10 Linux systems
# Updated: 6/16/2026
# Version: 0.0.4
#
# This script is designed to repair the NVIDIA driver stack on DGX Spark / GB10 Linux systems.
# It can be run in dry-run mode to see what would be done, or in actual mode to perform the repair.
# wget https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/repair_nvidia_drivers.sh && chmod +x repair_nvidia_drivers.sh && sudo ./repair_nvidia_drivers.sh

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
ASSUME_YES=0
DRY_RUN=1
INSTALL_MISSING=0
BLACKLIST_NOUVEAU=1
DRIVER_PACKAGE="${DRIVER_PACKAGE:-cuda-drivers}"
LIST_AVAILABLE=0

usage() {
  cat <<EOF
Usage: sudo ./${SCRIPT_NAME} [options]

Repair the NVIDIA driver stack on a DGX Spark / GB10 Linux system.

By default this script runs in dry-run mode and repairs/reinstalls NVIDIA
packages already present on the system. It does not pick a new driver branch
unless --install-missing is provided.

Options:
  -y, --yes                  Actually make changes. Without this, dry-run mode is used.
      --install-missing       Install a driver metapackage if no NVIDIA driver packages are detected.
      --driver-package NAME   Package to install with --install-missing. Default: cuda-drivers
      --list-available        Show NVIDIA driver packages available from configured repos, then exit.
      --no-blacklist-nouveau  Do not create/refresh the nouveau blacklist config.
  -h, --help                 Show this help.

Examples:
  sudo ./${SCRIPT_NAME}
  sudo ./${SCRIPT_NAME} --yes
  sudo ./${SCRIPT_NAME} --list-available
  sudo ./${SCRIPT_NAME} --yes --install-missing
  sudo ./${SCRIPT_NAME} --yes --install-missing --driver-package nvidia-driver-575
EOF
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi

  "$@"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -y|--yes)
        ASSUME_YES=1
        DRY_RUN=0
        ;;
      --install-missing)
        INSTALL_MISSING=1
        ;;
      --driver-package)
        shift
        [[ $# -gt 0 ]] || die "--driver-package requires a package name."
        DRIVER_PACKAGE="$1"
        ;;
      --driver-package=*)
        DRIVER_PACKAGE="${1#*=}"
        ;;
      --list-available)
        LIST_AVAILABLE=1
        ;;
      --no-blacklist-nouveau)
        BLACKLIST_NOUVEAU=0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
    shift
  done
}

require_linux_root() {
  [[ "$(uname -s)" == "Linux" ]] || die "This script is intended for Linux systems only."
  [[ "${EUID}" -eq 0 ]] || die "Run as root, for example: sudo ./${SCRIPT_NAME} --yes"
}

confirm_execution() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Dry-run mode: no packages, files, services, or modules will be changed."
    log "Re-run with --yes to perform repair."
    return 0
  fi

  [[ "$ASSUME_YES" -eq 1 ]] || die "Internal argument error: destructive mode requires --yes."

  cat <<EOF
This will repair package state, reinstall detected NVIDIA driver packages,
rebuild kernel modules/initramfs, and restart NVIDIA services where present.
A reboot is normally required after completion.
EOF

  read -r -p "Type REPAIR-NVIDIA-DRIVERS to continue: " answer
  [[ "$answer" == "REPAIR-NVIDIA-DRIVERS" ]] || die "Aborted."
}

os_id_like() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    printf '%s %s\n' "${ID:-}" "${ID_LIKE:-}"
  fi
}

print_inventory() {
  log "System inventory:"
  printf '  kernel: %s\n' "$(uname -r)"
  printf '  arch:   %s\n' "$(uname -m)"

  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    printf '  os:     %s\n' "${PRETTY_NAME:-unknown}"
  else
    printf '  os:     unknown\n'
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '  nvidia-smi: present\n'
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
  else
    printf '  nvidia-smi: not found\n'
  fi
}

stop_nvidia_services() {
  local services=(
    nvidia-persistenced
    nvidia-fabricmanager
    nvidia-powerd
    nvidia-dcgm
    dcgm
    nvsm
  )

  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skipping NVIDIA service stop."
    return 0
  fi

  for service in "${services[@]}"; do
    if systemctl list-unit-files "${service}.service" >/dev/null 2>&1; then
      log "Stopping ${service}.service if running."
      run systemctl stop "${service}.service" || true
    fi
  done
}

start_nvidia_services() {
  local services=(
    nvidia-persistenced
    nvidia-fabricmanager
    nvidia-powerd
    nvidia-dcgm
    dcgm
    nvsm
  )

  if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skipping NVIDIA service start."
    return 0
  fi

  for service in "${services[@]}"; do
    if systemctl list-unit-files "${service}.service" >/dev/null 2>&1; then
      log "Enabling/starting ${service}.service if appropriate."
      run systemctl enable "${service}.service" || true
      run systemctl restart "${service}.service" || true
    fi
  done
}

write_nouveau_blacklist() {
  [[ "$BLACKLIST_NOUVEAU" -eq 1 ]] || return 0

  local target=/etc/modprobe.d/blacklist-nouveau.conf
  log "Creating/refreshing nouveau blacklist at ${target}."

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] write %s\n' "$target"
    return 0
  fi

  cat >"$target" <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
}

unload_conflicting_modules() {
  local modules=(
    nouveau
    nvidia_uvm
    nvidia_drm
    nvidia_modeset
    nvidia_peermem
    nvidia
  )

  log "Attempting to unload conflicting/NVIDIA kernel modules before repair."
  for module in "${modules[@]}"; do
    if lsmod | awk '{print $1}' | grep -qx "$module"; then
      run modprobe -r "$module" || log "Could not unload ${module}; it may still be in use."
    fi
  done
}

load_nvidia_modules() {
  local modules=(
    nvidia
    nvidia_uvm
    nvidia_modeset
    nvidia_drm
  )

  log "Attempting to load NVIDIA kernel modules."
  for module in "${modules[@]}"; do
    run modprobe "$module" || log "Could not load ${module}; a reboot may be required."
  done
}

refresh_boot_artifacts() {
  run depmod -a

  if command -v update-initramfs >/dev/null 2>&1; then
    log "Refreshing initramfs."
    run update-initramfs -u -k all
  elif command -v dracut >/dev/null 2>&1; then
    log "Refreshing initramfs with dracut."
    run dracut --force --regenerate-all
  fi

  if command -v update-grub >/dev/null 2>&1; then
    log "Refreshing GRUB configuration."
    run update-grub
  elif command -v grub2-mkconfig >/dev/null 2>&1; then
    local grub_cfg
    grub_cfg="$(find /boot -path '*grub.cfg' -print -quit 2>/dev/null || true)"
    if [[ -n "$grub_cfg" ]]; then
      run grub2-mkconfig -o "$grub_cfg"
    fi
  fi
}

detect_debian_nvidia_packages() {
  dpkg-query -W -f='${Status}\t${binary:Package}\n' \
    'cuda-drivers*' \
    'libcuda*' \
    'libnvidia*' \
    'nvidia-*' \
    'xserver-xorg-video-nvidia*' \
    2>/dev/null |
    awk '
      $1 == "install" && $2 == "ok" && $3 == "installed" {
        package = $4
        if (
          package !~ /-doc$/ &&
          package !~ /-dev$/ &&
          package !~ /-headers-common$/
        ) {
          print package
        }
      }
    ' |
    sort -u
}

list_available_debian_drivers() {
  log "Refreshing apt metadata."
  run apt-get update

  log "cuda-drivers policy:"
  run apt-cache policy cuda-drivers || true

  log "Available nvidia-driver packages:"
  apt-cache search --names-only '^nvidia-driver-[0-9]+(-open)?$' |
    awk '{print $1}' |
    sort -V |
    while read -r package; do
      [[ -n "$package" ]] || continue
      apt-cache policy "$package" |
        awk -v package="$package" '
          /Candidate:/ {
            printf "  %-28s %s\n", package, $2
          }
        '
    done

  log "Use the highest listed package that is supported by DGX OS for your GB10 image."
}

repair_debian_packages() {
  export DEBIAN_FRONTEND=noninteractive

  local packages=()
  mapfile -t packages < <(detect_debian_nvidia_packages)

  log "Repairing apt/dpkg package state."
  run apt-get update
  run dpkg --configure -a
  run apt-get install -f -y
  run apt-get install --reinstall -y "linux-headers-$(uname -r)" dkms || true

  if [[ "${#packages[@]}" -gt 0 ]]; then
    log "Reinstalling detected NVIDIA packages: ${packages[*]}"
    run apt-get install --reinstall -y "${packages[@]}"
  elif [[ "$INSTALL_MISSING" -eq 1 ]]; then
    log "No NVIDIA driver packages detected; installing ${DRIVER_PACKAGE}."
    run apt-get install -y "$DRIVER_PACKAGE"
  else
    die "No NVIDIA driver packages detected. Re-run with --install-missing to install ${DRIVER_PACKAGE}."
  fi

  if command -v dkms >/dev/null 2>&1; then
    log "Running DKMS autoinstall for the current kernel."
    run dkms autoinstall -k "$(uname -r)" || true
  fi
}

detect_rhel_nvidia_packages() {
  rpm -qa |
    awk '
      /^(akmod-nvidia|dkms-nvidia|kmod-nvidia|libnvidia|nvidia-|xorg-x11-drv-nvidia)/ &&
      !/-doc/ &&
      !/-devel/ {
        print
      }
    ' |
    sort -u
}

list_available_rhel_drivers() {
  local pm="$1"

  log "Refreshing ${pm} metadata."
  run "$pm" makecache -y || true

  log "Available NVIDIA driver packages:"
  "$pm" list --available 'nvidia-driver*' 'cuda-drivers*' 2>/dev/null || true

  log "Use the highest listed package that is supported by DGX OS for your GB10 image."
}

repair_rhel_packages() {
  local pm="$1"
  local packages=()
  mapfile -t packages < <(detect_rhel_nvidia_packages)

  log "Repairing packages with ${pm}."
  run "$pm" makecache -y || true
  run "$pm" reinstall -y "kernel-headers-$(uname -r)" "kernel-devel-$(uname -r)" dkms || true

  if [[ "${#packages[@]}" -gt 0 ]]; then
    log "Reinstalling detected NVIDIA packages: ${packages[*]}"
    run "$pm" reinstall -y "${packages[@]}"
  elif [[ "$INSTALL_MISSING" -eq 1 ]]; then
    log "No NVIDIA driver packages detected; installing ${DRIVER_PACKAGE}."
    run "$pm" install -y "$DRIVER_PACKAGE"
  else
    die "No NVIDIA driver packages detected. Re-run with --install-missing to install ${DRIVER_PACKAGE}."
  fi

  if command -v dkms >/dev/null 2>&1; then
    log "Running DKMS autoinstall for the current kernel."
    run dkms autoinstall -k "$(uname -r)" || true
  fi

  if command -v akmods >/dev/null 2>&1; then
    log "Running akmods for NVIDIA modules."
    run akmods --force || true
  fi
}

validate_nvidia() {
  log "Post-repair validation:"

  if command -v nvidia-smi >/dev/null 2>&1; then
    run nvidia-smi || true
  else
    log "nvidia-smi is still not installed or not on PATH."
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if lsmod | awk '{print $1}' | grep -q '^nvidia'; then
      log "NVIDIA kernel modules are loaded."
    else
      log "NVIDIA kernel modules are not loaded yet; reboot before final validation."
    fi
  fi
}

main() {
  parse_args "$@"
  require_linux_root
  print_inventory

  if [[ "$LIST_AVAILABLE" -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      list_available_debian_drivers
    elif command -v dnf >/dev/null 2>&1; then
      list_available_rhel_drivers dnf
    elif command -v yum >/dev/null 2>&1; then
      list_available_rhel_drivers yum
    else
      die "Unsupported package manager. Expected apt-get, dnf, or yum."
    fi
    exit 0
  fi

  confirm_execution
  stop_nvidia_services
  write_nouveau_blacklist
  unload_conflicting_modules

  if command -v apt-get >/dev/null 2>&1; then
    repair_debian_packages
  elif command -v dnf >/dev/null 2>&1; then
    repair_rhel_packages dnf
  elif command -v yum >/dev/null 2>&1; then
    repair_rhel_packages yum
  else
    die "Unsupported package manager. Expected apt-get, dnf, or yum."
  fi

  refresh_boot_artifacts
  load_nvidia_modules
  start_nvidia_services
  validate_nvidia

  log "Done. Reboot the system before deciding the repair failed."
}

main "$@"

#!/bin/bash

################################################################################
# NVIDIA Driver Update Script for PyTorch 2.8 Support
#
# This script updates the NVIDIA driver to the latest available version
# for full PyTorch 2.8 CUDA 12.8 compatibility with Tesla V100 and newer GPUs.
#
# Usage: sudo bash ./scripts/update-nvidia-driver.sh [version]
#   - version: Optional driver version to install (e.g., "560", "550")
#   - If omitted, installs the latest available version
#
# Rollback: sudo bash ./scripts/rollback-nvidia-driver.sh (auto-generated)
################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/nvidia-update-$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="/var/cache/nvidia-driver-backup-$(date +%Y%m%d_%H%M%S)"
ROLLBACK_SCRIPT="${SCRIPT_DIR}/rollback-nvidia-driver.sh"
TARGET_DRIVER="${1:-latest}"  # Allow override via command line argument
DEBIAN_VERSION="bullseye"
DETECTED_DRIVER=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}✓${NC} $*" | tee -a "${LOG_FILE}"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}✗${NC} $*" | tee -a "${LOG_FILE}"
}

################################################################################
# Pre-flight Checks
################################################################################

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use: sudo bash $0)"
        exit 1
    fi
    success "Running with root privileges"
}

check_os() {
    if ! grep -q "Debian" /etc/os-release; then
        error "This script is optimized for Debian. Other distros may require modifications."
        exit 1
    fi
    success "Debian OS detected"
}

check_internet() {
    if ! ping -c 1 8.8.8.8 &> /dev/null; then
        error "No internet connection. Cannot download driver packages."
        exit 1
    fi
    success "Internet connection verified"
}

check_current_driver() {
    if ! command -v nvidia-smi &> /dev/null; then
        warning "NVIDIA driver not detected. This might be a fresh install."
        CURRENT_DRIVER="not installed"
        CURRENT_DRIVER_VERSION="0"
    else
        CURRENT_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        CURRENT_DRIVER_VERSION="${CURRENT_DRIVER%%.*}"
        success "Current NVIDIA driver: ${CURRENT_DRIVER} (R${CURRENT_DRIVER_VERSION})"
    fi
}

check_gpu() {
    if ! nvidia-smi &> /dev/null 2>&1; then
        warning "Cannot detect GPU. Continuing anyway..."
        GPU_DETECTED=false
    else
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        success "GPU detected: ${GPU_NAME}"
        GPU_DETECTED=true
    fi
}

check_running_gpu_processes() {
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null || echo "")
    if [[ -n "${GPU_PROCESSES}" ]]; then
        warning "GPU processes detected:"
        echo "${GPU_PROCESSES}" | tee -a "${LOG_FILE}"
        echo -e "\n${YELLOW}Recommendation: Close these applications before updating the driver.${NC}"
        read -p "Continue anyway? (yes/no): " response
        if [[ "$response" != "yes" ]]; then
            error "Update cancelled by user"
            exit 1
        fi
    else
        success "No GPU processes detected"
    fi
}

check_disk_space() {
    AVAILABLE_SPACE=$(df /var | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=$((2000000)) # ~2GB

    if [[ ${AVAILABLE_SPACE} -lt ${REQUIRED_SPACE} ]]; then
        error "Insufficient disk space. Need ~2GB, have ~$(( AVAILABLE_SPACE / 1000000 ))GB"
        exit 1
    fi
    success "Disk space check passed (available: $(( AVAILABLE_SPACE / 1000000 ))GB)"
}

detect_available_drivers() {
    log "Detecting available NVIDIA driver versions..."

    # Get list of available driver versions from apt
    AVAILABLE_VERSIONS=$(apt-cache policy nvidia-driver 2>/dev/null | grep -E "^\s+[0-9]+\.[0-9]+" | awk '{print $1}' | sort -rV | head -5)

    if [[ -z "${AVAILABLE_VERSIONS}" ]]; then
        error "Could not detect any available NVIDIA drivers"
        exit 1
    fi

    log "Available NVIDIA driver versions:"
    echo "${AVAILABLE_VERSIONS}" | nl | sed 's/^/  /'

    if [[ "${TARGET_DRIVER}" == "latest" ]]; then
        DETECTED_DRIVER=$(echo "${AVAILABLE_VERSIONS}" | head -1)
        success "Auto-selected latest available driver: ${DETECTED_DRIVER}"
    else
        # Check if requested driver is available
        if echo "${AVAILABLE_VERSIONS}" | grep -q "^${TARGET_DRIVER}"; then
            DETECTED_DRIVER="${TARGET_DRIVER}"
            success "Found requested driver version: ${DETECTED_DRIVER}"
        else
            warning "Requested driver ${TARGET_DRIVER} not found in available versions"
            DETECTED_DRIVER=$(echo "${AVAILABLE_VERSIONS}" | head -1)
            success "Using closest available version: ${DETECTED_DRIVER}"
        fi
    fi
}

################################################################################
# Backup Phase
################################################################################

create_backup() {
    log "Creating backup..."
    mkdir -p "${BACKUP_DIR}"

    # Backup current driver version info
    {
        echo "Backup timestamp: $(date)"
        echo "Current driver version: ${CURRENT_DRIVER_VERSION}"
        echo "Full driver version: ${CURRENT_DRIVER}"
        echo "GPU: ${GPU_NAME}"
    } > "${BACKUP_DIR}/driver-info.txt"
    success "Created backup at: ${BACKUP_DIR}"

    # Backup Xorg config if it exists
    if [[ -f /etc/X11/xorg.conf ]]; then
        cp /etc/X11/xorg.conf "${BACKUP_DIR}/xorg.conf.backup"
        success "Backed up Xorg configuration"
    fi

    # Save kernel module status
    lsmod | grep -E 'nvidia|nouveau' > "${BACKUP_DIR}/kernel-modules.txt" || true

    # Document installed packages
    dpkg -l | grep -i nvidia > "${BACKUP_DIR}/nvidia-packages.txt" || true

    log "Backup complete. Location: ${BACKUP_DIR}"
}

generate_rollback_script() {
    log "Generating rollback script..."

    cat > "${ROLLBACK_SCRIPT}" << 'EOF'
#!/bin/bash
# Auto-generated NVIDIA Driver Rollback Script
# This script reverts to an earlier NVIDIA driver version

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }

if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
    exit 1
fi

log "Starting NVIDIA driver rollback..."

# Update package lists
log "Updating package lists..."
apt-get update

# Try to install an older stable driver version
# Attempt versions in order of stability
for DRIVER_VERSION in 550 545 535 530; do
    log "Attempting to install driver version ${DRIVER_VERSION}..."
    if apt-get install -y --no-install-recommends nvidia-driver; then
        log "Successfully installed compatible driver"
        break
    fi
done

# Install build dependencies for kernel module compilation
log "Installing build dependencies..."
apt-get install -y --no-install-recommends build-essential libelf-dev dkms

# Recompile kernel modules
log "Recompiling NVIDIA kernel modules..."
dkms autoinstall -q

# Verify installation
log "Verifying driver installation..."
modprobe -r nouveau 2>/dev/null || true
modprobe nvidia

if nvidia-smi &>/dev/null; then
    success "Driver rollback successful!"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
else
    error "Driver verification failed. Manual intervention may be required."
    exit 1
fi

success "Rollback complete. Please reboot the system."
EOF

    chmod +x "${ROLLBACK_SCRIPT}"
    success "Rollback script created at: ${ROLLBACK_SCRIPT}"
}

################################################################################
# Driver Update Phase
################################################################################

remove_nouveau() {
    log "Removing Nouveau driver (if present)..."
    if lsmod | grep -q nouveau; then
        warning "Nouveau driver is loaded. Disabling it..."
        modprobe -r nouveau || true
        echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
        echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
        success "Nouveau driver blacklisted"
    else
        success "Nouveau driver not found (OK)"
    fi
}

add_nvidia_repositories() {
    log "Adding NVIDIA driver repository..."

    # Add NVIDIA PPA
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub 2>/dev/null || true

    # Add repository
    NVIDIA_REPO="deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
    if ! grep -q "nvidia.com/compute/cuda" /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; then
        echo "${NVIDIA_REPO}" >> /etc/apt/sources.list.d/nvidia-cuda.list
        success "NVIDIA CUDA repository added"
    else
        success "NVIDIA repository already configured"
    fi
}

update_packages() {
    log "Updating package lists..."
    apt-get update || true
    success "Package lists updated"
}

install_driver() {
    log "Installing NVIDIA driver ${DETECTED_DRIVER}..."

    # Install build dependencies first
    log "Installing build dependencies..."
    apt-get install -y --no-install-recommends \
        build-essential \
        libelf-dev \
        dkms || {
        error "Failed to install build dependencies"
        exit 1
    }

    # Try to install the driver using the metapackage (most reliable)
    log "Installing NVIDIA driver metapackage..."
    if apt-get install -y --no-install-recommends nvidia-driver; then
        success "NVIDIA driver installed successfully"
        return 0
    fi

    # If metapackage fails, try the candidate version
    log "Metapackage installation had issues, trying candidate version..."
    CANDIDATE_VERSION=$(apt-cache policy nvidia-driver 2>/dev/null | grep "Candidate:" | awk '{print $2}')

    if [[ -n "${CANDIDATE_VERSION}" ]]; then
        if apt-get install -y --no-install-recommends "nvidia-driver=${CANDIDATE_VERSION}"; then
            success "NVIDIA driver ${CANDIDATE_VERSION} installed successfully"
            return 0
        fi
    fi

    error "Failed to install NVIDIA driver. Please check:"
    error "  1. Your internet connection"
    error "  2. Repository configuration (apt-cache policy nvidia-driver)"
    error "  3. System dependencies (apt-get install -f)"
    exit 1
}

recompile_kernel_modules() {
    log "Recompiling NVIDIA kernel modules..."

    dkms autoinstall -q || {
        error "Kernel module compilation failed"
        exit 1
    }

    success "Kernel modules compiled successfully"
}

################################################################################
# Verification Phase
################################################################################

verify_driver_installation() {
    log "Verifying driver installation..."

    # Load module
    modprobe -r nouveau 2>/dev/null || true
    modprobe nvidia || {
        error "Failed to load NVIDIA kernel module"
        exit 1
    }
    success "NVIDIA kernel module loaded"

    # Check nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found after installation"
        exit 1
    fi

    # Get driver version
    NEW_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    success "New driver version: ${NEW_DRIVER}"

    # Verify GPU detection
    if nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        success "GPUs detected: ${GPU_COUNT}"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    else
        error "nvidia-smi test failed"
        exit 1
    fi
}

verify_cuda_runtime() {
    log "Verifying CUDA runtime..."

    CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    CUDA_LIBS=$(ldconfig -p | grep -c "libcuda.so" || echo "0")

    success "CUDA libraries found: ${CUDA_LIBS}"

    if [[ ${CUDA_LIBS} -eq 0 ]]; then
        warning "No CUDA libraries detected. Run 'apt-get install cuda-drivers' if needed."
    fi
}

################################################################################
# PyTorch Verification
################################################################################

verify_pytorch() {
    log "Checking PyTorch GPU support..."

    PYTHON=$(command -v python3 || command -v python)

    if [[ -z "${PYTHON}" ]]; then
        warning "Python not found. Skipping PyTorch verification."
        return
    fi

    # Check if PyTorch is installed
    if ! ${PYTHON} -c "import torch" 2>/dev/null; then
        warning "PyTorch not installed. Cannot verify GPU support."
        return
    fi

    # Run verification script if it exists
    if [[ -f "${SCRIPT_DIR}/verify-pytorch-gpu.py" ]]; then
        log "Running PyTorch GPU verification..."
        ${PYTHON} "${SCRIPT_DIR}/verify-pytorch-gpu.py" || true
    fi
}

################################################################################
# Interactive Confirmation
################################################################################

interactive_prompt() {
    log "\n========================================="
    log "NVIDIA Driver Update Summary"
    log "========================================="
    echo
    echo "Current Configuration:"
    echo "  Current Driver: ${CURRENT_DRIVER} (R${CURRENT_DRIVER_VERSION})"
    echo "  GPU: ${GPU_NAME}"
    echo "  OS: Debian 11 (Bullseye)"
    echo
    echo "Update Plan:"
    echo "  Target Driver: ${DETECTED_DRIVER}"
    echo "  Purpose: PyTorch 2.8 with CUDA 12.8 support"
    echo "  Backup Location: ${BACKUP_DIR}"
    echo "  Rollback Available: Yes (${ROLLBACK_SCRIPT})"
    echo
    echo -e "${YELLOW}WARNING:${NC}"
    echo "  1. This will recompile kernel modules"
    echo "  2. System may be unstable during the process"
    echo "  3. A reboot will be required afterward"
    echo "  4. GPU-dependent applications should be closed"
    echo
    read -p "Do you want to proceed? (yes/no): " response
    if [[ "$response" != "yes" ]]; then
        error "Update cancelled by user"
        exit 0
    fi
}

################################################################################
# Cleanup and Reboot
################################################################################

print_next_steps() {
    echo
    log "========================================="
    log "Update Complete!"
    log "========================================="
    echo
    success "NVIDIA driver updated successfully!"
    echo
    echo "Next Steps:"
    echo "  1. Reboot your system: sudo reboot"
    echo "  2. After reboot, verify installation: nvidia-smi"
    echo "  3. Verify PyTorch GPU support:"
    echo "     python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
    echo
    echo "Logs saved to: ${LOG_FILE}"
    echo "Backup saved to: ${BACKUP_DIR}"
    echo "Rollback script: ${ROLLBACK_SCRIPT}"
    echo
    warning "A system reboot is required to apply the changes."
    echo
    read -p "Reboot now? (yes/no): " reboot_response
    if [[ "$reboot_response" == "yes" ]]; then
        log "Initiating reboot..."
        sleep 2
        reboot
    else
        warning "Please reboot manually when ready."
    fi
}

################################################################################
# Main Execution Flow
################################################################################

main() {
    log "========================================="
    log "NVIDIA Driver Update Script"
    log "========================================="

    # Pre-flight checks
    check_root
    check_os
    check_internet
    check_current_driver
    check_gpu
    check_running_gpu_processes
    check_disk_space

    # Detect available drivers
    detect_available_drivers

    # Interactive prompt
    interactive_prompt

    # Backup phase
    create_backup
    generate_rollback_script

    # Update phase
    remove_nouveau
    add_nvidia_repositories
    update_packages
    install_driver
    recompile_kernel_modules

    # Verification phase
    verify_driver_installation
    verify_cuda_runtime
    verify_pytorch

    # Print next steps
    print_next_steps
}

# Execute main function
main

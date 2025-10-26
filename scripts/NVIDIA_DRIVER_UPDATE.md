# NVIDIA Driver Update Scripts

This directory contains scripts for updating NVIDIA drivers and verifying PyTorch GPU support.

## Overview

These scripts are designed to update your NVIDIA driver from R550 (CUDA 12.4) to R570+ (CUDA 12.8) to ensure full PyTorch 2.8 compatibility with optimal performance on Tesla V100 and newer GPUs.

## Scripts

### 1. `update-nvidia-driver.sh` - Main Update Script

**Purpose**: Updates NVIDIA driver to R570+ series for CUDA 12.8 support.

**Usage**:
```bash
sudo bash ./scripts/update-nvidia-driver.sh
```

**What it does**:
1. **Pre-flight Checks**:
   - Verifies root privileges
   - Checks OS compatibility (Debian/Ubuntu)
   - Validates internet connectivity
   - Detects current driver version
   - Checks for available GPUs
   - Warns about running GPU processes
   - Validates disk space (requires ~2GB)

2. **Backup Phase**:
   - Creates full backup of current driver configuration
   - Stores in `/var/cache/nvidia-driver-backup-*/`
   - Auto-generates rollback script for recovery

3. **Update Phase**:
   - Blacklists Nouveau driver (if present)
   - Adds NVIDIA CUDA repository
   - Installs NVIDIA driver R570 + utilities + DKMS
   - Recompiles kernel modules

4. **Verification Phase**:
   - Verifies driver installation
   - Checks CUDA runtime availability
   - Tests PyTorch GPU support (if available)

5. **Post-Update**:
   - Displays summary with next steps
   - Prompts for system reboot
   - Saves full logs to `/var/log/nvidia-update-*.log`

**Features**:
- ✅ Full logging to `/var/log/nvidia-update-*.log`
- ✅ Automatic backup creation
- ✅ Auto-generated rollback script
- ✅ Interactive confirmation prompts
- ✅ Comprehensive pre-flight checks
- ✅ Color-coded output for easy reading

### 2. `verify-pytorch-gpu.py` - GPU Verification Script

**Purpose**: Validates that PyTorch can successfully use the GPU.

**Usage**:
```bash
python scripts/verify-pytorch-gpu.py
```

**What it checks**:
- PyTorch installation and version
- CUDA availability
- GPU detection and properties
- CUDA version compatibility
- cuDNN version
- Basic CUDA tensor operations
- PyTorch GPU training readiness

**Output**:
```
==================================================
Verification Summary
==================================================
✓ PASS PyTorch Installation
✓ PASS PyTorch Version
✓ PASS CUDA Availability
✓ PASS GPU Detection
✓ PASS GPU Properties
✓ PASS CUDA Version
✓ PASS cuDNN Version
✓ PASS CUDA Operations
==================================================
✓ All checks passed! GPU is ready for PyTorch training.
```

## Workflow

### Before Updating

1. **Verify current status**:
   ```bash
   nvidia-smi
   python scripts/verify-pytorch-gpu.py
   ```

2. **Close GPU-dependent applications**:
   - Close any training scripts
   - Close any GPU compute jobs
   - The script will warn you and let you proceed anyway if needed

### Updating

1. **Run the update script**:
   ```bash
   sudo bash scripts/update-nvidia-driver.sh
   ```

2. **Follow prompts**:
   - Review the summary
   - Confirm you want to proceed
   - Let the script complete (5-10 minutes typically)
   - Optionally reboot when prompted

3. **After reboot** (if you chose to reboot):
   ```bash
   nvidia-smi
   python scripts/verify-pytorch-gpu.py
   ```

### If Something Goes Wrong

If the update fails or causes issues, use the generated rollback script:

```bash
sudo bash scripts/rollback-nvidia-driver.sh
```

This will restore the previous driver version.

## System Requirements

- **OS**: Debian/Ubuntu based systems
- **Disk Space**: ~2GB available in `/var`
- **Internet**: Required for downloading driver packages
- **Root Access**: `sudo` privileges required
- **GPU**: Any NVIDIA GPU supported by R570+ driver series
- **Python**: Python 3.7+ with PyTorch installed (for verification)

## Expected Outcomes

After successful update:
- NVIDIA driver version: R570 or newer
- CUDA version: 12.8
- Full PyTorch 2.8 compatibility
- Optimal performance for training on modern GPUs

## Troubleshooting

### Script fails to run
```bash
# Make sure script is executable
chmod +x scripts/update-nvidia-driver.sh
```

### Root privileges error
```bash
# Run with sudo
sudo bash scripts/update-nvidia-driver.sh
```

### NVIDIA repository download fails
- Check internet connectivity: `ping 8.8.8.8`
- Try manually: `apt-get update`

### Kernel module compilation fails
- Check disk space: `df -h /var`
- Check if another kernel module build is running: `ps aux | grep dkms`

### After update, CUDA not available
1. Run verification script: `python scripts/verify-pytorch-gpu.py`
2. Check nvidia-smi: `nvidia-smi`
3. Try rollback if needed: `sudo bash scripts/rollback-nvidia-driver.sh`

## Log Files

All operations are logged for debugging:

- **Update Log**: `/var/log/nvidia-update-YYYYMMDD_HHMMSS.log`
- **Backup Info**: `/var/cache/nvidia-driver-backup-YYYYMMDD_HHMMSS/driver-info.txt`
- **Rollback Script**: `scripts/rollback-nvidia-driver.sh` (auto-generated)

## Additional Resources

- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/driverDetails.aspx/123456)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch GPU Support](https://pytorch.org/docs/stable/notes/cuda.html)
- [DKMS Documentation](https://linux.die.net/man/8/dkms)

## Safety Notes

⚠️ **Important**:
1. This script modifies system-critical driver files
2. A failed update could cause GPU to become unavailable
3. Always ensure you have a rollback path
4. Test on non-critical systems first if possible
5. Keep the generated rollback script safe

## Version Compatibility

- **PyTorch 2.8+**: Requires CUDA 12.4 minimum (12.8 recommended)
- **NVIDIA R570+**: Provides CUDA 12.8 support
- **Tesla V100**: Compute Capability 7.0, fully supported

---

**Last Updated**: 2025-10-26
**Tested On**: Debian Bullseye, Tesla V100, PyTorch 2.8+

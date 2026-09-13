# Setup script for OmniPlate-RU environment on Host PC (DESKTOP-BAFS38R)
# Run this directly in PowerShell on the Host PC:
# cd D:\AIProjects\VolgaIT
# powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

Write-Host "=== Setting up OmniPlate-RU Python Environment ===" -ForegroundColor Cyan

# 1. Locate Python 3.10
$pythonPaths = @(
    "C:\Users\Vamsi\AppData\Local\Programs\Python\Python310\python.exe",
    "python"
)

$targetPython = $null
foreach ($p in $pythonPaths) {
    if (Test-Path $p) {
        $targetPython = $p
        break
    }
}

if (-not $targetPython) {
    $targetPython = "python"
}

Write-Host "[1/5] Using base Python: $targetPython" -ForegroundColor Green
& $targetPython --version

# 2. Create virtual environment if not exists
if (-not (Test-Path ".venv")) {
    Write-Host "[2/5] Creating virtual environment (.venv)..." -ForegroundColor Green
    & $targetPython -m venv .venv
} else {
    Write-Host "[2/5] Virtual environment (.venv) already exists." -ForegroundColor Yellow
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# 3. Upgrade pip & build tools
Write-Host "[3/5] Upgrading pip and tooling..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA 12.4
Write-Host "[4/5] Installing PyTorch with CUDA support..." -ForegroundColor Green
& $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install CV and ML dependencies
Write-Host "[5/5] Installing project dependencies..." -ForegroundColor Green
& $venvPip install ultralytics opencv-python Pillow numpy pandas tqdm albumentations onnx onnxruntime-gpu

Write-Host "`n=== Environment Setup Verification ===" -ForegroundColor Cyan
& $venvPython -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "`n[SUCCESS] Environment setup complete!" -ForegroundColor Green

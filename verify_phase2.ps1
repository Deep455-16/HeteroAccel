# verify_phase2.ps1
#
# Run this from the project root in a PowerShell that has cmake and MSVC
# on PATH (e.g. "Developer PowerShell for VS 2026", or a regular
# PowerShell after running vcvarsall.bat / VsDevCmd.bat).
#
# This performs Phase 2 verification steps 1, 2, 3, 7, 8, and 11 from the
# spec (build Release, run the full test suite, run the real Vulkan
# benchmark, and capture command-level evidence for synchronization /
# buffer allocation / pipeline creation / result retrieval) and writes
# every command's full output to phase2_verification_logs\, so nothing
# has to be taken on faith -- just share that folder back.

$ErrorActionPreference = "Continue"
$LogDir = "phase2_verification_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Run-Step {
    param([string]$Name, [scriptblock]$ScriptBlock)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $logFile = Join-Path $LogDir "$Name.log"
    & $ScriptBlock *>&1 | Tee-Object -FilePath $logFile
    "`nExit code: $LASTEXITCODE" | Add-Content $logFile
    Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Yellow
}

# --- Step 1: clean configure + Release build -------------------------------
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
# No -G specified: let CMake auto-detect your installed Visual Studio
# generator rather than us guessing the exact VS2026 generator string.
Run-Step "01_configure" { cmake -S . -B build -A x64 }
Run-Step "02_build_release" { cmake --build build --config Release }

# --- Step 2: full test suite -------------------------------------------
Run-Step "03_ctest_full_suite" { ctest --test-dir build -C Release --output-on-failure }

# Direct runs of the GPU-dependent tests for verbose, per-assertion output
# (these print the real GPU execute time and PASS/FAIL of the correctness
# check line-by-line, which the plain ctest summary doesn't show).
Run-Step "04_test_vulkan_vector_add_correctness" { & ".\build\tests\Release\test_vulkan_vector_add_correctness.exe" }
Run-Step "05_test_vulkan_buffer_transfer" { & ".\build\tests\Release\test_vulkan_buffer_transfer.exe" }
Run-Step "06_test_vulkan_backend_lifecycle" { & ".\build\tests\Release\test_vulkan_backend_lifecycle.exe" }
Run-Step "07_test_vulkan_invalid_buffer_handling" { & ".\build\tests\Release\test_vulkan_invalid_buffer_handling.exe" }

# --- Cross-reference: Phase 1 hardware detection (should still show Iris Xe) ---
Run-Step "08_hardware_json" { & ".\build\Release\adaptive-gpu.exe" hardware --json }

# --- Step 3/6: the real Vulkan benchmark (the Phase 2 headline result) -----
Run-Step "09_benchmark_vulkan" { & ".\build\Release\adaptive-gpu.exe" benchmark vulkan }

Write-Host "`nDone. Please share the contents of .\$LogDir\ (or paste the terminal output) back for verification." -ForegroundColor Green

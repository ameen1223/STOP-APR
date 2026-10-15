$ErrorActionPreference = 'Stop'
$root = 'D:\IJIES_STOP_APR'
$pythonExe = Join-Path $root '.venv\Scripts\python.exe'
$driver = Join-Path $root 'scripts\07_STOP_APR_LeakageFreeAblation.py'
$logs = Join-Path $root 'logs'

if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Missing Python environment: $pythonExe" }
if (-not (Test-Path -LiteralPath $driver)) { throw "Missing Phase07 script: $driver" }
if (-not (Test-Path -LiteralPath (Join-Path $root 'scripts\02_STOP_APR_FullProtocol.py'))) { throw 'Missing original Phase02 script.' }
if (-not (Test-Path -LiteralPath (Join-Path $root 'scripts\06_STOP_APR_EmpiricalFinal.py'))) { throw 'Missing original Phase06 script.' }
New-Item -ItemType Directory -Path $logs -Force | Out-Null

Write-Host 'START: STOP-APR Phase 07 v2 future-free replay + ablation'
& $pythonExe -u $driver --root $root
if ($LASTEXITCODE -ne 0) { throw 'Phase07 failed; send the PowerShell error or screenshot.' }

$needed = @(
  '07_leakage_free_summary.json',
  '07_corrected_comparison.csv',
  '07_component_ablation.csv',
  '07_old_vs_corrected.csv',
  '07_all_policy_results.csv',
  '07_split_diagnostics.csv',
  '07_strict_temporal_audit.json'
)
$found = @()
foreach ($name in $needed) {
  $path = Join-Path $logs $name
  if (Test-Path -LiteralPath $path) { $found += $path }
}
$zipPath = Join-Path $logs '07_STOP_APR_Audit_Results.zip'
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -LiteralPath $found -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host 'STATUS: COMPLETE'
Write-Host "UPLOAD: $zipPath"

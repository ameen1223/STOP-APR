$ErrorActionPreference = 'Stop'
$root = 'D:\IJIES_STOP_APR'
$pythonExe = Join-Path $root '.venv\Scripts\python.exe'
$driver = Join-Path $root 'scripts\08_STOP_APR_FinalEvidence.py'
$logs = Join-Path $root 'logs'

if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Python environment missing: $pythonExe" }
if (-not (Test-Path -LiteralPath $driver)) { throw "Runner missing: $driver" }
New-Item -ItemType Directory -Path $logs -Force | Out-Null
Write-Host 'START: STOP-APR Phase 08 final evidence, frozen Phase07'
& $pythonExe -u $driver --root $root
if ($LASTEXITCODE -ne 0) { throw 'Phase08 failed: send the last PowerShell error. Previous results were not changed.' }

$needed = @(
  '08_final_evidence_summary.json',
  '08_frozen_reproduction.csv',
  '08_paired_burden_comparison.csv',
  '08_single_event_cohort_sensitivity.csv',
  '08_calibration_sensitivity.csv',
  '08_component_subgroup_diagnostics.csv',
  '08_temporal_covariate_drift.csv',
  '08_agent_composition.csv',
  '08_cohort_overall.csv',
  '08_cohort_by_agent.csv'
)
$paths = @()
foreach ($name in $needed) {
  $p = Join-Path $logs $name
  if (-not (Test-Path -LiteralPath $p)) { throw "Missing final output: $p" }
  $paths += $p
}
$out = Join-Path $logs '08_STOP_APR_FinalEvidence.zip'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
Compress-Archive -LiteralPath $paths -DestinationPath $out -CompressionLevel Optimal
Write-Host 'STATUS: COMPLETE'
Write-Host "UPLOAD: $out"

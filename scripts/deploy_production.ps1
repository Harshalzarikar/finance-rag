# One-command public production deploy (Windows PowerShell)
#
# Prerequisites: Docker Desktop, DNS A record → this machine, ports 80/443 open.
#
#   .\scripts\deploy_production.ps1 -Domain app.example.com -Email ops@example.com
#
param(
    [Parameter(Mandatory = $true)]
    [string] $Domain,
    [Parameter(Mandatory = $true)]
    [string] $Email,
    [string] $GroqKey = "",
    [string] $CohereKey = "",
    [switch] $Saas,
    [switch] $SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$setupArgs = @(
    "scripts/setup_production_env.py",
    "--domain", $Domain,
    "--email", $Email,
    "--force"
)
if ($GroqKey) { $setupArgs += @("--groq-key", $GroqKey) }
if ($CohereKey) { $setupArgs += @("--cohere-key", $CohereKey) }
if ($Saas) { $setupArgs += "--saas" }

python @setupArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/verify_production_env.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SkipBuild) {
    docker compose up -d
} else {
    docker compose up -d --build
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker compose --profile tools up -d worker

Write-Host ""
Write-Host "Deployed. Wait for health, then open: https://$Domain"
Write-Host "Health: curl -fsS https://$Domain/api/health/ready"

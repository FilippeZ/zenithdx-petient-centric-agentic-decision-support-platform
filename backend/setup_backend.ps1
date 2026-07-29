#!/usr/bin/env pwsh
# =============================================================================
#  ZenithDx Backend - Ollama Setup Script
#  1. Checks / installs Ollama
#  2. Starts Ollama serve
#  3. Pulls llama3.2:3b and creates the doctor2 model
# =============================================================================

$ErrorActionPreference = "Stop"
$BackendDir = $PSScriptRoot

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  ZenithDx - Ollama Setup" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# ── Step 1: Check Ollama ──────────────────────────────────────────────────────
Write-Host "[1/3] Checking Ollama installation..." -ForegroundColor Yellow
$OllamaCmd = Get-Command ollama -ErrorAction SilentlyContinue

if ($null -eq $OllamaCmd) {
    Write-Host "  Ollama not found. Downloading installer..." -ForegroundColor Yellow
    $InstallerPath = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $InstallerPath -UseBasicParsing
    Write-Host "  Installing Ollama (silent)..." -ForegroundColor Yellow
    Start-Process -FilePath $InstallerPath -ArgumentList "/S" -Wait
    Write-Host "  Ollama installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  IMPORTANT: Close this terminal and re-open it, then run this script again." -ForegroundColor Red
    exit 0
}

Write-Host "  OK - Ollama found at $($OllamaCmd.Source)" -ForegroundColor Green

# ── Step 2: Start Ollama serve ────────────────────────────────────────────────
Write-Host "`n[2/3] Ensuring Ollama server is running..." -ForegroundColor Yellow

$serverRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3
    if ($response.StatusCode -eq 200) {
        $serverRunning = $true
    }
} catch {
    $serverRunning = $false
}

if ($serverRunning) {
    Write-Host "  OK - Ollama server already running." -ForegroundColor Green
} else {
    Write-Host "  Starting Ollama server in background..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Write-Host "  OK - Ollama server started." -ForegroundColor Green
}

# ── Step 3: Create doctor2 model ─────────────────────────────────────────────
Write-Host "`n[3/3] Setting up doctor2 model..." -ForegroundColor Yellow

$tagsRaw = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing | ConvertFrom-Json
$existingModels = $tagsRaw.models | ForEach-Object { $_.name }

$doctor2Exists = ($existingModels -contains "doctor2:latest") -or ($existingModels -contains "doctor2")

if ($doctor2Exists) {
    Write-Host "  OK - doctor2 model already exists." -ForegroundColor Green
} else {
    Write-Host "  Pulling llama3.2:3b base model (may take a few minutes on first run)..." -ForegroundColor Yellow
    & ollama pull llama3.2:3b
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to pull llama3.2:3b" -ForegroundColor Red
        exit 1
    }

    Write-Host "  Creating doctor2 from Modelfile..." -ForegroundColor Yellow
    Set-Location "$PSScriptRoot/../ollama"
    & ollama create doctor2 -f Modelfile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create doctor2 model" -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK - doctor2 model created!" -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Setup complete! Ollama is running with doctor2." -ForegroundColor Green
Write-Host "  Now start the backend: python -m uvicorn main:app --port 8000" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

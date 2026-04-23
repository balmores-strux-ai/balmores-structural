#Requires -Version 5.1
<#
  ONE-SHOT RELEASE: Knowledge-Panel SEO update for balmoreslab.com.

  Run this from the project root:
      .\SHIP-SEO.ps1

  What it does, in order:
    1. Verifies (and if needed, prompts) GitHub CLI authentication.
    2. Pushes every updated SEO file to balmores-strux-ai/balmores-structural
       via the GitHub REST Contents API -- no git.exe required.
    3. Waits 90 seconds for Render to auto-build www.balmoreslab.com.
    4. Fires every legitimate crawl-accelerator: IndexNow (Bing, Yandex,
       Seznam), Google + Bing sitemap pings, and opens Google Search
       Console's "Request Indexing" deep links for / and /about.
#>

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { $gh = "gh" }

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " BALMORES LAB -- Knowledge Panel SEO Release " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: auth check ---------------------------------------------------
$null = & $gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "[1/4] GitHub CLI is not authenticated." -ForegroundColor Yellow
  Write-Host "      Launching browser login. Accept the permission prompt," -ForegroundColor Yellow
  Write-Host "      paste the one-time code gh shows you, then come back." -ForegroundColor Yellow
  Write-Host ""
  & $gh auth login --hostname github.com --web --git-protocol https
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Authentication failed. Aborting." -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host "[1/4] GitHub CLI already authenticated." -ForegroundColor Green
}

# --- Step 2: push via REST API (delegates to the existing script) ---------
Write-Host ""
Write-Host "[2/4] Pushing SEO files to balmores-strux-ai/balmores-structural..." -ForegroundColor Cyan
& (Join-Path $repoRoot "scripts\push-seo-via-gh-api.ps1")
# Note: push-seo-via-gh-api.ps1 already handles the 90s Render wait (step 3)
# and calls ping-crawlers.ps1 (step 4) on completion.

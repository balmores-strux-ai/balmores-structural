#Requires -Version 5.1
# Push files without the post-push wait + ping crawler step.
$ErrorActionPreference = "Continue"

$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { $gh = "gh" }

$owner  = "balmores-strux-ai"
$repo   = "balmores-structural"
$branch = "main"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

$files = @(
  "frontend/app/layout.tsx",
  "frontend/app/page.tsx",
  "frontend/app/about/page.tsx",
  "frontend/app/cv/page.tsx",
  "frontend/app/research/page.tsx",
  "frontend/app/sitemap.ts",
  "frontend/app/robots.ts",
  "frontend/app/manifest.ts",
  "frontend/app/opengraph-image.tsx",
  "frontend/app/about/opengraph-image.tsx",
  "frontend/app/feed.xml/route.ts",
  "frontend/app/.well-known/webfinger/route.ts",
  "frontend/public/seo-schema.json",
  "frontend/public/ykjm52si9r4gfvwhul8ob7cd3nqxpe01.txt",
  "frontend/public/humans.txt",
  "frontend/public/llms.txt",
  "frontend/public/foaf.rdf",
  "frontend/public/.well-known/security.txt",
  "CITATION.cff",
  "codemeta.json",
  "scripts/ping-crawlers.ps1",
  "scripts/push-seo-via-gh-api.ps1",
  "scripts/push-only.ps1",
  "SHIP-SEO.ps1",
  "seo/KNOWLEDGE-PANEL-KIT.md",
  "seo/louie-balmores-bios.md",
  "seo/wikidata-louie-balmores.quickstatements.tsv",
  "seo/wikidata-README.md",
  "seo/github-profile-README.md"
)

function Get-RemoteSha($path) {
  $enc  = [System.Uri]::EscapeUriString($path)
  $json = & $gh api "/repos/$owner/$repo/contents/$enc`?ref=$branch" 2>&1
  if ($LASTEXITCODE -ne 0) { return $null }
  try   { return ($json | ConvertFrom-Json).sha } catch { return $null }
}

$success = 0
$failed  = 0

foreach ($rel in $files) {
  $full = Join-Path $repoRoot $rel
  if (-not (Test-Path $full)) {
    Write-Host "SKIP (missing): $rel" -ForegroundColor DarkYellow
    continue
  }
  $bytes  = [IO.File]::ReadAllBytes($full)
  $b64    = [Convert]::ToBase64String($bytes)
  $sha    = Get-RemoteSha $rel

  $body = @{
    message = "SEO: Knowledge Panel update for Louie Doniego Balmores - $rel"
    content = $b64
    branch  = $branch
  }
  if ($sha) { $body.sha = $sha }

  $tmp = New-TemporaryFile
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($tmp, ($body | ConvertTo-Json -Depth 6 -Compress), $utf8NoBom)

  $enc = [System.Uri]::EscapeUriString($rel)
  $action = if ($sha) { "update" } else { "create" }
  $out = & $gh api --method PUT "/repos/$owner/$repo/contents/$enc" --input $tmp 2>&1
  $ok = ($LASTEXITCODE -eq 0)
  Remove-Item $tmp -Force

  if ($ok) {
    Write-Host ("  [OK]   {0,-8} {1}" -f $action, $rel) -ForegroundColor Green
    $success++
  } else {
    Write-Host ("  [FAIL] {0,-8} {1}" -f $action, $rel) -ForegroundColor Red
    Write-Host "         $out" -ForegroundColor DarkRed
    $failed++
  }
}

Write-Host ""
Write-Host ("Pushed {0} files, {1} failed." -f $success, $failed) -ForegroundColor Cyan
if ($failed -eq 0) {
  Write-Host "All good. Render will now auto-build balmoreslab.com." -ForegroundColor Green
}

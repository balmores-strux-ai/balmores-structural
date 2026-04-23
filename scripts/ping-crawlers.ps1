#Requires -Version 5.1
<#
  Fires every legitimate crawl-accelerator available, in order of speed:

    1. IndexNow  -> Bing, Yandex, Seznam (instant). Google has confirmed it
       is evaluating IndexNow signals and major platforms forward submissions.
    2. Google sitemap ping (still honored for sitemap refresh even though the
       /ping endpoint is officially deprecated).
    3. Bing sitemap ping (officially supported).
    4. Opens Google Search Console URL-inspection deep links in the default
       browser for 1-click "Request Indexing" on "/" and "/about".

  Safe to re-run. No credentials required.
#>

$ErrorActionPreference = "Continue"

$SITE       = "https://www.balmoreslab.com"
$HOST_NOSCHEME = "www.balmoreslab.com"
$KEY        = "ykjm52si9r4gfvwhul8ob7cd3nqxpe01"
$KEY_URL    = "$SITE/$KEY.txt"
$SITEMAP    = "$SITE/sitemap.xml"

$URLS = @(
  "$SITE/",
  "$SITE/about",
  "$SITE/cv",
  "$SITE/research",
  "$SITE/seo-schema.json",
  "$SITE/sitemap.xml",
  "$SITE/robots.txt",
  "$SITE/feed.xml",
  "$SITE/humans.txt",
  "$SITE/llms.txt",
  "$SITE/foaf.rdf",
  "$SITE/.well-known/webfinger"
)

Write-Host ""
Write-Host "=== 1/4  IndexNow  (Bing, Yandex, Seznam) ===" -ForegroundColor Cyan
$body = @{
  host        = $HOST_NOSCHEME
  key         = $KEY
  keyLocation = $KEY_URL
  urlList     = $URLS
} | ConvertTo-Json -Depth 4

# IndexNow central endpoint forwards to all participating engines.
try {
  $r = Invoke-WebRequest -Uri "https://api.indexnow.org/indexnow" `
    -Method POST -ContentType "application/json; charset=utf-8" `
    -Body $body -UseBasicParsing -TimeoutSec 15
  Write-Host ("  api.indexnow.org -> HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
} catch {
  Write-Host ("  api.indexnow.org -> {0}" -f $_.Exception.Message) -ForegroundColor Yellow
}

# Also hit Bing + Yandex directly in case the central hub is slow.
foreach ($ep in @("https://www.bing.com/indexnow", "https://yandex.com/indexnow")) {
  try {
    $r = Invoke-WebRequest -Uri $ep -Method POST `
      -ContentType "application/json; charset=utf-8" `
      -Body $body -UseBasicParsing -TimeoutSec 15
    Write-Host ("  $ep -> HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
  } catch {
    Write-Host ("  $ep -> {0}" -f $_.Exception.Message) -ForegroundColor Yellow
  }
}

Write-Host ""
Write-Host "=== 2/4  Google sitemap ping ===" -ForegroundColor Cyan
$googlePing = "https://www.google.com/ping?sitemap=" + [Uri]::EscapeDataString($SITEMAP)
try {
  $r = Invoke-WebRequest -Uri $googlePing -UseBasicParsing -TimeoutSec 15
  Write-Host ("  google.com/ping -> HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
} catch {
  Write-Host ("  google.com/ping -> {0}" -f $_.Exception.Message) -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 3/4  Bing sitemap ping ===" -ForegroundColor Cyan
$bingPing = "https://www.bing.com/ping?sitemap=" + [Uri]::EscapeDataString($SITEMAP)
try {
  $r = Invoke-WebRequest -Uri $bingPing -UseBasicParsing -TimeoutSec 15
  Write-Host ("  bing.com/ping -> HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
} catch {
  Write-Host ("  bing.com/ping -> {0}" -f $_.Exception.Message) -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 4/4  Opening Google Search Console 'Request Indexing' deep links ===" -ForegroundColor Cyan
Write-Host "  (one tab per URL -- click 'REQUEST INDEXING' in each)"

foreach ($u in @("$SITE/", "$SITE/about", "$SITE/cv", "$SITE/research")) {
  $inspect = "https://search.google.com/search-console/inspect?resource_id=" `
             + [Uri]::EscapeDataString("sc-domain:balmoreslab.com") `
             + "&id=" + [Uri]::EscapeDataString($u)
  Start-Process $inspect
  Start-Sleep -Milliseconds 400
}

Write-Host ""
Write-Host "All crawl pings dispatched." -ForegroundColor Green
Write-Host "Expected result timing:"
Write-Host "  * Bing / Yandex:  minutes to a few hours"
Write-Host "  * Google index:   typically 1-24 hours after Request Indexing"
Write-Host "  * Knowledge Panel consideration: days to weeks (identity graph needs cross-platform validation)"

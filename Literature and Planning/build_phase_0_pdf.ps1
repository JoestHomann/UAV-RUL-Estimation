param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "phase_0_dataAnalysis.pdf")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$sourcePath = Join-Path $PSScriptRoot "phase_0_dataAnalysis.md"
$layoutRoot = Join-Path $PSScriptRoot "pdf_layout"
$headerPath = Join-Path $layoutRoot "phase_0_layout.tex"
$filterPath = Join-Path $layoutRoot "phase_0_structure.lua"
$repositoryRoot = Split-Path $PSScriptRoot -Parent

$pandocCommand = Get-Command pandoc -ErrorAction SilentlyContinue
if ($pandocCommand) {
    $pandocPath = $pandocCommand.Source
} else {
    $pandocPath = Join-Path $env:LOCALAPPDATA "Pandoc\pandoc.exe"
}

if (-not (Test-Path -LiteralPath $pandocPath)) {
    throw "Pandoc was not found. Install it with: winget install --id JohnMacFarlane.Pandoc --exact"
}

$resourcePath = "$PSScriptRoot;$repositoryRoot"
$arguments = @(
    $sourcePath
    "--from=gfm"
    "--standalone"
    "--toc"
    "--toc-depth=2"
    "--resource-path=$resourcePath"
    "--lua-filter=$filterPath"
    "--include-in-header=$headerPath"
    "--pdf-engine=xelatex"
    "--metadata", "title=UAV Remaining Useful Life Estimation"
    "--variable", "papersize=a4"
    "--variable", "fontsize=11pt"
    "--variable", "monofont=Noto Sans Mono"
    "--variable", "geometry:top=24mm,bottom=23mm,left=22mm,right=22mm"
    "--variable", "linestretch=1.06"
    "--variable", "colorlinks=true"
    "--variable", "linkcolor=blue"
    "--variable", "urlcolor=blue"
    "--output=$OutputPath"
)

& $pandocPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Pandoc exited with code $LASTEXITCODE."
}

Write-Host "Created $OutputPath"

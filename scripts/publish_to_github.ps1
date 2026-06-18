param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUser,

    [string]$RepoName = "hf-digital-mode-iq-dataset",

    [switch]$Private
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found. Install Git for Windows first."
}

$visibility = "--public"
if ($Private) {
    $visibility = "--private"
}

if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

git add .
git commit -m "Initial release of HF digital-mode IQ dataset pipeline"

if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh repo create "$GitHubUser/$RepoName" $visibility --source . --remote origin --push
} else {
    git remote add origin "https://github.com/$GitHubUser/$RepoName.git"
    git push -u origin main
    Write-Host "Repository pushed. If this fails, create the GitHub repository in the web UI first."
}

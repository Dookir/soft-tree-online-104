$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    $tools = Join-Path $PSScriptRoot '.tools'
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    $settings = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'config/uv-wheels.json') | ConvertFrom-Json
    $row = $settings.platforms.'windows-x86_64'
    $uv = Join-Path $tools 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        Write-Host 'Downloading the fixed-version installer...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $wheel = Join-Path $tools 'uv-bootstrap.whl'
        Invoke-WebRequest -UseBasicParsing -Uri $row.url -OutFile $wheel
        $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $wheel).Hash.ToLowerInvariant()
        if ($sha -ne $row.sha256) { throw 'Installer SHA256 mismatch.' }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [IO.Compression.ZipFile]::OpenRead($wheel)
        try {
            $entries = @($archive.Entries | Where-Object { [IO.Path]::GetFileName($_.FullName) -eq 'uv.exe' -and $_.Length -gt 1000000 })
            if ($entries.Count -ne 1) { throw 'Unexpected installer archive layout.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entries[0], $uv, $true)
        } finally { $archive.Dispose() }
    }
    $version = & $uv --version
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^uv 0\.13\.7\b') { throw 'Wrong installer version.' }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $tools 'python'
    $env:UV_CACHE_DIR = Join-Path $tools 'cache'
    $python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        & $uv --system-certs venv --python 3.10.21 (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
    }
    & $python (Join-Path $PSScriptRoot 'bootstrap.py') --uv $uv
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation or environment checks failed.' }
    Write-Host 'Setup finished. Run 02_REPRODUCE_WINDOWS.bat next.'
} catch {
    Write-Error $_
    exit 1
}

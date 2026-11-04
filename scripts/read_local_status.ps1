param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'
$StatusRepo = Split-Path -Parent $PSScriptRoot
$StatusPath = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $StatusRepo $Path }
$StatusPath = [IO.Path]::GetFullPath($StatusPath)
if ([IO.Path]::GetPathRoot($StatusPath) -ne 'F:\') { throw 'Project status must be on F:.' }
# Unlike a default reader, this handle permits the writer's atomic rename.
$StatusShare = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
$StatusStream = [IO.File]::Open($StatusPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, $StatusShare)
$StatusReader = $null
try {
    $StatusReader = [IO.StreamReader]::new($StatusStream, [Text.Encoding]::UTF8)
    $StatusText = $StatusReader.ReadToEnd()
} finally {
    if ($null -ne $StatusReader) { $StatusReader.Dispose() } else { $StatusStream.Dispose() }
}
$StatusText

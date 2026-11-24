<#
.SYNOPSIS
Launch an isolated, opt-in UFC 6 timing diagnostic without replacing the installed executable.
.DESCRIPTION
Uses the emulator's virtual-pad file protocol, not global keyboard input. No saved settings or
patches are edited. The game's ordinary caches and save directory remain in WorkingDirectory.
The caller navigates the game and closes it after collecting the required scene windows.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $EmulatorPath,
    [Parameter(Mandatory = $true)] [string] $GamePath,
    [Parameter(Mandatory = $true)] [string] $WorkingDirectory,
    [Parameter(Mandatory = $true)] [string] $PatchPath,
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [ValidateSet(60, 120, 240)] [int] $Vblank = 120,
    [switch] $VertexIndexProbe
)
$ErrorActionPreference = 'Stop'
if (Get-Process -Name kyty_emulator -ErrorAction SilentlyContinue) {
    throw 'An emulator is already running. Leave the existing session untouched.'
}
$exe = (Resolve-Path -LiteralPath $EmulatorPath).Path
$game = (Resolve-Path -LiteralPath $GamePath).Path
$working = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$patch = (Resolve-Path -LiteralPath $PatchPath).Path
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh output directory for each run.' }
$output = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
$variables = @{
    KYTY_PERF_CSV = "$output\perf.csv"
    KYTY_PAD_FILE = "$output\pad.txt"
    KYTY_TIME_PROBE = "$output\timing.txt"
    KYTY_TIME_PROBE_CALLER = '9014ff59c'
    KYTY_TIME_PROBE_TRACE_PARENT = '9000351b3'
    KYTY_TIME_PROBE_DUMP = '9014ff59c:0:20'
    KYTY_TIME_PROBE_DUMP_REGISTER = 'stack-r14'
    KYTY_TIME_PROBE_DUMP_PARENT = '9000351b3'
    KYTY_TIME_PROBE_DUMP_MS = '0'
    KYTY_PROBE_VERTEX_INDEX = $(if ($VertexIndexProbe) { '1' } else { '0' })
    VK_LOADER_LAYERS_DISABLE = '~implicit~'
}
$saved = @{}
foreach ($name in $variables.Keys) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $variables[$name], 'Process')
}
try {
    $arguments = @('--game-patch', $patch, '--screen-width', '1280', '--screen-height', '720',
        '--user-name', 'Player', '--user-id', '1000', '--controller-volume', '50',
        '--controller-vibration', '0', '--keyboard-player', '0', '--present-mode', 'Mailbox',
        '--gpu', '0', '--no-trophy-notifications', '--skip-notice-screen', '--tessellation',
        '--shader-cache', '--vblank-frequency', "$Vblank", '--console-language', '1',
        '--shader-optimization-type', 'Performance', '--shader-log-direction', 'Silent',
        '--printf-direction', 'Silent', '--redzone', '--game', $game)
    # These arguments are data passed directly to the executable, never shell commands.
    $commandLine = ($arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' '
    $metadata = [ordered]@{
        started_utc = [DateTime]::UtcNow.ToString('o')
        executable = $exe
        executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
        patch = $patch
        patch_sha256 = (Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash
        vblank = $Vblank
        command_line = $commandLine
        environment = $variables
        note = 'Diagnostic build; engine timing is not proof of animation or physics speed.'
    }
    $process = Start-Process -FilePath $exe -ArgumentList $commandLine -WorkingDirectory $working `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$output\console.txt" `
        -RedirectStandardError "$output\console-err.txt"
    $metadata.process_id = $process.Id
    $metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$output\run.json" -Encoding UTF8
    $metadata | ConvertTo-Json -Depth 4
} finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}

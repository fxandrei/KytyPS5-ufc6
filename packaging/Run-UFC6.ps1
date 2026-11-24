[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$GamePath,
    [switch]$Output1080p,
    [switch]$ExperimentalUnmappedBuffers,
    [switch]$OriginalTiming
)
$ErrorActionPreference='Stop'
$runtimeRoot=$PSScriptRoot
$emulator=Join-Path $runtimeRoot 'kyty_emulator.exe'
if(-not (Test-Path -LiteralPath $emulator)){throw 'Extract the complete release archive before running this script.'}
$game=(Resolve-Path -LiteralPath $GamePath).Path
if(-not (Test-Path -LiteralPath (Join-Path $game 'eboot.bin'))){throw 'GamePath must point to the extracted game directory containing eboot.bin.'}
$arguments=@('--game',$game,'--shader-cache','--screen-width','1280','--screen-height','720',
    '--vblank-frequency','120','--skip-notice-screen','--printf-direction','Silent')
if(-not $OriginalTiming){
    $arguments+=@('--game-patch',(Join-Path $runtimeRoot 'patches\ufc6-realtime-25ms.json'))
}
$names=@('KYTY_OUTPUT_1080P','KYTY_DEVICE_LOCAL_GUEST_BUFFERS','KYTY_HOST_BARRIER_AB')
$previous=@{}
foreach($name in $names){$previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try {
    $env:KYTY_OUTPUT_1080P=if($Output1080p){'1'}else{'0'}
    $env:KYTY_DEVICE_LOCAL_GUEST_BUFFERS=if($ExperimentalUnmappedBuffers){'1'}else{'0'}
    $env:KYTY_HOST_BARRIER_AB='0'
    Push-Location $runtimeRoot
    try { & $emulator @arguments; $gameExit=$LASTEXITCODE } finally { Pop-Location }
} finally {
    foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$previous[$name],'Process')}
}
exit $gameExit

<#
.SYNOPSIS
Read UFC 6's round HUD from the emulator test window and measure its rate against a monotonic clock.
.DESCRIPTION
No input is sent. Missing/ambiguous clocks and HUD layout changes break measurement segments.
HUD placement is not a combat-state classifier: top HUD can appear while fighters stand.
Accelerated mode is reported as a measured rate, never assumed to be one displayed second per real second.
Screenshot/OCR work can perturb performance; use a separate run for small FPS A/B comparisons.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][int]$GameProcessId,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(1,86400)][int]$Seconds=60,
    [ValidateRange(0.5,60)][double]$IntervalSeconds=2,
    [ValidateSet('Unknown','Accelerated','RealTime')][string]$ClockMode='Unknown',
    [ValidateRange(0.01,20)][double]$ExpectedRate=1,
    [switch]$CheckExpectedRate,
    [string]$ExpectedExecutable='D:\games\_kytyps5\kyty_emulator.exe',
    [string]$HarnessPath='D:\Projects\kyty-dev\kyty-dev.ps1'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSScriptRoot 'Ufc6RoundClock.psm1') -Force
$game=Get-Process -Id $GameProcessId
if($game.Path -ne $ExpectedExecutable) { throw 'Unexpected emulator executable.' }
$null=New-Item -ItemType Directory -Force -Path $OutputDirectory
$output=(Resolve-Path -LiteralPath $OutputDirectory).Path
$shots=Join-Path $output 'shots'
$null=New-Item -ItemType Directory -Force -Path $shots
$parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($HarnessPath,[ref]$null,[ref]$parseErrors)
if($parseErrors.Count) { throw 'Cannot parse test harness.' }
$functions=@('Initialize-Win32','Test-BitmapBlack','Get-WindowBitmap','Save-WindowShot')
foreach($node in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $functions -contains $n.Name},$false)) {
    $definition=$node.Extent.Text
    $definition=$definition.Replace('if (Test-BitmapBlack $bmp) { $g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size) }','')
    . ([scriptblock]::Create($definition))
}
Initialize-Win32
$hwnd=[KytyWin32]::FindMainWindow($GameProcessId)
if($hwnd -eq [IntPtr]::Zero) { throw 'Emulator test window not found.' }
$watch=[Diagnostics.Stopwatch]::StartNew()
$rows=[Collections.Generic.List[object]]::new()
$segment=0; $previous=$null; $nextSample=0.0
$csv=Join-Path $output 'clock.csv'
$details=Join-Path $output 'clock-details.jsonl'
if(Test-Path -LiteralPath $csv) { throw 'Choose a fresh output directory for each clock run.' }
while($watch.Elapsed.TotalSeconds -lt $Seconds) {
    if(-not (Get-Process -Id $GameProcessId -ErrorAction SilentlyContinue)) { break }
    $wait=$nextSample-$watch.Elapsed.TotalSeconds
    if($wait -gt 0) { Start-Sleep -Milliseconds ([int]($wait*1000)) }
    if($watch.Elapsed.TotalSeconds -ge $Seconds) { break }
    $start=$watch.Elapsed.TotalSeconds
    $image=Join-Path $shots ('{0:D5}.png' -f $rows.Count)
    $valid=$false; $clock=$null; $reason='capture-failed'; $hudRegion='unknown'; $phase='unknown'; $candidates=@()
    if(Save-WindowShot $hwnd $image) {
        $captureTime=($start+$watch.Elapsed.TotalSeconds)/2
        $read=Read-Ufc6RoundClock -ImagePath $image
        $valid=$read.valid; $clock=$read.clockSeconds; $hudRegion=$read.hudRegion; $phase=$read.phase
        $candidates=$read.candidates
        $reason=if($valid) { 'clock-read' } else { 'missing-or-ambiguous-clock' }
    } else { $captureTime=$watch.Elapsed.TotalSeconds }
    $currentProcess=Get-Process -Id $GameProcessId -ErrorAction SilentlyContinue
    if(-not $currentProcess) { break }
    $title=$currentProcess.MainWindowTitle
    $fpsMatch=[regex]::Match([string]$title,'fps:\s*(\d+(?:\.\d+)?)')
    $fps=if($fpsMatch.Success) { [double]::Parse($fpsMatch.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture) } else { $null }
    $continuous=$valid -and $null -ne $previous -and $previous.valid -and $previous.hudRegion -eq $hudRegion -and
        ($captureTime-$previous.monotonicSeconds) -le ($IntervalSeconds*2.5) -and
        $clock -le $previous.clockSeconds -and ($previous.clockSeconds-$clock) -le (10*($captureTime-$previous.monotonicSeconds)+2)
    if(-not $continuous) { $segment++ }
    $row=[pscustomobject]@{
        utc=[DateTime]::UtcNow.ToString('o');monotonicSeconds=$captureTime;valid=$valid
        hudRegion=$hudRegion;phase=$phase;clockSeconds=$clock;clockMode=$ClockMode;segment=$segment;fps=$fps
        sampleCostMs=1000*($watch.Elapsed.TotalSeconds-$start);reason=$reason;screenshot=$image
    }
    $rows.Add($row)
    $row | Export-Csv -LiteralPath $csv -NoTypeInformation -Append
    [pscustomobject]@{sample=$row;candidates=$candidates} | ConvertTo-Json -Depth 7 -Compress | Add-Content -LiteralPath $details -Encoding UTF8
    $previous=$row
    $nextSample=[math]::Max($nextSample+$IntervalSeconds,$watch.Elapsed.TotalSeconds)
}
# Least-squares slopes over uninterrupted segments of at least 8 seconds. Clock
# quantization is then small compared with the observed interval.
$rates=@($rows | Where-Object valid | Group-Object segment | ForEach-Object {
    $group=@($_.Group)
    if($group.Count -lt 4) { return }
    $span=$group[-1].monotonicSeconds-$group[0].monotonicSeconds
    if($span -lt 8) { return }
    $meanX=($group | Measure-Object monotonicSeconds -Average).Average
    $meanY=($group | Measure-Object clockSeconds -Average).Average
    $numerator=0.0;$denominator=0.0
    foreach($point in $group) {
        $x=$point.monotonicSeconds-$meanX
        $numerator += $x*($point.clockSeconds-$meanY);$denominator += $x*$x
    }
    $rate=-$numerator/$denominator
    $rateErrorPercent=if($CheckExpectedRate) { 100*($rate/$ExpectedRate-1) } else { $null }
    [pscustomobject]@{
        segment=$group[0].segment;hudRegion=$group[0].hudRegion;phase=$group[0].phase;samples=$group.Count;wallSeconds=$span
        displayedSecondsPerWallSecond=$rate
        meanFps=($group | Measure-Object fps -Average).Average
        expectedRate=$(if($CheckExpectedRate){$ExpectedRate}else{$null});percentFromExpected=$rateErrorPercent
        verdict=$(if(-not $CheckExpectedRate){'rate-only'}elseif([math]::Abs($rateErrorPercent) -le 5){'within-5-percent'}else{'outside-5-percent'})
    }
})
$summary=[pscustomobject]@{
    processId=$GameProcessId;clockMode=$ClockMode;samples=$rows.Count
    validSamples=@($rows | Where-Object valid).Count;segments=$rates
    interpretation='HUD clock rate only; animation/physics speed is not independently proven. HUD layout changes, missing HUD and resets are excluded. HUD placement does not identify standing or ground combat.'
    sampleCsv=$csv
}
$summary | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $output 'summary.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 7

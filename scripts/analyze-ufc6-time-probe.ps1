<#
.SYNOPSIS
Summarize an explicitly selected frame event in a KYTY_TIME_PROBE trace.
.DESCRIPTION
Read-only offline analysis. The clock helper is called several times per frame, so choose
both its caller and a game-loop stack address. This measures host intervals between calls,
not the game's actual animation or gameplay speed. Clamp figures are hypotheses only.

Example for UFC 6 01.005.000's GameSimulationLoop:
  .\scripts\analyze-ufc6-time-probe.ps1 -TracePath <trace.txt> `
      -Caller 9014ff59c -StackAddress 9000351b3 -ThreadName GameSimulationLoop

The input must have at least two '# sync TSC QPC QPC_FREQUENCY' lines. No machine-frequency
guess is used. Select a continuous standing-fight window with -FromSeconds/-ToSeconds;
times are relative to the first event in the trace. Menus, intros and ground phases must
be analyzed separately. The script writes JSON to stdout and does not modify the trace.
Exactly 32-byte dumps are decoded as four little-endian u64 timing fields. The selected
probe must have captured R14 output at the first mode-2 clock, i.e. the previous frame's
completed timing. Other registers/locations have different meanings. -AllDumps emits every
decoded row; the default includes a summary and at most -DumpRows examples.
Select -PatchModel explicitly when comparing a different patch: legacy25 (default),
base-budget, none, or unknown. Predictions assume -ExpectedInitialScale (default 1).
Zero output is reported as a possible paused/zero-scale state, with prediction skipped.
New dumps include thread=<id> and are filtered to the selected event thread. Older dumps
remain readable but are reported as unattributed because their thread cannot be verified.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $TracePath,
    [Parameter(Mandatory = $true)] [string] $Caller,
    [Parameter(Mandatory = $true)] [string] $StackAddress,
    [string] $Source = 'ProcessTimeCounter',
    [string] $ThreadName = 'GameSimulationLoop',
    [double] $FromSeconds = 0,
    [double] $ToSeconds = [double]::PositiveInfinity,
    [ValidateSet('legacy25', 'base-budget', 'none', 'unknown')]
    [string] $PatchModel = 'legacy25',
    [ValidateRange(0, 16)] [decimal] $ExpectedInitialScale = 1,
    [ValidateRange(0, 1000000)] [int] $DumpRows = 12,
    [switch] $AllDumps
)

$ErrorActionPreference = 'Stop'
if ($ToSeconds -le $FromSeconds) { throw 'ToSeconds must exceed FromSeconds.' }

function Convert-HexAddress([string] $Value) {
    return [Convert]::ToUInt64(($Value -replace '^0[xX]', ''), 16)
}

function Get-Percentile([double[]] $Sorted, [double] $Fraction) {
    $position = ($Sorted.Count - 1) * $Fraction
    $low = [int][Math]::Floor($position)
    $high = [int][Math]::Ceiling($position)
    return $Sorted[$low] + ($Sorted[$high] - $Sorted[$low]) * ($position - $low)
}

$callerAddress = Convert-HexAddress $Caller
$parentAddress = Convert-HexAddress $StackAddress
$syncs = [Collections.Generic.List[object]]::new()
$selected = [Collections.Generic.List[object]]::new()
$dumps = [Collections.Generic.List[object]]::new()
$otherDumpSizes = 0
$threadNames = @{}
$firstTick = $null
$stream = [IO.FileStream]::new((Resolve-Path -LiteralPath $TracePath).Path,
    [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
$reader = [IO.StreamReader]::new($stream)
try {
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line -match '^# sync (\d+) (\d+) (\d+)$') {
            $syncs.Add([pscustomobject]@{
                ticks = [decimal]$Matches[1]
                seconds = [decimal]$Matches[2] / [decimal]$Matches[3]
            })
            continue
        }
        if ($line -match '^# thread (\d+) (.*)$') {
            $threadNames[$Matches[1]] = $Matches[2]
            continue
        }
        if ($line -match '^# dump (\d+) ([0-9a-fA-F]+) ([0-9a-fA-F]+)(?: thread=(\d+))?$') {
            $tick = [decimal]$Matches[1]
            if ($null -eq $firstTick -or $tick -lt $firstTick) { $firstTick = $tick }
            if ($Matches[3].Length -eq 64) {
                $dumps.Add([pscustomobject]@{
                    ticks = $tick; address = $Matches[2]; hex = $Matches[3]
                    thread = $(if ($Matches.ContainsKey(4)) { $Matches[4] } else { $null })
                })
            } else { ++$otherDumpSizes }
            continue
        }
        if ($line.StartsWith('#') -or [string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\s+'
        if ($parts.Count -lt 5) { continue }
        $tick = [decimal]$parts[0]
        if ($null -eq $firstTick -or $tick -lt $firstTick) { $firstTick = $tick }
        if ($parts[1] -ne $Source -or (Convert-HexAddress $parts[3]) -ne $callerAddress) { continue }
        $hasParent = $false
        for ($i = 5; $i -lt $parts.Count; $i++) {
            if ((Convert-HexAddress $parts[$i]) -eq $parentAddress) { $hasParent = $true; break }
        }
        if ($hasParent) { $selected.Add([pscustomobject]@{ ticks = $tick; thread = $parts[4] }) }
    }
} finally { $reader.Dispose() }

if ($syncs.Count -lt 2) { throw 'At least two sync records are required to calibrate host time.' }
$syncs = @($syncs | Sort-Object ticks)
$syncSpan = $syncs[-1].seconds - $syncs[0].seconds
if ($syncSpan -le 0) { throw 'Sync records do not span positive host time.' }
$tickHz = ($syncs[-1].ticks - $syncs[0].ticks) / $syncSpan
if ($tickHz -le 0) { throw 'Sync records have a nonpositive TSC frequency.' }

$events = @($selected | Where-Object {
    $time = [double](($_.ticks - $firstTick) / $tickHz)
    $time -ge $FromSeconds -and $time -le $ToSeconds -and
        (!$ThreadName -or $threadNames[$_.thread] -eq $ThreadName)
} | Sort-Object ticks)
if ($events.Count -lt 3) { throw 'Fewer than three matching events; check caller, stack, thread and window.' }
$threadIds = @($events | Select-Object -ExpandProperty thread -Unique)
if ($threadIds.Count -ne 1) { throw 'The selected event spans multiple threads; select one ThreadName.' }

$intervals = [Collections.Generic.List[double]]::new()
for ($i = 1; $i -lt $events.Count; $i++) {
    $intervals.Add([double](($events[$i].ticks - $events[$i - 1].ticks) / $tickHz))
}
$wallSeconds = [double](($events[-1].ticks - $events[0].ticks) / $tickHz)
$sortedMs = [double[]]@($intervals | ForEach-Object { $_ * 1000 } | Sort-Object)
$clamped25ms = 0.0
$scale100ms = 0.0
foreach ($dt in $intervals) {
    $clamped25ms += [Math]::Min($dt, 0.025)
    $scale100ms += [Math]::Min($dt, 0.025) * [Math]::Max(1.0, [Math]::Min($dt, 0.1) / 0.025)
}

$dumpsInWindow = @($dumps | Where-Object {
    $time = [double](($_.ticks - $firstTick) / $tickHz)
    $time -ge $FromSeconds -and $time -le $ToSeconds
} | Sort-Object ticks)
$unattributedDumpCount = @($dumpsInWindow | Where-Object { $null -eq $_.thread }).Count
$excludedOtherThreadDumps = @($dumpsInWindow | Where-Object {
    $null -ne $_.thread -and $_.thread -ne $threadIds[0]
}).Count
$decoded = @($dumpsInWindow | Where-Object {
    $null -eq $_.thread -or $_.thread -eq $threadIds[0]
} | ForEach-Object {
    $dump = $_
    $bytes = [byte[]]::new(32)
    for ($i = 0; $i -lt 32; $i++) {
        $bytes[$i] = [Convert]::ToByte($dump.hex.Substring($i * 2, 2), 16)
    }
    # The captured guest is x86-64, with little-endian u64 fields.
    if (![BitConverter]::IsLittleEndian) {
        for ($i = 0; $i -lt 32; $i += 8) { [Array]::Reverse($bytes, $i, 8) }
    }
    $nominal = [BitConverter]::ToUInt64($bytes, 0)
    $base = [BitConverter]::ToUInt64($bytes, 8)
    $elapsed = [BitConverter]::ToUInt64($bytes, 16)
    $wait = [BitConverter]::ToUInt64($bytes, 24)
    # A 32-byte read is not evidence that its address names timing output. Reject implausible
    # layouts before computing any ratio; unwinding through a guest wrapper can expose its
    # canary/allocator data instead of the frame-sync register. Passing these checks still
    # requires independent confirmation of the captured object's identity and field layout.
    $validationErrors = [Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt 32; $i += 4) {
        $word = [BitConverter]::ToUInt32($bytes, $i)
        if ($word -in @([uint32]3735928559, [uint32]3435973836, [uint32]3452816845,
                         [uint32]3722304989, [uint32]4277075694)) {
            $validationErrors.Add('sentinel_or_poison_word')
            break
        }
    }
    if ($base -lt 1000000 -or $base -gt 1000000000) {
        $validationErrors.Add('base_outside_1ms_to_1s')
    }
    if (($nominal -ne 0 -and $nominal -lt 1000) -or
        [decimal]$nominal -gt [decimal]$base * 16) {
        $validationErrors.Add('nominal_nonzero_outside_1us_to_16x_base')
    }
    if ($elapsed -lt 1000 -or $elapsed -gt 60000000000) {
        $validationErrors.Add('elapsed_outside_1us_to_60s')
    }
    if ([decimal]$wait -gt [decimal]$elapsed + 1000000) {
        $validationErrors.Add('wait_exceeds_elapsed_by_more_than_1ms')
    }
    $valid = $validationErrors.Count -eq 0
    $expectedRatio = $null
    $expectedNominal = $null
    $ratio = $null
    $budgetElapsedRatio = $null
    $errorNs = $null
    $mismatch = $null
    $pausedOrZeroScale = $valid -and $nominal -eq 0
    $predictionSkipReason = $null
    $patchMultiplier = $null
    $initialNominal = $null
    if ($valid) {
        $ratio = [double]$nominal / [double]$base
        $budgetElapsedRatio = [double]$nominal / [double]$elapsed
        if ($pausedOrZeroScale) {
            # A zero scale is a legitimate paused state. The output alone cannot prove
            # the incoming scale, so do not compare it against the configured default 1.
            $predictionSkipReason = 'paused_or_zero_scale'
        } elseif ($PatchModel -eq 'unknown') {
            $predictionSkipReason = 'unknown_patch_model'
        } else {
            $cappedElapsed = [decimal][Math]::Min([double]$elapsed, 100000000)
            $patchMultiplier = switch ($PatchModel) {
                'legacy25' {
                    if ($elapsed -gt 25000000) { $cappedElapsed / 25000000 }
                    else { [decimal]1 }
                }
                'base-budget' {
                    # The candidate caps first, then skips adjustment unless the
                    # positive base budget is below that capped elapsed duration.
                    if ($base -gt 0 -and $cappedElapsed -gt $base) {
                        $cappedElapsed / [decimal]$base
                    } else { [decimal]1 }
                }
                'none' { [decimal]1 }
            }
            $initialNominal = [decimal]::Floor([decimal]$base * $ExpectedInitialScale)
            $expectedNominal = [decimal]::Floor($initialNominal * $patchMultiplier)
            $expectedRatio = [double]$expectedNominal / [double]$base
            $errorNs = [decimal]$nominal - $expectedNominal
            # Allow two ns for integer arithmetic/rounding in the cave's scaled budget.
            $mismatch = [Math]::Abs($errorNs) -gt 2
        }
    } else {
        $predictionSkipReason = 'invalid_timing_layout'
    }
    [pscustomobject][ordered]@{
        host_seconds_from_trace_start = [Math]::Round([double](($dump.ticks - $firstTick) / $tickHz), 6)
        host_qpc_seconds = [Math]::Round([double]($syncs[0].seconds + ($dump.ticks - $syncs[0].ticks) / $tickHz), 9)
        guest_address = $dump.address
        host_thread_id = $dump.thread
        thread_attribution = $(if ($null -eq $dump.thread) { 'legacy_unattributed' } else { 'explicit_thread_id' })
        raw_hex = $dump.hex
        nominal_scaled_ns = $nominal
        base_ns = $base
        elapsed_ns = $elapsed
        wait_ns = $wait
        scale_ratio = $ratio
        paused_or_zero_scale = $pausedOrZeroScale
        patch_model = $PatchModel
        expected_initial_scale = $ExpectedInitialScale
        expected_initial_nominal_ns = $initialNominal
        expected_patch_multiplier = $patchMultiplier
        expected_current_patch_ratio = $expectedRatio
        expected_nominal_scaled_ns = $expectedNominal
        patch_prediction_error_ns = $errorNs
        catch_mismatch = $mismatch
        prediction_evaluated = $null -ne $mismatch
        prediction_skip_reason = $predictionSkipReason
        engine_time_budget_ratio_vs_elapsed = $budgetElapsedRatio
        valid_timing_layout = $valid
        validation_errors = @($validationErrors)
        valid_base_and_elapsed = $valid
    }
})
$validDumps = @($decoded | Where-Object valid_base_and_elapsed)
$dumpSummary = [ordered]@{
    selected_event_thread_id = $threadIds[0]
    dumps_in_selected_time_window = $dumpsInWindow.Count
    excluded_other_thread_dumps = $excludedOtherThreadDumps
    unattributed_dump_count = $unattributedDumpCount
    attribution_warning = $(if ($unattributedDumpCount -gt 0) {
        'Legacy dumps contain no thread ID and cannot be filtered. They may contain timing from another thread; independently verify their attribution before comparing phases.'
    } else { $null })
    decoded_32byte_dumps = $decoded.Count
    ignored_other_dump_sizes = $otherDumpSizes
    valid_base_and_elapsed = $validDumps.Count
    valid_timing_layout = $validDumps.Count
    rejected_timing_layout = $decoded.Count - $validDumps.Count
    paused_or_zero_scale_count = @($validDumps | Where-Object paused_or_zero_scale).Count
    patch_model = $PatchModel
    expected_initial_scale = $ExpectedInitialScale
    prediction_evaluated_count = @($decoded | Where-Object prediction_evaluated).Count
    prediction_skipped_count = @($decoded | Where-Object { !$_.prediction_evaluated }).Count
    catch_mismatch_count = @($validDumps | Where-Object catch_mismatch).Count
    mismatch_tolerance_ns = 2
    interpretation = 'Engine timing diagnostic only. Plausibility checks do not establish object identity: independently verify the address/layout. Predictions use the explicit patch_model and expected_initial_scale; paused/zero-scale, unknown-model and invalid-layout rows skip prediction. Valid zero-scale rows retain zero measured ratios. Dump timestamp belongs to the next frame, fields to its previous completed mode-2 timing. Ratios do not prove gameplay speed.'
    rows = $(if ($AllDumps) { @($decoded) } else { @($decoded | Select-Object -First $DumpRows) })
}
if ($validDumps.Count -gt 0) {
    $dumpSummary.scale_ratio_p50 = Get-Percentile ([double[]]@($validDumps.scale_ratio | Sort-Object)) 0.5
    $dumpSummary.engine_time_budget_ratio_vs_elapsed_p50 = Get-Percentile ([double[]]@($validDumps.engine_time_budget_ratio_vs_elapsed | Sort-Object)) 0.5
    $dumpSummary.elapsed_p50_ms = (Get-Percentile ([double[]]@($validDumps.elapsed_ns | Sort-Object)) 0.5) / 1000000
    $dumpSummary.elapsed_p99_ms = (Get-Percentile ([double[]]@($validDumps.elapsed_ns | Sort-Object)) 0.99) / 1000000
}

[ordered]@{
    trace = (Resolve-Path -LiteralPath $TracePath).Path
    event = "$Source/$Caller/stack:$StackAddress"
    thread = $threadNames[$threadIds[0]]
    thread_id = $threadIds[0]
    patch_model = $PatchModel
    expected_initial_scale = $ExpectedInitialScale
    wall_seconds = [Math]::Round($wallSeconds, 6)
    intervals = $intervals.Count
    events_per_second = [Math]::Round($intervals.Count / $wallSeconds, 3)
    mean_interval_ms = [Math]::Round(1000 * $wallSeconds / $intervals.Count, 3)
    interval_p50_ms = [Math]::Round((Get-Percentile $sortedMs 0.5), 3)
    interval_p99_ms = [Math]::Round((Get-Percentile $sortedMs 0.99), 3)
    intervals_over_100ms = @($intervals | Where-Object { $_ -gt 0.1 }).Count
    calibrated_tsc_hz = [Math]::Round([double]$tickHz)
    hypothetical_clamp25_game_seconds_per_wall_second = [Math]::Round($clamped25ms / $wallSeconds, 6)
    hypothetical_clamp25_then_scale100_game_seconds_per_wall_second = [Math]::Round($scale100ms / $wallSeconds, 6)
    actual_gameplay_speed = $null
    dump_diagnostics = $dumpSummary
    limitation = 'Host event intervals and hypothetical clamp models only. Actual speed requires a verified gameplay-time or animation reference; round-clock rate is not assumed to equal real time.'
} | ConvertTo-Json -Depth 6

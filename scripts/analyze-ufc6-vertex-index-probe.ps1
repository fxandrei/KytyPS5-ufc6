<#
.SYNOPSIS
Summarize explicit windows from KYTY_PROBE_VERTEX_INDEX console output without changing the run.
.DESCRIPTION
VI records have no absolute timestamps. List the one-based windows first, then select a known
scene range explicitly. No perf.csv or FPS inference is performed. Timing values are sampled
nanosecond sums, not frame times. Vertex sampled_calls includes empty plans; version 2 also
reports active_calls and active phase sums for calls with a non-empty guest range. Dirty-state
observations were removed in version 2, since the old query could allocate/warm tracker regions.
Unavailable or missing fields remain null. Repeated plan time is only an optimistic bound
before memo-key overhead; ObtainBuffer synchronization cannot simply be skipped.
.EXAMPLE
.\scripts\analyze-ufc6-vertex-index-probe.ps1 -ConsolePath .\console.txt -ListWindows
.EXAMPLE
.\scripts\analyze-ufc6-vertex-index-probe.ps1 -ConsolePath .\console.txt -StartWindow 20 -EndWindow 30 -Label fight
#>
[CmdletBinding(DefaultParameterSetName = 'Range')]
param(
    [Parameter(Mandatory = $true)] [string] $ConsolePath,
    [Parameter(Mandatory = $true, ParameterSetName = 'Range')]
    [ValidateRange(1, 2147483647)] [int] $StartWindow,
    [Parameter(Mandatory = $true, ParameterSetName = 'Range')]
    [ValidateRange(1, 2147483647)] [int] $EndWindow,
    [Parameter(ParameterSetName = 'Range')] [string] $Label = 'unclassified',
    [Parameter(Mandatory = $true, ParameterSetName = 'List')] [switch] $ListWindows
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$culture = [Globalization.CultureInfo]::InvariantCulture

function Read-Fields([string] $Text) {
    $fields = @{}
    foreach ($match in [regex]::Matches($Text, '(?<key>[a-z_]+)=(?<value>[^\s]+)')) {
        $fields[$match.Groups['key'].Value] = $match.Groups['value'].Value
    }
    return $fields
}
function Number($Fields, [string] $Name) {
    if (!$Fields.ContainsKey($Name)) { throw "Missing VI field '$Name'." }
    return [double]::Parse($Fields[$Name], $culture)
}
function Optional-Field($Fields, [string] $Name) {
    if ($Fields.ContainsKey($Name)) { return $Fields[$Name] }
    return $null
}
function Sum-Fields($Records, [string[]] $Names) {
    $result = [ordered]@{}
    foreach ($name in $Names) {
        $total = 0.0
        $present = $true
        foreach ($record in $Records) {
            if (!$record.ContainsKey($name)) { $present = $false; break }
            $total += Number $record $name
        }
        # Missing legacy fields stay null, rather than becoming apparent zero-hit evidence.
        $result[$name] = $(if ($present) { $total } else { $null })
    }
    return $result
}
function Ratio($Numerator, $Denominator, [double] $Scale = 1.0) {
    if ($null -eq $Numerator -or $null -eq $Denominator -or $Denominator -le 0) { return $null }
    return [math]::Round($Scale * $Numerator / $Denominator, 4)
}

$path = (Resolve-Path -LiteralPath $ConsolePath).Path
$windows = [Collections.Generic.List[object]]::new()
$current = $null
$lineNumber = 0
# The emulator can keep the redirected log open for writing throughout a live run.
$stream = [IO.FileStream]::new($path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
    [IO.FileShare]::ReadWrite)
$reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
try {
    while ($null -ne ($line = $reader.ReadLine())) {
        ++$lineNumber
        if ($line -notmatch 'vi-probe\s') { continue }
        $fields = Read-Fields $line
        if ($fields.ContainsKey('seconds') -and $fields.ContainsKey('draws')) {
            $current = [pscustomobject]@{
                Number = $windows.Count + 1; Line = $lineNumber; Stats = $fields
                Program = $null; Timings = @{}
            }
            $windows.Add($current)
        } elseif ($null -ne $current -and $fields.ContainsKey('kind')) {
            $kind = $fields['kind']
            if ($kind -notin @('vertex', 'index', 'vertex_retry', 'index_retry')) { continue }
            if ($current.Timings.ContainsKey($kind)) {
                throw "Duplicate kind '$kind' in window $($current.Number); interleaved records are unsupported."
            }
            $current.Timings[$kind] = $fields
        } elseif ($null -ne $current -and $fields.ContainsKey('program_identity')) {
            $current.Program = $fields
        }
    }
} finally {
    $reader.Dispose()
    $stream.Dispose()
}
$inventory = @($windows | ForEach-Object {
    [pscustomobject]@{
        window = $_.Number; source_line = $_.Line; seconds = Number $_.Stats 'seconds'
        draws = Number $_.Stats 'draws'; complete = $_.Timings.Count -eq 4
        per_program_available = $null -ne $_.Program
        version = Optional-Field $_.Stats 'version'
        dirty_observation = Optional-Field $_.Stats 'dirty_observation'
    }
})
if ($ListWindows) {
    [pscustomobject]@{ console = $path; windows = $inventory } | ConvertTo-Json -Depth 6
    return
}
if ($EndWindow -lt $StartWindow) { throw 'EndWindow must be at least StartWindow.' }
if ($EndWindow -gt $windows.Count) { throw "Only $($windows.Count) VI windows are currently readable." }
$selected = @($windows | Where-Object { $_.Number -ge $StartWindow -and $_.Number -le $EndWindow })
foreach ($window in $selected) {
    if ($window.Timings.Count -ne 4) { throw "Window $($window.Number) is incomplete; choose complete windows." }
    $s = $window.Stats
    if ((Number $s 'vertex_repeat') -gt (Number $s 'vertex_draws') -or
        (Number $s 'index_repeat') -gt (Number $s 'index_draws') -or
        (Number $s 'sampled') -gt (Number $s 'draws')) {
        throw "Inconsistent VI counters in window $($window.Number)."
    }
    foreach ($kind in @('vertex', 'index', 'vertex_retry', 'index_retry')) {
        $t = $window.Timings[$kind]
        if ($t.ContainsKey('active_calls') -and (
            (Number $t 'active_calls') -gt (Number $t 'sampled_calls') -or
            (Number $t 'active_plan_ns') -gt (Number $t 'plan_ns') -or
            (Number $t 'active_slots_ns') -gt (Number $t 'slots_ns'))) {
            throw "Inconsistent active VI timings in window $($window.Number), kind '$kind'."
        }
    }
}
$summary = Sum-Fields @($selected | ForEach-Object { $_.Stats }) @(
    'seconds', 'draws', 'sampled', 'vertex_draws', 'vertex_repeat', 'index_draws',
    'index_repeat', 'host_index_draws', 'merge_retry_passes')
$programRecords = @($selected | Where-Object { $null -ne $_.Program } | ForEach-Object { $_.Program })
$program = $null
if ($programRecords.Count -eq $selected.Count) {
    $program = Sum-Fields $programRecords @('program_cache_hits', 'program_cache_misses',
        'program_cache_evictions', 'vertex_same_program_draws', 'vertex_same_program_compares',
        'vertex_repeat_same_program', 'index_same_program_draws', 'index_same_program_compares',
        'index_repeat_same_program')
    $program['vertex_repeat_pct_all'] = Ratio $program.vertex_repeat_same_program $program.vertex_same_program_draws 100
    $program['vertex_repeat_pct_warm'] = Ratio $program.vertex_repeat_same_program $program.vertex_same_program_compares 100
    $program['index_repeat_pct_all'] = Ratio $program.index_repeat_same_program $program.index_same_program_draws 100
    $program['index_repeat_pct_warm'] = Ratio $program.index_repeat_same_program $program.index_same_program_compares 100
    $program['identity'] = 'module_id (compiled specialization, not a general guest-program cache)'
    $program['capacity'] = [int](Number $programRecords[0] 'program_capacity')
}
$timings = [ordered]@{}
foreach ($kind in @('vertex', 'index', 'vertex_retry', 'index_retry')) {
    $records = @($selected | ForEach-Object { $_.Timings[$kind] })
    $t = Sum-Fields $records @('sampled_calls', 'active_calls', 'plan_ns', 'active_plan_ns',
        'obtain_ns', 'slots_ns', 'active_slots_ns',
        'repeat_plan_ns', 'repeat_calls', 'repeat_obtain_ns', 'ranges', 'cpu_dirty_ranges',
        'stream_ranges', 'repeat_cpu_dirty_ranges', 'repeat_stream_ranges',
        'repeat_same_program_calls', 'repeat_same_program_plan_ns')
    $measured = $t.plan_ns + $t.obtain_ns + $t.slots_ns
    $t['plan_ns_per_sampled_call'] = Ratio $t.plan_ns $t.sampled_calls
    $t['obtain_ns_per_sampled_call'] = Ratio $t.obtain_ns $t.sampled_calls
    $t['slots_ns_per_sampled_call'] = Ratio $t.slots_ns $t.sampled_calls
    $t['active_pct_sampled_calls'] = Ratio $t.active_calls $t.sampled_calls 100
    $t['plan_ns_per_active_call'] = Ratio $t.active_plan_ns $t.active_calls
    # ObtainBuffer is never called for an empty vertex plan or empty index source.
    $t['obtain_ns_per_active_call'] = Ratio $t.obtain_ns $t.active_calls
    $t['slots_ns_per_active_call'] = Ratio $t.active_slots_ns $t.active_calls
    $t['global_repeated_plan_bound_ns_per_sampled_call'] = Ratio $t.repeat_plan_ns $t.sampled_calls
    $t['same_program_repeated_plan_bound_ns_per_sampled_call'] = Ratio $t.repeat_same_program_plan_ns $t.sampled_calls
    $t['same_program_plan_ns_per_repeated_call'] = Ratio $t.repeat_same_program_plan_ns $t.repeat_same_program_calls
    $t['same_program_repeated_plan_bound_pct_measured'] = Ratio $t.repeat_same_program_plan_ns $measured 100
    $t['cpu_dirty_pct_sampled_ranges'] = Ratio $t.cpu_dirty_ranges $t.ranges 100
    $t['stream_pct_sampled_ranges'] = Ratio $t.stream_ranges $t.ranges 100
    $timings[$kind] = $t
}
[pscustomobject]@{
    console = $path; label = $Label; start_window = $StartWindow; end_window = $EndWindow
    selected_windows = @($inventory | Where-Object { $_.window -ge $StartWindow -and $_.window -le $EndWindow })
    counters = $summary
    vertex_repeat_pct_global = Ratio $summary.vertex_repeat $summary.vertex_draws 100
    index_repeat_pct_global = Ratio $summary.index_repeat $summary.index_draws 100
    per_program = $program; timings_sampled = $timings
    interpretation = @(
        'Window numbers follow VI record order; logs have no absolute timestamps or automatic scene labels.'
        'Rates use summed numerators/denominators, not averages of window percentages.'
        'Timing sums are nanoseconds over roughly 1/64 draws; divide by sampled calls, without another factor of 64.'
        'Vertex sampled_calls includes empty plans. Active timings select calls with at least one non-empty guest range; all non-empty index calls are active.'
        'Clock granularity/overhead and probe key/counter work affect small timings; no FPS or ms/frame projection is made.'
        'Repeated plan time is an optimistic bound: descriptor-size/key work and cache overhead may remain.'
        'ObtainBuffer time is not classified as eliminable: CPU uploads, stream lifetimes and synchronization still matter.'
        'Version 2 omits dirty observations (null), avoiding extra tracker allocation, locks and warming before ObtainBuffer.'
        'Legacy dirty and all stream repeat counters describe global repeats; they are not per-program-repeat guards.'
        'A low global-repeat rate does not rule out per-module or other bounded memo policies.'
    )
} | ConvertTo-Json -Depth 9

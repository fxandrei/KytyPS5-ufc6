<#
.SYNOPSIS
Assemble and verify the disabled 20ms UFC 6 timing calibration; never deploy or launch it.
.DESCRIPTION
This is an intermediate experiment after the game-frame-budget version felt too fast.
The generated JSON remains disabled. Arithmetic checks validate the proposed formula,
not animation, physics, audio, or real-time gameplay speed.
#>
[CmdletBinding()]
param([string] $LlvmBin = 'D:\Tools\LLVM\bin')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$output = Join-Path $repo '_Build\diagnostics\20ms-patch'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$source = Join-Path $PSScriptRoot 'ufc6-realtime-20ms.s'
# Both experiments hook the same version-specific sites and share their verified layout.
$linker = Join-Path $PSScriptRoot 'ufc6-realtime-base-budget.ld'
$object = Join-Path $output 'timing.o'
$elf = Join-Path $output 'timing.elf'
$hookFile = Join-Path $output 'hook.bin'
$caveFile = Join-Path $output 'cave.bin'
& (Join-Path $LlvmBin 'clang.exe') --target=x86_64-unknown-linux-gnu -c -x assembler $source -o $object
if ($LASTEXITCODE -ne 0) { throw 'Assembly failed.' }
& (Join-Path $LlvmBin 'ld.lld.exe') -T $linker $object -o $elf
if ($LASTEXITCODE -ne 0) { throw 'Linking failed.' }
& (Join-Path $LlvmBin 'llvm-objcopy.exe') -O binary --only-section=.hook $elf $hookFile
if ($LASTEXITCODE -ne 0) { throw 'Hook extraction failed.' }
& (Join-Path $LlvmBin 'llvm-objcopy.exe') -O binary --only-section=.cave $elf $caveFile
if ($LASTEXITCODE -ne 0) { throw 'Cave extraction failed.' }
$listing = & (Join-Path $LlvmBin 'llvm-objdump.exe') -d -M intel --no-show-raw-insn $elf
if ($LASTEXITCODE -ne 0) { throw 'Disassembly failed.' }
$listingText = $listing -join "`n"
$listingText | Set-Content -LiteralPath (Join-Path $output 'disassembly.txt') -Encoding UTF8
if ($listingText -notmatch 'jmp\s+0x90f000000\b' -or
    $listingText -notmatch 'jmp\s+0x9000353e9\b') {
    throw 'The linked hook/cave jump destinations are incorrect.'
}
if ($listingText -notmatch 'cmp\s+rax,\s+(0x1312d00|20000000)\b' -or
    $listingText -notmatch 'mov\s+ecx,\s+(0x1312d00|20000000)\b' -or
    $listingText -notmatch 'cmp\s+rax,\s+(0x5f5e100|100000000)\b' -or
    $listingText -notmatch 'imul\s+qword ptr \[r14\]' -or
    $listingText -notmatch 'idiv\s+rcx') {
    throw 'The linked cave does not contain the intended 20ms threshold/divisor and 100ms cap.'
}
$hook = [IO.File]::ReadAllBytes($hookFile)
$cave = [IO.File]::ReadAllBytes($caveFile)
$hookHex = ([BitConverter]::ToString($hook) -replace '-', '').ToLowerInvariant()
$caveHex = ([BitConverter]::ToString($cave) -replace '-', '').ToLowerInvariant()
if ($hookHex -ne '4c89f84c29e0e91aacfc0e909090') {
    throw 'The generated hook does not match the verified 14-byte frame-sync hook.'
}
if (!$caveHex.StartsWith('498946184c897b38')) {
    throw 'The cave no longer replays the displaced original stores.'
}

# Arithmetic reference cases validate signed comparison, truncation and cap behavior.
# They do not execute the generated machine code or establish normal gameplay speed.
function Get-CalibratedBudget([long] $initial, [long] $elapsed, [long] $threshold) {
    if ($elapsed -le $threshold) { return $initial }
    $capped = [Math]::Min($elapsed, [long]100000000)
    return [long]([Numerics.BigInteger]::Divide(
        ([Numerics.BigInteger]$initial * [Numerics.BigInteger]$capped),
        [Numerics.BigInteger]$threshold))
}
$nominal = [long]16666666
$cases = @(
    @{ Name='20 FPS'; Initial=$nominal; Elapsed=50000000L; Expected=41666665L },
    @{ Name='30 FPS'; Initial=$nominal; Elapsed=33333333L; Expected=27777776L },
    @{ Name='40 FPS'; Initial=$nominal; Elapsed=25000000L; Expected=20833332L },
    @{ Name='50 FPS'; Initial=$nominal; Elapsed=20000000L; Expected=$nominal },
    @{ Name='60 FPS'; Initial=$nominal; Elapsed=16666666L; Expected=$nominal },
    @{ Name='150ms hitch capped'; Initial=$nominal; Elapsed=150000000L; Expected=83333330L },
    @{ Name='Paused initial scale'; Initial=0L; Elapsed=33333333L; Expected=0L },
    @{ Name='Half initial scale'; Initial=8333333L; Elapsed=33333333L; Expected=13888888L },
    @{ Name='Negative elapsed unchanged'; Initial=$nominal; Elapsed=-1L; Expected=$nominal },
    @{ Name='Below threshold unchanged'; Initial=$nominal; Elapsed=10000000L; Expected=$nominal },
    @{ Name='Zero elapsed unchanged'; Initial=$nominal; Elapsed=0L; Expected=$nominal }
)
$results = foreach ($case in $cases) {
    $actual = Get-CalibratedBudget $case.Initial $case.Elapsed 20000000L
    if ($actual -ne $case.Expected) { throw "Numeric case '$($case.Name)' failed: $actual != $($case.Expected)" }
    [pscustomobject]@{ Case=$case.Name; ElapsedNs=$case.Elapsed; InitialNs=$case.Initial; CandidateNs=$actual; ExpectedNs=$case.Expected }
}
$results | Export-Csv -LiteralPath (Join-Path $output 'numeric-cases.csv') -NoTypeInformation
$ratioResults = foreach ($elapsed in @(50000000L, 33333333L, 25000000L)) {
    $candidate = Get-CalibratedBudget $nominal $elapsed 20000000L
    $legacy = Get-CalibratedBudget $nominal $elapsed 25000000L
    # The ideal rational relationship is 4*candidate == 5*legacy. Integer truncation
    # can differ by less than one candidate nanosecond, hence the four-ns tolerance here.
    $crossError = [Math]::Abs(4L * $candidate - 5L * $legacy)
    if ($crossError -gt 4) { throw "Legacy ratio check failed at elapsed=$elapsed ns." }
    [pscustomobject]@{ ElapsedNs=$elapsed; CandidateNs=$candidate; Legacy25msNs=$legacy; CandidateOverLegacy=[double]$candidate/[double]$legacy; FourCandidateMinusFiveLegacyNs=4L*$candidate-5L*$legacy }
}
$ratioResults | Export-Csv -LiteralPath (Join-Path $output 'legacy-ratio-cases.csv') -NoTypeInformation

$document = [ordered]@{
    id = 'PPSA23566'
    version = '01.005.000'
    process = 'eboot.bin'
    mods = @([ordered]@{
        name = 'EXPERIMENT UNVALIDATED: elapsed / 20ms, capped at 100ms; intermediate calibration, about 1.25x legacy elapsed/25ms at 20-40 FPS'
        enabled = $false
        memory = @(
            [ordered]@{ offset = '0x353db'; off = '4c89f84c29e0498946184c897b38'; on = $hookHex },
            [ordered]@{ offset = '0xf000000'; off = ('00' * $cave.Length); on = $caveHex }
        )
    })
}
$patchFile = Join-Path $PSScriptRoot 'ufc6-realtime-20ms.json'
$json = $document | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($patchFile, $json + "`n", (New-Object Text.UTF8Encoding($false)))
Write-Output "Generated disabled experimental JSON: $patchFile"
Write-Output "Verified hook: $($hook.Length) bytes, cave: $($cave.Length) bytes; disassembly: $output\disassembly.txt"
$results | Format-Table -AutoSize
$ratioResults | Format-Table -AutoSize
Write-Output 'Eleven arithmetic cases and three legacy-ratio checks passed. Game behavior remains unvalidated.'

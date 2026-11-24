<#
.SYNOPSIS
Assemble and verify an opt-in UFC 6 timing experiment; never deploy or launch it.
#>
[CmdletBinding()]
param([string] $LlvmBin = 'D:\Tools\LLVM\bin')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$output = Join-Path $repo '_Build\diagnostics\base-budget-patch'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$source = Join-Path $PSScriptRoot 'ufc6-realtime-base-budget.s'
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
$hook = [IO.File]::ReadAllBytes($hookFile)
$cave = [IO.File]::ReadAllBytes($caveFile)
$hookHex = ([BitConverter]::ToString($hook) -replace '-', '').ToLowerInvariant()
$caveHex = ([BitConverter]::ToString($cave) -replace '-', '').ToLowerInvariant()
if ($hookHex -ne '4c89f84c29e0e91aacfc0e909090') {
    throw 'The generated hook does not match the verified 14-byte frame-sync hook.'
}
$document = [ordered]@{
    id = 'PPSA23566'
    version = '01.005.000'
    process = 'eboot.bin'
    mods = @([ordered]@{
        name = 'EXPERIMENT UNVALIDATED: elapsed / game frame budget, elapsed capped at 100 ms; alternative to elapsed/25 ms'
        enabled = $false
        memory = @(
            [ordered]@{ offset = '0x353db'; off = '4c89f84c29e0498946184c897b38'; on = $hookHex },
            [ordered]@{ offset = '0xf000000'; off = ('00' * $cave.Length); on = $caveHex }
        )
    })
}
$patchFile = Join-Path $PSScriptRoot 'ufc6-realtime-base-budget.json'
$json = $document | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($patchFile, $json + "`n", (New-Object Text.UTF8Encoding($false)))
Write-Output "Generated disabled experimental JSON: $patchFile"
Write-Output "Verified hook: $($hook.Length) bytes, cave: $($cave.Length) bytes; disassembly: $output\disassembly.txt"

# Integer arithmetic reference cases. These validate the proposed formula and
# truncation/cap behavior, not live animation/physics behavior in the game.
function Get-CandidateBudget([long] $initial, [long] $budget, [long] $elapsed) {
    $capped = [Math]::Min($elapsed, [long]100000000)
    if ($budget -le 0 -or $capped -le $budget) { return $initial }
    return [long]([Numerics.BigInteger]::Divide(
        ([Numerics.BigInteger]$initial * [Numerics.BigInteger]$capped),
        [Numerics.BigInteger]$budget))
}
$nominal = [long]16666666
$cases = @(
    @{ Name='20 FPS'; Initial=$nominal; Budget=$nominal; Elapsed=50000000L; Expected=50000000L },
    @{ Name='30 FPS'; Initial=$nominal; Budget=$nominal; Elapsed=33333333L; Expected=33333333L },
    @{ Name='40 FPS'; Initial=$nominal; Budget=$nominal; Elapsed=25000000L; Expected=25000000L },
    @{ Name='60 FPS'; Initial=$nominal; Budget=$nominal; Elapsed=16666666L; Expected=16666666L },
    @{ Name='150 ms hitch'; Initial=$nominal; Budget=$nominal; Elapsed=150000000L; Expected=100000000L },
    @{ Name='Zero budget guard'; Initial=123L; Budget=0L; Elapsed=33333333L; Expected=123L },
    @{ Name='Negative budget guard'; Initial=123L; Budget=-1L; Elapsed=33333333L; Expected=123L },
    @{ Name='Below budget'; Initial=$nominal; Budget=$nominal; Elapsed=10000000L; Expected=$nominal },
    @{ Name='Paused initial scale'; Initial=0L; Budget=$nominal; Elapsed=33333333L; Expected=0L },
    @{ Name='Half initial scale'; Initial=8333333L; Budget=$nominal; Elapsed=33333333L; Expected=16666666L },
    @{ Name='Budget above cap'; Initial=200000000L; Budget=200000000L; Elapsed=300000000L; Expected=200000000L }
)
$results = foreach ($case in $cases) {
    $actual = Get-CandidateBudget $case.Initial $case.Budget $case.Elapsed
    if ($actual -ne $case.Expected) { throw "Numeric case '$($case.Name)' failed: $actual != $($case.Expected)" }
    [pscustomobject]@{ Case=$case.Name; ElapsedNs=$case.Elapsed; InitialNs=$case.Initial; BudgetNs=$case.Budget; CandidateNs=$actual; ExpectedNs=$case.Expected }
}
$results | Export-Csv -LiteralPath (Join-Path $output 'numeric-cases.csv') -NoTypeInformation
$results | Format-Table -AutoSize
Write-Output 'All arithmetic reference cases passed. Game behavior remains unvalidated.'

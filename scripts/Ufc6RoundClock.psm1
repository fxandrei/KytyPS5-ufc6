# OCR of UFC 6's round HUD. This measures the displayed clock; Accelerated mode
# deliberately differs from wall time and is not a direct physics-speed oracle.
Set-StrictMode -Version Latest
$script:ClockOcr=$null
function Initialize-Ufc6ClockOcr {
    if($script:ClockOcr) { return }
    Add-Type -AssemblyName System.Drawing
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
    $null=[Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
    $null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
    $null=[Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
    $null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
    $null=[Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
    $null=[Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
    $script:AsTask=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and
        $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    $language=New-Object Windows.Globalization.Language 'en-US'
    $script:ClockOcr=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
    if(-not $script:ClockOcr) { throw 'Windows English OCR language is unavailable.' }
}
function Wait-Ufc6WinRt($Operation,[Type]$ResultType) {
    $task=$script:AsTask.MakeGenericMethod($ResultType).Invoke($null,@($Operation))
    if(-not $task.Wait(10000)) { throw 'Windows OCR operation timed out.' }
    $task.GetAwaiter().GetResult()
}
function Invoke-Ufc6ImageOcr([string]$Path) {
    $file=Wait-Ufc6WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($Path)) ([Windows.Storage.StorageFile])
    $stream=Wait-Ufc6WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder=Wait-Ufc6WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap=Wait-Ufc6WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        try { Wait-Ufc6WinRt ($script:ClockOcr.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult]) }
        finally { $bitmap.Dispose() }
    } finally { $stream.Dispose() }
}
function Read-Ufc6RoundClock {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ImagePath,[string]$CropDirectory)
    Initialize-Ufc6ClockOcr
    $source=[Drawing.Bitmap]::FromFile((Resolve-Path -LiteralPath $ImagePath).Path)
    # Coordinates include the standard Windows title bar, reference 1302 x 776.
    # Only small known HUD regions are searched, so menus cannot supply arbitrary numbers.
    # HUD placement does not identify combat state: it can move to the top
    # while both fighters remain standing.
    $regions=@(
        @{Name='bottom'; X=610;Y=674;W=84;H=42},
        @{Name='top'; X=613;Y=106;W=78;H=34},
        @{Name='bottom-kumite'; X=620;Y=688;W=64;H=30; Yellow=$true}
    )
    $results=@()
    try {
        foreach($region in $regions) {
            $scaleX=$source.Width/1302.0; $scaleY=$source.Height/776.0
            $rect=[Drawing.Rectangle]::new([int]($region.X*$scaleX),[int]($region.Y*$scaleY),[int]($region.W*$scaleX),[int]($region.H*$scaleY))
            # This HUD font is condensed. An isotropic 4x resize misread 3:39
            # as 1:39. Widening it by 1.5x helps recognition. Require agreement
            # from at least two resize variants and reject conflicting numbers.
            $variants=@()
            $yellow=$region.ContainsKey('Yellow') -and $region.Yellow
            $transforms=@(
                @{Name='NearestNeighbor'; Mode='NearestNeighbor'; Threshold=0},
                @{Name='HighQualityBicubic'; Mode='HighQualityBicubic'; Threshold=0},
                @{Name='Bilinear'; Mode='Bilinear'; Threshold=0}
            )
            if($region.Name -eq 'top') {
                # Top HUD has no solid backplate. Isolating its brightest
                # white pixels can recover the text against the light canvas.
                $transforms+=@(
                    @{Name='White240Nearest'; Mode='NearestNeighbor'; Threshold=240},
                    @{Name='White245Nearest'; Mode='NearestNeighbor'; Threshold=245},
                    @{Name='White245Bilinear'; Mode='Bilinear'; Threshold=245}
                )
            }
            foreach($transform in $transforms) {
                if($transform.Threshold) {
                    $rawReadings=@($variants | Where-Object {
                        $_.mode -in @('NearestNeighbor','HighQualityBicubic','Bilinear') -and $_.matched
                    })
                    $rawGroups=@($rawReadings | Group-Object seconds)
                    # Thresholding can remove thin strokes. Use it only when
                    # the unmodified image could not supply a consensus.
                    if($rawReadings.Count -ge 2 -and $rawGroups.Count -eq 1) { continue }
                }
                $mode=$transform.Mode
                $mask=$null
                if($transform.Threshold -or $yellow) {
                    $mask=[Drawing.Bitmap]::new($rect.Width,$rect.Height)
                    for($y=0;$y -lt $rect.Height;$y++) {
                        for($x=0;$x -lt $rect.Width;$x++) {
                            $pixel=$source.GetPixel($rect.X+$x,$rect.Y+$y)
                            $foreground=if($yellow) {
                                $pixel.R -ge 190 -and $pixel.G -ge 140 -and $pixel.B -lt 120 -and
                                ($pixel.R-$pixel.B) -gt 100 -and ($pixel.G-$pixel.B) -gt 70
                            } else { [Math]::Min($pixel.R,[Math]::Min($pixel.G,$pixel.B)) -gt $transform.Threshold }
                            $color=if($foreground) {
                                [Drawing.Color]::Black
                            } else { [Drawing.Color]::White }
                            $mask.SetPixel($x,$y,$color)
                        }
                    }
                }
                $widthScale=if($yellow) { 5 } else { 6 }
                $crop=[Drawing.Bitmap]::new($rect.Width*$widthScale,$rect.Height*4)
                $graphics=[Drawing.Graphics]::FromImage($crop)
                try {
                    $graphics.InterpolationMode=[Drawing.Drawing2D.InterpolationMode]::$mode
                    if($mask) { $graphics.DrawImage($mask,0,0,$crop.Width,$crop.Height) }
                    else { $graphics.DrawImage($source,[Drawing.Rectangle]::new(0,0,$crop.Width,$crop.Height),$rect,[Drawing.GraphicsUnit]::Pixel) }
                } finally {
                    $graphics.Dispose()
                    if($mask) { $mask.Dispose() }
                }
                $temporary=-not $CropDirectory
                if($temporary) { $path=Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString()+'.png') }
                else {
                    $null=New-Item -ItemType Directory -Force -Path $CropDirectory
                    $path=Join-Path (Resolve-Path -LiteralPath $CropDirectory).Path (([IO.Path]::GetFileNameWithoutExtension($ImagePath))+'-'+$region.Name+'-'+$transform.Name+'.png')
                }
                try {
                    $crop.Save($path,[Drawing.Imaging.ImageFormat]::Png)
                    $ocr=Invoke-Ufc6ImageOcr $path
                    $raw=$ocr.Text.Trim()
                    # Kumite uses a yellow MM:SS clock and may start above
                    # five minutes. Its separate color mask excludes the MMA HUD.
                    $pattern=if($yellow) { '^([0-9]{1,2})\s*[:;.]\s*([0-5]\d)$' } else { '^([0-5])\s*[:;.]\s*([0-5]\d)$' }
                    $maximum=if($yellow) { 5999 } else { 300 }
                    $match=[regex]::Match($raw,$pattern)
                    $seconds=if($match.Success) { [int]$match.Groups[1].Value*60+[int]$match.Groups[2].Value } else { $null }
                    $variants += [pscustomobject]@{
                        mode=$transform.Name; text=$raw
                        matched=($match.Success -and $seconds -le $maximum)
                        seconds=$seconds
                    }
                } finally {
                    $crop.Dispose()
                    if($temporary -and (Test-Path -LiteralPath $path)) { Remove-Item -LiteralPath $path }
                }
            }
            $readings=@($variants | Where-Object matched)
            $groups=@($readings | Group-Object seconds)
            $agreed=($groups.Count -eq 1 -and $readings.Count -ge 2)
            $results += [pscustomobject]@{
                region=$region.Name
                text=($variants.text -join ' | ')
                matched=$agreed
                seconds=$(if($agreed) { $readings[0].seconds } else { $null })
                votes=$readings.Count
                conflicting=($groups.Count -gt 1)
                variants=$variants
            }
        }
    } finally { $source.Dispose() }
    $matches=@($results | Where-Object matched)
    [pscustomobject]@{
        valid=($matches.Count -eq 1)
        hudRegion=$(if($matches.Count -eq 1) { $matches[0].region } else { 'unknown' })
        # Retained for callers using the old field; describes HUD layout only.
        phase=$(if($matches.Count -eq 1) { $matches[0].region+'-hud' } else { 'unknown' })
        clockSeconds=$(if($matches.Count -eq 1) { $matches[0].seconds } else { $null })
        candidates=$results
    }
}
Export-ModuleMember -Function Read-Ufc6RoundClock

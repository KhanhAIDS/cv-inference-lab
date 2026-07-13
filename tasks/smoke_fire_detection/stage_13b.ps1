param(
    [string]$FiglibRoot = "datasets\smoke_fire_detection\FIgLib",
    [string]$ProbeOutputRoot = "artifacts\smoke_fire_detection\staging\figlib_probe",
    [string]$ProbeManifestOut = "artifacts\smoke_fire_detection\figlib_upload_probe_manifest.json",
    [string]$SubsetOutputRoot = "artifacts\smoke_fire_detection\staging\figlib_subset64",
    [string]$SubsetManifestOut = "artifacts\smoke_fire_detection\figlib_subset64_manifest.json",
    [int]$SubsetCount = 64,
    [int]$Seed = 20260707,
    [double]$ProbeGiB = 1.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-CameraId([string]$SequenceId) {
    $parts = @($SequenceId -split "_", 3)
    if ($parts.Count -eq 3) {
        return $parts[2]
    }
    $tokens = $SequenceId.Split("-")
    for ($index = $tokens.Count - 1; $index -ge 0; $index--) {
        if ($tokens[$index] -in @("mobo", "iqeye")) {
            $start = [Math]::Max(0, $index - 2)
            return ($tokens[$start..($tokens.Count - 1)] -join "-")
        }
    }
    return "unknown"
}

function Get-HashKey([string]$Value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    $hash = [Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    return ([BitConverter]::ToString($hash) -replace "-", "").ToLowerInvariant()
}

function Get-SequenceRecords([string]$Root, [int]$SelectionSeed) {
    $records = @()
    foreach ($directory in (Get-ChildItem -LiteralPath $Root -Directory | Sort-Object Name)) {
        $files = @(Get-ChildItem -LiteralPath $directory.FullName -File -Recurse)
        $bytes = ($files | Measure-Object -Property Length -Sum).Sum
        $yearMatch = [Regex]::Match($directory.Name, "^(\d{4})")
        $year = if ($yearMatch.Success) { [int]$yearMatch.Groups[1].Value } else { 0 }
        $records += [PSCustomObject]@{
            SequenceId = $directory.Name
            CameraId = Get-CameraId $directory.Name
            Year = $year
            Bytes = [int64]$bytes
            FileCount = $files.Count
            SortKey = Get-HashKey "$SelectionSeed|$($directory.Name)"
            Source = $directory.FullName
        }
    }
    return $records
}

function Get-TotalBytes($Records) {
    if (-not $Records) {
        return [int64]0
    }
    $measure = @($Records | Measure-Object -Property Bytes -Sum)
    if ($measure.Count -eq 0 -or $null -eq $measure[0].Sum) {
        return [int64]0
    }
    return [int64]$measure[0].Sum
}

function Select-OnePerCameraByYear($Records, [int]$Count, [int]$SelectionSeed) {
    $selected = @()
    $usedCameras = @{}
    $years = @($Records.Year | Sort-Object -Unique)
    while ($selected.Count -lt $Count) {
        $added = $false
        foreach ($year in $years) {
            $candidate = @($Records | Where-Object { $_.Year -eq $year -and -not $usedCameras.ContainsKey($_.CameraId) } | Sort-Object SortKey | Select-Object -First 1)
            if ($candidate.Count -eq 1) {
                $selected += $candidate[0]
                $usedCameras[$candidate[0].CameraId] = $true
                $added = $true
                if ($selected.Count -ge $Count) {
                    break
                }
            }
        }
        if (-not $added) {
            break
        }
    }
    if ($selected.Count -lt $Count) {
        throw "Not enough camera-disjoint sequences for subset: need $Count, got $($selected.Count)"
    }
    return @($selected | Sort-Object SequenceId)
}

function Select-Probe($Records, [int64]$TargetBytes) {
    $selected = @()
    $usedCameras = @{}
    $years = @($Records.Year | Sort-Object -Unique)
    while (((Get-TotalBytes $selected) -lt $TargetBytes) -or $selected.Count -eq 0) {
        $added = $false
        foreach ($year in $years) {
            $candidate = @($Records | Where-Object { $_.Year -eq $year -and -not $usedCameras.ContainsKey($_.CameraId) } | Sort-Object SortKey | Select-Object -First 1)
            if ($candidate.Count -eq 1) {
                $selected += $candidate[0]
                $usedCameras[$candidate[0].CameraId] = $true
                $added = $true
                if (((Get-TotalBytes $selected) -ge $TargetBytes) -and $selected.Count -gt 0) {
                    break
                }
            }
        }
        if (-not $added) {
            break
        }
    }
    if ((Get-TotalBytes $selected) -lt $TargetBytes) {
        throw "Not enough data to reach probe target bytes"
    }
    return @($selected | Sort-Object SequenceId)
}

function Write-Manifest($Path, $Kind, $Records, [int]$SelectionSeed, [int]$DevCount = 0) {
    $totalBytes = Get-TotalBytes $Records
    $totalFiles = [int](($Records | Measure-Object -Property FileCount -Sum).Sum)
    $manifest = [ordered]@{
        dataset = "FIgLib"
        kind = $Kind
        seed = $SelectionSeed
        sequence_count = $Records.Count
        file_count = $totalFiles
        bytes = $totalBytes
        sequence_ids = @($Records.SequenceId | Sort-Object)
        camera_ids = @($Records.CameraId | Sort-Object -Unique)
        years = @($Records.Year | Sort-Object -Unique)
    }
    if ($DevCount -gt 0) {
        $splitRecords = @($Records | Sort-Object SortKey)
        $manifest.dev_sequence_ids = @($splitRecords | Select-Object -First $DevCount | ForEach-Object SequenceId | Sort-Object)
        $manifest.test_sequence_ids = @($splitRecords | Select-Object -Skip $DevCount | ForEach-Object SequenceId | Sort-Object)
    }
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Copy-Selected($Records, [string]$OutputRoot) {
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "Output already exists: $OutputRoot"
    }
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    foreach ($record in $Records) {
        Copy-Item -LiteralPath $record.Source -Destination $OutputRoot -Recurse
    }
}

if (-not (Test-Path -LiteralPath $FiglibRoot -PathType Container)) {
    throw "FIgLib root does not exist: $FiglibRoot"
}
$records = @(Get-SequenceRecords $FiglibRoot $Seed)
if ($records.Count -eq 0) {
    throw "No FIgLib sequences found"
}
$probeTargetBytes = [int64]($ProbeGiB * 1GB)
$probeRecords = @(Select-Probe $records $probeTargetBytes)
$subsetRecords = @(Select-OnePerCameraByYear $records $SubsetCount $Seed)
Copy-Selected $probeRecords $ProbeOutputRoot
Copy-Selected $subsetRecords $SubsetOutputRoot
Write-Manifest $ProbeManifestOut "upload_probe" $probeRecords $Seed
Write-Manifest $SubsetManifestOut "exploratory_subset" $subsetRecords $Seed 44
Write-Output ("Probe: {0} sequence, {1} files, {2:N0} bytes" -f $probeRecords.Count, (($probeRecords | Measure-Object -Property FileCount -Sum).Sum), (Get-TotalBytes $probeRecords))
Write-Output ("Subset: {0} sequence, {1} files, {2:N0} bytes" -f $subsetRecords.Count, (($subsetRecords | Measure-Object -Property FileCount -Sum).Sum), (Get-TotalBytes $subsetRecords))

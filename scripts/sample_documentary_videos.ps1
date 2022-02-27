# $rootfolder = "dokumentarer"
$outdir = "frames-for-training"
mkdir -p $outdir


# 10 first videos are shown here as example
$f = @()
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.01.From.Pole.to.Pole.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.02.Mountains.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.03.Fresh.Water.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.04.Caves.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.05.Deserts.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.06.Ice.Worlds.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.07.Great.Plains.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.08.Jungles.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.09.Shallow.Seas.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.10.Seasonal.Forests.2006.mkv"
$f = $f + "/Volumes/video/2006 Planet Earth 4K/Planet.Earth.11.Ocean.Deep.2006.mkv"


$len = $f.Count

write-host $len

0..$len | foreach-object -ThrottleLimit 8 -parallel {
    $idx = $_
    $vid = ($using:f)[$idx]
    $vidinfo = ffprobe -show_format -loglevel quiet $vid

    # get duration
    $duration = $null
    foreach ($line in $vidinfo) {
        if($line.Contains("duration")) {
            $duration = $line.split("duration=")[1]
            $duration = [Math]::Floor($duration) 
        }
    }
    if($null -eq $duration) {
        Write-Host no duration found for $vid
        continue
    }
    if($duration -lt 100) {
        Write-Host $duration too short for $vid
        continue
    }

    Write-Host "$idx".PadLeft(6,'0') : $duration : $vid

    $savedir = "$idx".PadLeft(6,'0')
    mkdir -p $using:outdir/$savedir > $null

    $counter = 0
    $frame_pos = 100
    

    # save frames
    while (($duration -gt ($frame_pos + 100)) -and ($counter -lt 500)) {
        $sub_savedir = "$counter".PadLeft(6,'0')
        mkdir -p $using:outdir/$savedir/$sub_savedir > $null
        ffmpeg -loglevel quiet -accurate_seek -ss $frame_pos -i $vid -frames:v 9 -q:v 2 -vf scale=1024:1024 -y "$using:outdir/$savedir/$sub_savedir/%02d.jpg"

        $frame_pos += 
        $counter++
        write-host $frame_pos $counter "$using:outdir/$savedir/$sub_savedir"
    }


}


param([string]$N="xl",[int]$I=50,[int]$G=500,[int]$S=800,[int]$E=5,[string]$L="training.log",[switch]$T,[switch]$W)
$root=Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$logPath=Join-Path $root $L
if($W){if(Test-Path $logPath){Get-Content $logPath -Wait}else{Write-Host "No log $logPath"};exit}
$existing=Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -match "train_alphazero"}
if($existing){Write-Host "Already running!" -ForegroundColor Yellow; $existing | Format-Table Id; if((Read-Host "Continue? y/N") -ne "y"){exit}}
$cmdLine = "set PYTHONIOENCODING=utf-8 && uv run python -m penguinchess.ai.train_alphazero --network $N --iterations $I --games $G --simulations $S --eval-interval $E --auto-eval"
$cmdLine = $cmdLine -replace '"','\"'
$fullCmd = "start /B cmd /c `"$cmdLine > `"`"$logPath`"`" 2>&1`""
Start-Process -FilePath cmd -ArgumentList "/c",$fullCmd -WindowStyle Hidden -WorkingDirectory $root
Start-Sleep 1
Write-Host "Training started" -ForegroundColor Green
Write-Host "  xl $I iters" -ForegroundColor Cyan
Write-Host "  log: $logPath" -ForegroundColor Cyan
Write-Host "  .\scripts\train_bg.ps1 -Watch  -- follow log" -ForegroundColor Yellow
if($T){Start-Sleep 3;Get-Content $logPath -Wait}

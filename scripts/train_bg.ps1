param([string]$N="xl",[int]$I=50,[int]$G=500,[int]$S=800,[int]$E=5,[string]$L="training.log",[switch]$T,[switch]$W)
$root=Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$logPath=Join-Path $root $L
if($W){if(Test-Path $logPath){Get-Content $logPath -Wait}else{Write-Host "No log $logPath"};exit}
$existing=Get-Process -Name "python" -ErrorAction SilentlyContinue|Where-Object{$_.CommandLine -match"train_alphazero"}
if($existing){Write-Host "Already running!" -ForegroundColor Yellow;$existing|Format-Table Id;if((Read-Host"Continue?")-ne"y"){exit}}
$null = Start-Job -Name AZTrain -ScriptBlock {
    $env:PYTHONUNBUFFERED=1; $env:PYTHONIOENCODING="utf-8"
    Set-Location $args[0]
    uv run python -m penguinchess.ai.train_alphazero --network $args[1] --iterations $args[2] --games $args[3] --simulations $args[4] --eval-interval $args[5] --auto-eval *> $args[6]
} -ArgumentList $root, $N, $I, $G, $S, $E, $logPath
Start-Sleep 2
Write-Host "Training started (Job: AZTrain)" -ForegroundColor Green
Write-Host "  xl $I iters  log: $logPath" -ForegroundColor Cyan
Write-Host "  .\scripts\train_bg.ps1 -Watch  -- follow log" -ForegroundColor Yellow
Write-Host "  Get-Job AZTrain  -- status" -ForegroundColor Yellow
if($T){Start-Sleep 3;Get-Content $logPath -Wait}

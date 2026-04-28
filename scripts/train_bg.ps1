# train_bg.ps1: Start background AlphaZero training with log tracking
# Usage:
#   .\scripts\train_bg.ps1                           # default XL 50 iters
#   .\scripts\train_bg.ps1 -Network xl -Iter 20
#   .\scripts\train_bg.ps1 -Tail                      # start + follow log
#   .\scripts\train_bg.ps1 -Watch                     # only follow log

param(
    [string]$Network = "xl",
    [int]$Iterations = 50,
    [int]$Games = 500,
    [int]$Simulations = 800,
    [int]$EvalInterval = 5,
    [string]$LogFile = "training.log",
    [switch]$Tail,
    [switch]$Watch
)

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$logPath = Join-Path $root $LogFile

if ($Watch) {
    if (Test-Path $logPath) { Get-Content $logPath -Wait }
    else { Write-Host "No log: $logPath" -ForegroundColor Red }
    exit
}

# Check existing training process
$existing = Get-Process -Name "python" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "train_alphazero" }
if ($existing) {
    Write-Host "Training already running!" -ForegroundColor Yellow
    $existing | Format-Table Id, StartTime
    $choice = Read-Host "Continue? (y/N)"
    if ($choice -ne "y") { exit }
}

$cmd = "`$env:PYTHONIOENCODING='utf-8'; & uv run python -m penguinchess.ai.train_alphazero --network $Network --iterations $Iterations --games $Games --simulations $Simulations --eval-interval $EvalInterval --auto-eval"

$startArgs = @{
    FilePath = "powershell"
    ArgumentList = "-Command", $cmd
    NoNewWindow = $false
    WorkingDirectory = $root
    RedirectStandardOutput = $logPath
    RedirectStandardError = $logPath
}

$proc = Start-Process @startArgs -PassThru
Write-Host "Training started (PID: $($proc.Id))" -ForegroundColor Green
Write-Host "  Network: $Network  Iter: $Iterations" -ForegroundColor Cyan
Write-Host "  Log: $logPath" -ForegroundColor Cyan
Write-Host "  .\scripts\train_bg.ps1 -Watch   -- follow log" -ForegroundColor Yellow

if ($Tail) { Start-Sleep 2; Get-Content $logPath -Wait }

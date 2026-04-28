# train_bg.ps1 — 后台 AlphaZero 训练启动器
# 用法:
#   .\scripts\train_bg.ps1                         # 默认 XL, 50 迭代
#   .\scripts\train_bg.ps1 -Network large -Iter 20  # Large 网络, 20 迭代
#   .\scripts\train_bg.ps1 -Tail                     # 启动后自动追踪日志
#   .\scripts\train_bg.ps1 -Watch                    # 只查看当前进度（不启动新训练）

param(
    [string]$Network = "xl",        # xl / large / resnet
    [int]$Iterations = 50,
    [int]$Games = 500,
    [int]$Simulations = 800,
    [int]$EvalInterval = 5,
    [string]$LogFile = "training.log",
    [switch]$Tail,                   # 启动后自动 tail
    [switch]$Watch                   # 只查看日志，不启动训练
)

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$logPath = Join-Path $root $LogFile

if ($Watch) {
    if (Test-Path $logPath) {
        Get-Content $logPath -Wait
    } else {
        Write-Host "日志文件不存在: $logPath" -ForegroundColor Red
    }
    exit
}

# 检查是否已有训练在运行
$existing = Get-Process -Name "python" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "train_alphazero" }
if ($existing) {
    Write-Host "警告: 已有训练进程在运行！" -ForegroundColor Yellow
    $existing | Format-Table Id, StartTime
    $choice = Read-Host "继续? (y/N)"
    if ($choice -ne "y") { exit }
}

# 构建命令
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
Write-Host "训练已后台启动 (PID: $($proc.Id))" -ForegroundColor Green
Write-Host "  网络:    $Network" -ForegroundColor Cyan
Write-Host "  迭代:    $Iterations" -ForegroundColor Cyan
Write-Host "  日志:    $logPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Get-Content $logPath -Wait     # 实时追踪" -ForegroundColor Yellow
Write-Host "  .\scripts\train_bg.ps1 -Watch   # 同样效果" -ForegroundColor Yellow
Write-Host "  Get-Content $logPath -Tail 5    # 只看最后几行" -ForegroundColor Yellow

if ($Tail) {
    Start-Sleep 2
    Get-Content $logPath -Wait
}

# Script de deploy para Windows PowerShell
# Deploy do backend para o servidor remoto

$SERVER = "root@31.97.241.157"
$REMOTE_PATH = "~/python-freela/"
$DIST_DIR = ".\dist"

Write-Host "[*] Iniciando deploy para $SERVER" -ForegroundColor Green
Write-Host "[*] Diretorio remoto: $REMOTE_PATH" -ForegroundColor Cyan

# Criar pasta dist temporária se não existir
if (Test-Path $DIST_DIR) {
    Write-Host "`n[*] Limpando pasta dist existente..." -ForegroundColor Yellow
    Remove-Item -Path $DIST_DIR -Recurse -Force
}

Write-Host "`n[*] Criando pasta dist com arquivos necessarios..." -ForegroundColor Cyan
New-Item -Path $DIST_DIR -ItemType Directory -Force | Out-Null

# Copiar arquivos Python (exceto deploy.ps1 e arquivos de cache)
Get-ChildItem -Path "." -Filter "*.py" -File | Where-Object {
    $_.Name -ne "deploy.ps1" -and
    $_.FullName -notmatch "__pycache__" -and
    $_.FullName -notmatch "\.git"
} | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $DIST_DIR -Force
    Write-Host "   [+] $($_.Name)" -ForegroundColor Gray
}

# Copiar requirements.txt
if (Test-Path "requirements.txt") {
    Copy-Item -Path "requirements.txt" -Destination $DIST_DIR -Force
    Write-Host "   [+] requirements.txt" -ForegroundColor Gray
}

# Copiar start_backend.sh se existir
if (Test-Path "start_backend.sh") {
    Copy-Item -Path "start_backend.sh" -Destination $DIST_DIR -Force
    Write-Host "   [+] start_backend.sh" -ForegroundColor Gray
}

# Copiar start_server.py se existir (já deve estar nos .py, mas garantindo)
if (Test-Path "start_server.py") {
    Copy-Item -Path "start_server.py" -Destination $DIST_DIR -Force
    Write-Host "   [+] start_server.py" -ForegroundColor Gray
}

Write-Host "`n[*] Copiando arquivos para o servidor..." -ForegroundColor Cyan
# Usar scp para copiar tudo de uma vez (equivalente ao comando: scp -r .\dist\* root@...)
# No PowerShell, precisamos expandir o * antes de passar para o scp
$distFullPath = (Resolve-Path $DIST_DIR -ErrorAction SilentlyContinue).Path

if (-not $distFullPath) {
    Write-Host "   [ERRO] Pasta dist nao encontrada" -ForegroundColor Red
    exit 1
}

# Expandir todos os itens na pasta dist e copiar (equivale a: scp -r .\dist\*)
Write-Host "   Enviando todos os arquivos..." -ForegroundColor Gray
Get-ChildItem -Path $distFullPath | ForEach-Object {
    $itemName = $_.Name
    if ($_.PSIsContainer) {
        # É uma pasta - usar scp -r
        Write-Host "   Copiando pasta: $itemName" -ForegroundColor Gray
        scp -r "$($_.FullName)" "${SERVER}:${REMOTE_PATH}$itemName"
    } else {
        # É um arquivo
        Write-Host "   Copiando arquivo: $itemName" -ForegroundColor Gray
        scp "$($_.FullName)" "${SERVER}:${REMOTE_PATH}$itemName"
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   [ERRO] Erro ao copiar $itemName" -ForegroundColor Red
        # Limpar pasta dist antes de sair
        Remove-Item -Path $DIST_DIR -Recurse -Force -ErrorAction SilentlyContinue
        exit 1
    }
}

# Limpar pasta dist temporária
Write-Host "`n[*] Limpando pasta dist temporaria..." -ForegroundColor Yellow
Remove-Item -Path $DIST_DIR -Recurse -Force

Write-Host "`n[OK] Deploy concluido com sucesso!" -ForegroundColor Green
Write-Host "`n[*] Proximos passos no servidor:" -ForegroundColor Yellow
Write-Host "   1. ssh $SERVER" -ForegroundColor White
Write-Host "   2. cd $REMOTE_PATH" -ForegroundColor White
Write-Host "   3. pip install -r requirements.txt" -ForegroundColor White
Write-Host "   4. python start_server.py" -ForegroundColor White


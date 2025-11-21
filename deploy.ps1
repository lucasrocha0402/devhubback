# Script de deploy para Windows PowerShell
# Deploy apenas do backend para o servidor remoto

$SERVER = "root@31.97.241.157"
$REMOTE_PATH_BACKEND = "/root/python-freela/"
$DIST_DIR = ".\dist"

Write-Host "[*] Iniciando deploy para $SERVER" -ForegroundColor Green
Write-Host "[*] Backend remoto: $REMOTE_PATH_BACKEND" -ForegroundColor Cyan

# Criar pasta dist temporária se não existir
if (Test-Path $DIST_DIR) {
    Write-Host "`n[*] Limpando pasta dist existente..." -ForegroundColor Yellow
    Remove-Item -Path $DIST_DIR -Recurse -Force
}

Write-Host "`n[*] Criando pasta dist com arquivos necessarios..." -ForegroundColor Cyan
New-Item -Path $DIST_DIR -ItemType Directory -Force | Out-Null

# Copiar arquivos Python (exceto arquivos de cache)
Get-ChildItem -Path "." -Filter "*.py" -File | Where-Object {
    $_.FullName -notmatch "__pycache__" -and
    $_.FullName -notmatch "\.git"
} | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $DIST_DIR -Force
    Write-Host "   [+] $($_.Name)" -ForegroundColor Gray
}

# Copiar arquivos extras importantes
$extraFiles = @("requirements.txt", ".env")
foreach ($file in $extraFiles) {
    if (Test-Path $file) {
        Copy-Item -Path $file -Destination $DIST_DIR -Force
        Write-Host "   [+] $file" -ForegroundColor Gray
    }
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
if (-not (Test-Path $DIST_DIR)) {
    Write-Host "   [ERRO] Pasta dist nao encontrada" -ForegroundColor Red
    exit 1
}

$distFullPath = (Get-Item $DIST_DIR).FullName

# Expandir todos os itens na pasta dist e copiar (equivale a: scp -r .\dist\*)
Write-Host "   Enviando todos os arquivos..." -ForegroundColor Gray
Get-ChildItem -Path $distFullPath | ForEach-Object {
    $itemName = $_.Name
    if (Test-Path -Path $_.FullName -PathType Container) {
        # É uma pasta - usar scp -r
        Write-Host "   Copiando pasta: $itemName" -ForegroundColor Gray
        scp -r "$($_.FullName)" "${SERVER}:${REMOTE_PATH_BACKEND}$itemName"
    } else {
        # É um arquivo
        Write-Host "   Copiando arquivo: $itemName" -ForegroundColor Gray
        scp "$($_.FullName)" "${SERVER}:${REMOTE_PATH_BACKEND}$itemName"
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

Write-Host "`n[OK] Deploy completo concluido com sucesso!" -ForegroundColor Green
Write-Host "`n[*] Proximos passos no servidor:" -ForegroundColor Yellow
Write-Host "   BACKEND:" -ForegroundColor Cyan
Write-Host "   1. ssh $SERVER" -ForegroundColor White
Write-Host "   2. cd $REMOTE_PATH_BACKEND" -ForegroundColor White
Write-Host "   3. pip install -r requirements.txt" -ForegroundColor White
Write-Host "   4. python start_server.py" -ForegroundColor White


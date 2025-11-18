# Script de deploy para Windows PowerShell
# Deploy do backend e frontend para o servidor remoto

$SERVER = "root@31.97.241.157"
$REMOTE_PATH_BACKEND = "/root/python-freela/"
$REMOTE_PATH_FRONTEND = "/root/devhub-frontend/dist"
$FRONTEND_PATH = "C:\Users\lukas\OneDrive\Área de Trabalho\desenvolvimento_lucas\devhubfront\devhubfront"
$DIST_DIR = ".\dist"

Write-Host "[*] Iniciando deploy para $SERVER" -ForegroundColor Green
Write-Host "[*] Backend remoto: $REMOTE_PATH_BACKEND" -ForegroundColor Cyan
Write-Host "[*] Frontend remoto: $REMOTE_PATH_FRONTEND" -ForegroundColor Cyan

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

# ============================================
# DEPLOY DO FRONTEND
# ============================================
Write-Host "`n[*] ========================================" -ForegroundColor Cyan
Write-Host "[*] Iniciando deploy do FRONTEND" -ForegroundColor Cyan
Write-Host "[*] ========================================" -ForegroundColor Cyan

if (-not (Test-Path $FRONTEND_PATH)) {
    Write-Host "   [ERRO] Caminho do frontend nao encontrado: $FRONTEND_PATH" -ForegroundColor Red
    Write-Host "   [AVISO] Continuando apenas com deploy do backend..." -ForegroundColor Yellow
} else {
    Write-Host "`n[*] Verificando build do frontend..." -ForegroundColor Cyan
    
    # Verificar se existe pasta dist no frontend
    $frontendDistPath = Join-Path $FRONTEND_PATH "dist"
    if (-not (Test-Path $frontendDistPath)) {
        Write-Host "   [*] Pasta dist nao encontrada. Executando build..." -ForegroundColor Yellow
        Push-Location $FRONTEND_PATH
        npm run build
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   [ERRO] Erro ao fazer build do frontend!" -ForegroundColor Red
            Pop-Location
            Write-Host "   [AVISO] Continuando apenas com deploy do backend..." -ForegroundColor Yellow
        } else {
            Pop-Location
            Write-Host "   [+] Build do frontend concluido!" -ForegroundColor Green
        }
    } else {
        Write-Host "   [+] Pasta dist encontrada!" -ForegroundColor Green
    }
    
    # Verificar novamente se dist existe após build
    if (Test-Path $frontendDistPath) {
        Write-Host "`n[*] Copiando arquivos do frontend para o servidor..." -ForegroundColor Cyan
        
        # Criar diretório remoto se não existir
        Write-Host "   [*] Criando diretorio remoto (se necessario)..." -ForegroundColor Gray
        ssh $SERVER "mkdir -p $REMOTE_PATH_FRONTEND" 2>&1 | Out-Null
        
        # Copiar todos os arquivos da pasta dist recursivamente
        Write-Host "   [*] Enviando arquivos do frontend..." -ForegroundColor Gray
        $frontendDistFullPath = (Get-Item $frontendDistPath).FullName
        
        # Obter todos os arquivos e pastas recursivamente
        $allItems = Get-ChildItem -Path $frontendDistFullPath -Recurse
        
        $fileCount = 0
        $errorCount = 0
        
        foreach ($item in $allItems) {
            # Calcular caminho relativo
            $relativePath = $item.FullName.Substring($frontendDistFullPath.Length + 1)
            $remotePath = "$REMOTE_PATH_FRONTEND/$relativePath"
            
            if ($item.PSIsContainer) {
                # É uma pasta - criar no servidor
                ssh $SERVER "mkdir -p `"$remotePath`"" 2>&1 | Out-Null
            } else {
                # É um arquivo - copiar
                $remoteDir = $remotePath.Substring(0, $remotePath.LastIndexOf('/'))
                ssh $SERVER "mkdir -p `"$remoteDir`"" 2>&1 | Out-Null
                
                scp "$($item.FullName)" "${SERVER}:${remotePath}"
                
                if ($LASTEXITCODE -eq 0) {
                    $fileCount++
                    if ($fileCount % 10 -eq 0) {
                        Write-Host "   [+] $fileCount arquivos copiados..." -ForegroundColor Gray
                    }
                } else {
                    $errorCount++
                    Write-Host "   [ERRO] Erro ao copiar: $relativePath" -ForegroundColor Red
                }
            }
        }
        
        if ($errorCount -eq 0) {
            Write-Host "   [+] $fileCount arquivos copiados com sucesso!" -ForegroundColor Green
            Write-Host "   [+] Frontend deployado com sucesso!" -ForegroundColor Green
        } else {
            Write-Host "   [!] $fileCount arquivos copiados, $errorCount erros!" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n[OK] Deploy completo concluido com sucesso!" -ForegroundColor Green
Write-Host "`n[*] Proximos passos no servidor:" -ForegroundColor Yellow
Write-Host "   BACKEND:" -ForegroundColor Cyan
Write-Host "   1. ssh $SERVER" -ForegroundColor White
Write-Host "   2. cd $REMOTE_PATH_BACKEND" -ForegroundColor White
Write-Host "   3. pip install -r requirements.txt" -ForegroundColor White
Write-Host "   4. python start_server.py" -ForegroundColor White
Write-Host "`n   FRONTEND:" -ForegroundColor Cyan
Write-Host "   Frontend deployado em: $REMOTE_PATH_FRONTEND" -ForegroundColor White


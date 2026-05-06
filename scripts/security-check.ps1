$ErrorActionPreference = 'Stop'
# Block destructive commands — customize this blocklist for your repo
$blockedPatterns = @('rm -rf /', 'DROP DATABASE', 'format C:', 'mkfs')
$input = $args -join ' '
foreach ($pattern in $blockedPatterns) {
    if ($input -match [regex]::Escape($pattern)) {
        Write-Error "Blocked: destructive pattern detected"
        exit 1
    }
}

# Start the API (port 8000) and the web dev server (port 5173), then open the browser.
$root = Split-Path -Parent $PSScriptRoot
Start-Process -WorkingDirectory $root -FilePath "python" -ArgumentList "-m", "uvicorn", "api.server:app", "--port", "8000"
if (-not (Test-Path "$root\web\node_modules")) { npm --prefix "$root\web" install }
Start-Process -WorkingDirectory "$root\web" -FilePath "npm.cmd" -ArgumentList "run", "dev"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"

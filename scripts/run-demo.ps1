# Launch demo with proxy bypass (Windows-friendly)
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
Set-Location $PSScriptRoot\..
& .\.venv\Scripts\python.exe -m vla_mini.demo @args

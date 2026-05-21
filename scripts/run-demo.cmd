@echo off
cd /d %~dp0..
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
.\.venv\Scripts\python.exe -m vla_mini.demo %*

@echo off
setlocal

echo Starting reverse proxy tunnel to 189...
echo Keep this window open while 189 needs to use your local ikuuu proxy.
echo.

ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R 17890:127.0.0.1:7890 189

echo.
echo Tunnel closed.
pause

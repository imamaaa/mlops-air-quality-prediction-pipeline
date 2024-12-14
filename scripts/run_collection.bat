@echo off
echo [%DATE% %TIME%] Starting weather collection script >> run_log.txt

cd /d %~dp0\..
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Failed to change directory >> run_log.txt
    exit /b %ERRORLEVEL%
)

call .venv\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Failed to activate virtual environment >> run_log.txt
    exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Starting data collection... >> run_log.txt
python scripts\fetch_weather_data.py
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Python script failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Running DVC operations... >> run_log.txt
dvc add data/weather
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Failed to add weather data to DVC >> run_log.txt
    exit /b %ERRORLEVEL%
)

dvc add data/pollution
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Failed to add pollution data to DVC >> run_log.txt
    exit /b %ERRORLEVEL%
)

dvc commit -f
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: DVC commit failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

dvc push
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: DVC push failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

echo [%DATE% %TIME%] Running Git operations... >> run_log.txt
git add data.dvc
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Git add failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

git commit -m "Update weather and pollution data %DATE% %TIME%"
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Git commit failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

git push
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Git push failed >> run_log.txt
    exit /b %ERRORLEVEL%
)

call deactivate
echo [%DATE% %TIME%] Script completed successfully >> run_log.txt
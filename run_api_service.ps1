# Запускает api.py постоянным фоновым сервисом. Используется задачей
# Планировщика заданий Windows "rag-transcripts-api" (автозапуск при входе
# в систему). Лог пишется рядом, в run_api_service.log.
#
# Важно: НЕ ставить $ErrorActionPreference = "Stop" здесь — uvicorn пишет
# свои INFO-логи в stderr, а PowerShell оборачивает построчный вывод
# нативных программ из stderr в ErrorRecord; при "Stop" это убивает
# скрипт сразу после первой же строчки лога.
Set-Location $PSScriptRoot
& ".\venv\Scripts\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000 *>> ".\run_api_service.log"

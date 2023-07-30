@echo start program
start /wait /B /HIGH ./bin/out.exe %*
@echo Return code: %errorlevel%
@echo off


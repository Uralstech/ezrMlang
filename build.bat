@echo off
title ezrM builder

cls
pyinstaller ezrMShell.spec --distpath ./Builds --workpath ./Temp --clean
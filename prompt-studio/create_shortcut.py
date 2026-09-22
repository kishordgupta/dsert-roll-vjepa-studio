"""Create a Windows desktop launcher from a WSL checkout."""
from pathlib import Path
import os,subprocess
root=Path(__file__).resolve().parent
distro=os.environ.get("WSL_DISTRO_NAME","Ubuntu-24.04")
profile=subprocess.check_output(["cmd.exe","/c","echo","%USERPROFILE%"],text=True).strip()
desktop=Path(subprocess.check_output(["wslpath",profile],text=True).strip())/"Desktop"
cmd="@echo off\r\nwsl.exe -d "+distro+" -- bash \""+str(root/"start.sh")+"\"\r\nif errorlevel 1 goto failed\r\nstart \"\" \"http://127.0.0.1:8765\"\r\nexit /b 0\r\n:failed\r\necho Studio failed. See server.log.\r\npause\r\n"
(desktop/"DSERT Prompt Studio.cmd").write_text(cmd)
print("Desktop launcher created")

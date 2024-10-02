import subprocess

print(subprocess.run(["sh 1_session-install-dependencies/download_requirements.sh"], shell=True))
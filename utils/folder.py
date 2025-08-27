import os

def SetupExpDir(basedir: str):
	os.makedirs(basedir, exist_ok=True)
	expIdx = 1
	while os.path.exists(os.path.join(basedir, f"exp-{expIdx}")):
		expIdx = expIdx + 1
	expDir = os.path.join(basedir, f"exp-{expIdx}")
	os.makedirs(expDir, exist_ok=True)
	return os.path.abspath(expDir)
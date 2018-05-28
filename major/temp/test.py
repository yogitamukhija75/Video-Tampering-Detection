import os
import subprocess
print os.getcwd()
import shutil


path =  os.path.join(os.path.dirname(os.path.realpath(__file__)),'frames') 
shutil.rmtree(path)
os.mkdir(path)
# subprocess.check_call(['rm -rf'], cwd =os.path.join(os.path.dirname(os.path.realpath(__file__)),'frames'))

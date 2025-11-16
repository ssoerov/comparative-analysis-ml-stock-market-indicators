import subprocess
with open('requirements-lock.txt', 'w', encoding='utf-8') as f:
    out = subprocess.check_output(['python', '-m', 'pip', 'freeze']).decode('utf-8')
    f.write(out)
print('requirements-lock.txt создан.')


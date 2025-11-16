import pkgutil
print('TF present' if pkgutil.find_loader('tensorflow') else 'TF missing')


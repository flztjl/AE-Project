import os

for fno in os.listdir('/media/lj/Data/Python/HTL/2nd day/'):
    path = os.path.join('/media/lj/Data/Python/HTL/2nd day/', fno)
    for fn in os.listdir(path):
        fn_path = os.path.join(path, fn)
        fn_new = fn.replace(' ', '-')
        os.rename(fn_path, os.path.join(path, fn_new))

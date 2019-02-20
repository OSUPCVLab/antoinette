import os

def delFile(dataset_dir):

    print('Reading training files...')

    for path, subdirs, files in os.walk(dataset_dir):
        for name in subdirs:
            if 'depth' in name or 'flow' in name or 'flow_vis' in name or 'invalid' in name :

                os.system('rm -rf {0}'.format(os.path.join(path, name)))

dic = { "office1\\","office2\\","office3\\" }

for item in dic:
    delFile(item)

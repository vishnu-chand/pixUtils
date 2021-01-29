'''
production checklist
0. test code and check gpu usage/ test with auto restart
1. delete getDevUtils from common utils
2. common utils self.debug = False
2. enable mail
3. remove unwanted things rm -rf /tmp/virtualBG/   rm everything in cd;ls
conda clean --all; sudo yum autoremove; sudo yum autoclean; sudo yum clean expire-cache
5. deploy with nohoup
'''

from pixUtils.pyUtils import *


def zipVideos():
    des = dirop(f'des_{getTimeStamp()}', rm=True)
    for src in glob('src/*.*'):
        os.system(f'ffmpeg -i {src} -vcodec libx264 -crf 28 {dirop(des, basename(src))}')

zipVideos()


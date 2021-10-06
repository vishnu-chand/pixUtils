import os
from glob import glob
for pdf in glob('src/*.pdf'):
    os.system(f'./k2pdfopt {pdf} -o des/{os.path.basename(pdf)} -ws 0.200 -x')

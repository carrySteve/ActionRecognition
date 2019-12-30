# following code comes from MSR

import os
import numpy as np
import base64
import cv2
import progressbar
import yaml


class FileProgressingbar:
    fileobj = None
    pbar = None

    def __init__(self, fileobj, msg):
        fileobj.seek(0, os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0, os.SEEK_SET)
        self.fileobj = fileobj
        widgets = [
            msg,
            progressbar.AnimatedMarker(), ' ',
            progressbar.Percentage(), ' ',
            progressbar.Bar(), ' ',
            progressbar.ETA()
        ]
        self.pbar = progressbar.ProgressBar(widgets=widgets,
                                            maxval=flen).start()

    def update(self):
        # self.fileobj.seek(0)
        self.pbar.update(self.fileobj.tell())


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein, 'r') as tsvin, open(idxout, 'w') as tsvout:
        bar = FileProgressingbar(tsvin,
                                 'Generating lineidx {0}: '.format(idxout))
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
            bar.update()
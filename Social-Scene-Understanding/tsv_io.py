# following code comes from MSR

import os.path as op
from io_common import generate_lineidx, FileProgressingbar


class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None

        self._ensure_lineidx_loaded()

    def num_rows(self):
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            #print ('yes')
            generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                bar = FileProgressingbar(
                    fp, "Loading lineidx {0}: ".format(self.lineidx))
                self._lineidx = []
                for i in fp:
                    self._lineidx.append(int(i.strip()))
                    bar.update()
                print

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
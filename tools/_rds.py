"""Minimal reader for R's XDR serialization (format 2/3), enough for named
character-vector lists with attributes."""
import bz2
import gzip
import json
import lzma
import struct
import sys

NILVALUE_SXP=254; NILSXP=0; SYMSXP=1; LISTSXP=2; CHARSXP=9; LGLSXP=10
INTSXP=13; REALSXP=14; STRSXP=16; VECSXP=19; REFSXP=255

class R:
    def __init__(self, b):
        self.b=b; self.i=0; self.refs=[]
    def int(self):
        v=struct.unpack('>i', self.b[self.i:self.i+4])[0]; self.i+=4; return v
    def dbl(self):
        v=struct.unpack('>d', self.b[self.i:self.i+8])[0]; self.i+=8; return v
    def raw(self,n):
        v=self.b[self.i:self.i+n]; self.i+=n; return v

    def flags(self):
        f=self.int()
        return f & 0xFF, (f>>8)&0xFF != 0 if False else None, f

    def item(self):
        f=self.int()
        typ = f & 0xFF
        has_attr = bool((f>>9) & 1)
        has_tag  = bool((f>>10) & 1)
        if typ == NILVALUE_SXP or typ == NILSXP:
            return None
        if typ == REFSXP:
            idx = (f >> 8)
            if idx == 0: idx = self.int()
            return self.refs[idx-1]
        if typ == SYMSXP:
            v = self.item()          # the CHARSXP
            self.refs.append(v)
            return v
        if typ == CHARSXP:
            n = self.int()
            if n == -1: return None
            return self.raw(n).decode('utf-8', 'replace')
        if typ in (LGLSXP, INTSXP):
            n=self.int(); vals=[self.int() for _ in range(n)]
            return self.attach(vals, has_attr)
        if typ == REALSXP:
            n=self.int(); vals=[self.dbl() for _ in range(n)]
            return self.attach(vals, has_attr)
        if typ == STRSXP:
            n=self.int(); vals=[self.item() for _ in range(n)]
            return self.attach(vals, has_attr)
        if typ == VECSXP:
            n=self.int(); vals=[self.item() for _ in range(n)]
            return self.attach(vals, has_attr)
        if typ == LISTSXP:
            # pairlist (used for attributes)
            out={}
            tag = self.item() if has_tag else None
            val = self.item()
            out[tag]=val
            rest = self.item()
            if isinstance(rest, dict): out.update(rest)
            return out
        raise ValueError(f"unhandled SEXP type {typ} at {self.i}")

    def attach(self, vals, has_attr):
        attrs = self.item() if has_attr else None
        return {'values': vals, 'attrs': attrs} if attrs else vals


def load_rda(path):
    raw=open(path,'rb').read()
    if raw[:3]==b'BZh': raw=bz2.decompress(raw)
    elif raw[:2]==b'\x1f\x8b': raw=gzip.decompress(raw)
    elif raw[:5]==b'\xfd7zXZ': raw=lzma.decompress(raw)
    assert raw[:5]==b'RDX2\n' or raw[:5]==b'RDX3\n', raw[:8]
    p=raw.index(b'\n', 5)  # 'X\n'
    body=raw[p+1:]
    r=R(body)
    r.int()  # version
    r.int(); r.int()  # writer / min reader
    if raw[:5]==b'RDX3\n':
        n=r.int(); r.raw(n)  # native encoding
    return r.item()

if __name__=='__main__':
    obj=load_rda(sys.argv[1])
    print(json.dumps(obj, indent=1)[:3000])

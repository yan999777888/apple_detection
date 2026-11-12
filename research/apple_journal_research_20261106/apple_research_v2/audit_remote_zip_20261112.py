"""Bounded HTTP-range ZIP inventory; never extracts or executes remote files."""
import argparse
import collections
import io
import json
from pathlib import Path
import urllib.request
import zipfile


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, limit=16*1024*1024):
        self.url, self.size, self.limit = url, size, limit
        self.pos = self.transferred = 0
        self.requests = []

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        if self.pos < 0: raise ValueError('Negative offset')
        return self.pos

    def read(self, size=-1):
        amount = self.size-self.pos if size < 0 else min(size,self.size-self.pos)
        if amount <= 0: return b''
        if self.transferred+amount > self.limit: raise ValueError('Metadata byte budget exceeded')
        end = self.pos+amount-1
        request = urllib.request.Request(self.url,headers={'Range':f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(request,timeout=40) as response:
            expected=f'bytes {self.pos}-{end}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not honor exact bounded range')
            content=response.read(amount+1)
            if len(content)!=amount: raise ValueError('Range length mismatch')
        self.requests.append({'start':self.pos,'end':end,'bytes':amount})
        self.pos+=amount; self.transferred+=amount
        return content


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if output.drive.lower()!='f:' or output.exists(): raise ValueError('Fresh F output required')
    meta=json.loads(args.metadata.read_text(encoding='utf8'))[0]
    remote=RemoteZip(meta['content_details']['download_url'],meta['size'])
    try:
        with zipfile.ZipFile(remote) as archive:
            rows=[{'path':x.filename,'size':x.file_size,'compressed_size':x.compress_size,'crc':x.CRC}
                  for x in archive.infolist() if not x.is_dir()]
    except Exception as error:
        chain=[]
        current=error
        while current is not None:
            chain.append(repr(current)); current=current.__cause__ or current.__context__
        failure=output.with_suffix('.failed.json')
        output.parent.mkdir(parents=True,exist_ok=True)
        if not failure.exists():
            failure.write_text(json.dumps({'status':'inventory_not_obtained','errors':chain,
                'bytes_transferred':remote.transferred,'requests':remote.requests,
                'note':'HTTP access failure is not evidence of archive corruption.'},indent=2),encoding='utf8')
        raise
    apples=[x for x in rows if 'apple' in x['path'].lower()]
    result={'scope':'Central directory only; no image pixels or annotation semantics verified',
            'url':remote.url,'archive_bytes':meta['size'],'bytes_transferred':remote.transferred,
            'requests':remote.requests,'files':len(rows),'apple_path_files':len(apples),
            'apple_folders':dict(collections.Counter(str(Path(x['path']).parent) for x in apples)),
            'rows':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','requests','url')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()

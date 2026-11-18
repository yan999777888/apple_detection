"""Select only original apple JPEGs from the authors' large ZIP on Mendeley Data."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
from urllib.parse import urlsplit
import zipfile
import zlib

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "afruitdb_apples_v1"
SOURCE = {
    "size": 3147056662,
    "sha256_hash": "df2e6fb4fd14a962be2814e445d9c015aa2edf65d84562f40a227bc92c990925",
    "download_url": "https://data.mendeley.com/public-files/datasets/bz65dz2pbj/files/"
                    "9937ac08-0a7e-4cac-b7ea-b79b1033f952/file_downloaded",
}
SIZE = SOURCE["size"]
URL = SOURCE["download_url"]
ACCEPTED = {".jpg", ".jpeg", ".png"}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fetch(start, end):
    """Demand an exact range, validating the response before it enters the archive."""
    assert 0 <= start <= end < SIZE
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    last = ""
    for attempt in range(9):
        try:
            url = f"{URL}?source=archive&part={start}"
            with requests.get(url, headers=headers, stream=True, timeout=(12,35)) as r:
                if r.status_code != 206:
                    raise ValueError(f"HTTP {r.status_code} from {urlsplit(r.url).hostname}")
                if r.headers.get("Content-Range","") != f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("inexact response range " + str(r.headers.get("Content-Range")))
                data = b"".join(r.iter_content(1024*512))
                if len(data) != end-start+1:
                    raise ValueError("Incomplete ZIP span")
                return data
        except (requests.RequestException, ValueError) as exc:
            last = str(exc).split(" (Caused by")[0].split("?")[0][-260:]
            time.sleep(min(4,.35*(attempt+1)))
    raise OSError(f"Author archive range failed {start}..{end}: {last}")


def index():
    inventory = OUT/"official_inventory.json"
    if inventory.is_file():
        return json.loads(inventory.read_bytes())
    tail = fetch(SIZE-65557,SIZE-1)
    e = tail.rindex(b"PK\x05\x06")
    disk, cd_disk, on_disk, n, cd_size, cd_off, comment = struct.unpack_from("<4H2IH",tail,e+4)
    assert disk == cd_disk == 0 and on_disk == n and cd_off+cd_size+22+comment == SIZE
    directory = fetch(cd_off,cd_off+cd_size-1)
    eocd = bytearray(tail[e:])
    struct.pack_into("<I",eocd,16,0)
    with zipfile.ZipFile(io.BytesIO(directory+eocd)) as f:
        all_members=f.infolist()
        rows=[]
        for z in all_members:
            p=Path(z.filename)
            if p.suffix.lower() not in ACCEPTED or p.name.startswith(".") or "__MACOSX" in p.parts:
                continue
            if not any(part.casefold() in {"apple","apples"} for part in p.parts):
                continue
            label=p.parts[-2].casefold().strip()
            m=re.fullmatch(r"([123])(?:st|nd|rd|th)\s*grade",label)
            if not m: raise ValueError(f"Unknown grade directory {label}")
            rows.append(dict(source_member=z.filename, grade=int(m[1])-1, offset=z.header_offset,
                             compressed_bytes=z.compress_size, bytes=z.file_size, crc32=f"{z.CRC:08x}",
                             compression=z.compress_type))
    if len(rows)<300: raise ValueError(f"Unexpected source count {len(rows)}; sample={all_members[:7]}")
    data=dict(dataset="AFruitDB V1",doi="10.17632/bz65dz2pbj.1",license="CC BY 4.0",
              source_url=URL,source_zip_bytes=SIZE,published_zip_sha256=SOURCE["sha256_hash"],
              central_directory_sha256=hashlib.sha256(directory).hexdigest(),all_zip_members=n,
              archive_integrity="Selected original files checked against central-directory CRC32. Full archive not downloaded.",
              rows=rows)
    dump(inventory,data)
    return data


def path_for(row):
    return OUT/"original_photos"/row["source_member"].split("/")[-2]/Path(row["source_member"]).name


def correct(blob,row):
    return len(blob)==row["bytes"] and f"{zlib.crc32(blob):08x}"==row["crc32"]


def download_group(rows):
    missing=[r for r in rows if not path_for(r).is_file() or not correct(path_for(r).read_bytes(),r)]
    if not missing: return 0
    start=missing[0]["offset"]
    last=missing[-1]
    end=last["offset"]+last["compressed_bytes"]+4095
    buf=fetch(start,end)
    for row in missing:
        i=row["offset"]-start
        if buf[i:i+4] != b"PK\x03\x04": raise ValueError("Wrong local file header")
        name_len,extra_len=struct.unpack_from("<HH",buf,i+26)
        name=buf[i+30:i+30+name_len].decode("utf8")
        if name!=row["source_member"]: raise ValueError("ZIP local path mismatch")
        j=i+30+name_len+extra_len
        packed=buf[j:j+row["compressed_bytes"]]
        if row["compression"] not in (0,8): raise ValueError("Unexpected ZIP compression")
        blob=zlib.decompress(packed,-15) if row["compression"]==8 else packed
        if not correct(blob,row): raise ValueError("Size or CRC mismatch")
        with Image.open(io.BytesIO(blob)) as image:
            image.verify()
        path=path_for(row)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+".part");tmp.write_bytes(blob);tmp.replace(path)
    return len(missing)


def acquire():
    rows=sorted(index()["rows"],key=lambda v:v["offset"])
    if len({str(path_for(r)).casefold() for r in rows})!=len(rows): raise ValueError("Duplicate source paths")
    groups=[]
    for r in rows:
        if (not groups or r["offset"]+r["compressed_bytes"]-groups[-1][0]["offset"]>6*1024**2
            or r["offset"]-groups[-1][-1]["offset"]-groups[-1][-1]["compressed_bytes"]>20000):
            groups.append([])
        groups[-1].append(r)
    print("Inventory",len(rows),"apple files,",sum(r["bytes"] for r in rows),"bytes,",len(groups),"spans",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download_group,g) for g in groups]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if i%12==0 or i==len(groups): print("completed spans",i,"/",len(groups),flush=True)
    return rows


if __name__=="__main__":
    acquire()

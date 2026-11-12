"""Download pinned public research archives; never overwrite evidence."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ASSETS = [
    dict(name="mendeley_gfcmdbvw65_v4.zip", url="https://data.mendeley.com/public-files/datasets/gfcmdbvw65/files/f543b8ca-0adb-40ef-9ce5-540d2cb9ce14/file_downloaded", size=290131787,
         algorithm="sha256", checksum="049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce", license="CC BY 4.0", source="https://data.mendeley.com/datasets/gfcmdbvw65/4"),
    dict(name="ssda_v1.zip", url="https://zenodo.org/api/records/10957715/files/nbngochan/Surface-Subtle-Defects-Apple-v1.0.0.zip/content", size=160596573,
         algorithm="md5", checksum="86ad2aad9cbe1bdba8b85814a2cfe8c8", license="CC BY 4.0", source="https://zenodo.org/records/10957715"),
]


def main():
    out = Path(__file__).resolve().parent / "downloads"
    out.mkdir(exist_ok=True)
    for spec in ASSETS:
        destination = out / spec["name"]
        if destination.exists():
            digest = hashlib.new(spec["algorithm"], destination.read_bytes()).hexdigest()
            if destination.stat().st_size != spec["size"] or digest != spec["checksum"]:
                raise ValueError("Existing file does not match pinned source: " + str(destination))
            print("VERIFIED_EXISTING " + spec["name"], flush=True)
            continue
        partial = destination.with_suffix(".zip.part")
        started = time.time()
        print("START " + spec["name"], flush=True)
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "AppleResearch/1.0 (public dataset reproducibility)"})
        digest, sha256, count = hashlib.new(spec["algorithm"]), hashlib.sha256(), 0
        with urllib.request.urlopen(req, timeout=60) as response, partial.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > spec["size"]:
                    raise ValueError("Download exceeds pinned size")
                digest.update(block)
                sha256.update(block)
                stream.write(block)
                if count % (32 * 1024 * 1024) < len(block):
                    print(f"PROGRESS {spec['name']} {count}/{spec['size']}", flush=True)
        if count != spec["size"] or digest.hexdigest() != spec["checksum"]:
            raise ValueError("Download size/checksum differs; preserve .part for investigation")
        partial.rename(destination)
        receipt = {**spec, "bytes_received": count, "sha256": sha256.hexdigest(),
                   "verified": True, "elapsed_seconds": time.time()-started,
                   "retrieved_utc_epoch": time.time(), "archive_executed": False}
        destination.with_suffix(".receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
        print("COMPLETE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

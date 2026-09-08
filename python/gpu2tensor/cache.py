"""Bounded immutable compiler artifacts; never cache numerical results."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


class ArtifactCache:
    def __init__(self, directory, max_entries=128, max_bytes=64 * 1024 * 1024):
        self.directory = Path(directory)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        if max_entries < 1 or max_bytes < 1:
            raise ValueError('Cache bounds must be positive')
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(description):
        encoded = json.dumps(description, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key):
        path = self.directory / (key + '.zip')
        try:
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > self.max_bytes:
                    return None
                manifest = json.loads(archive.read('manifest.json'))
                if not isinstance(manifest, dict):
                    return None
                values = {name: archive.read(name) for name in manifest}
            if any(hashlib.sha256(value).hexdigest() != manifest[name] for name, value in values.items()):
                return None
            os.utime(path, None)
            return values
        except (FileNotFoundError, KeyError, ValueError, zipfile.BadZipFile):
            return None

    def put(self, key, values):
        if sum(len(value) for value in values.values()) > self.max_bytes:
            return
        # A racing worker can publish the same key safely. Readers see one
        # complete archive; partial or corrupt entries are treated as misses.
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            manifest = {name: hashlib.sha256(value).hexdigest() for name, value in values.items()}
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('manifest.json', json.dumps(manifest))
                for name, value in values.items():
                    archive.writestr(name, value)
            temporary.replace(self.directory / (key + '.zip'))
            paths = []
            for path in self.directory.glob('*.zip'):
                try:
                    stat = path.stat()
                    paths.append((stat.st_mtime_ns, path, stat.st_size))
                except FileNotFoundError:
                    pass
            total = 0
            for index, (_, path, size) in enumerate(sorted(paths, reverse=True)):
                total += size
                if index >= self.max_entries or total > self.max_bytes:
                    path.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)

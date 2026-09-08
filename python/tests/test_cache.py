import zipfile

from gpu2tensor.cache import ArtifactCache


def test_compiler_artifacts_are_hashed_and_corrupt_entries_miss(tmp_path):
    cache=ArtifactCache(tmp_path)
    key=cache.key({'source':'one','architecture':'sm_89'})
    cache.put(key,{'binary':b'compiled'})
    assert cache.get(key)=={'binary':b'compiled'}
    with zipfile.ZipFile(tmp_path / (key + '.zip')) as archive:
        manifest = archive.read('manifest.json')
    with zipfile.ZipFile(tmp_path / (key + '.zip'), 'w') as archive:
        archive.writestr('manifest.json', manifest)
        archive.writestr('binary', b'corrupted binary')
    assert cache.get(key) is None
    (tmp_path/(key+'.zip')).write_bytes(b'broken archive')
    assert cache.get(key) is None
    assert key != cache.key({'source':'two','architecture':'sm_89'})
    assert key != cache.key({'source':'one','architecture':'sm_80'})


def test_cache_bounds_are_enforced(tmp_path):
    cache=ArtifactCache(tmp_path,max_entries=2,max_bytes=4096)
    for index in range(3): cache.put(str(index),{'binary':bytes([index])*100})
    assert len(list(tmp_path.glob('*.zip')))==2
    cache.put('oversize',{'binary':b'x'*4097})
    assert cache.get('oversize') is None

# Copyright 2026 MESFET5198 (Lumina-Vec, Apache License 2.0)
# Modified 2026 by KasoluINT (Lumina-IMTVec): switched to pure-ONNX multi-CNN worker,
# removed legacy vtracer/presets/print_mm routes, added RGBA preview & AI repaint flags.
# See LICENSE-APACHE-2.0 and NOTICE for full license and attribution notices.
import io, json, os, re, subprocess, sys, threading, time, uuid
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file, abort
from lumina_vec import __version__
from lumina_vec.presets import PRESETS        # data only: keeps numpy / cv2 / torch out of startup

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('LVEC_DATA') or HERE / 'data')
UP = ROOT / 'uploads'; OUT = ROOT / 'outputs'
WEIGHTS = Path(os.environ.get('LVEC_WEIGHTS') or HERE / 'weights' / 'realesr-animevideov3.pth')
INDEX = Path(__file__).with_name('index.html')
UP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__); app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

jobs = OrderedDict()            # job id -> state dict (newest last); persisted to JOBS_FILE
lock = threading.Lock()
runner = threading.Semaphore(1)  # CPU-heavy: run one conversion at a time, queue the rest
MAX_PIXELS = 40_000_000
JOBS_FILE = ROOT / 'jobs.json'
VOLATILE = ('progress', 'stage')   # live-only fields, not worth a disk write per update


def _save():
    """Write the job list to /data (call with `lock` held). Atomic: tmp file + rename."""
    tmp = JOBS_FILE.with_suffix('.tmp')
    data = {j: {k: v for k, v in s.items() if k not in VOLATILE} for j, s in jobs.items()}
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, JOBS_FILE)


def _set(job, **kw):
    with lock:
        jobs[job].update(kw)
        if set(kw) - set(VOLATILE):
            _save()


def _load():
    """Job history survives restarts: read /data/jobs.json, mark jobs a restart cut off as
    interrupted, and rebuild records for result files that have none (older versions kept the
    list in memory only)."""
    data = {}
    if JOBS_FILE.is_file():
        try:
            data = json.loads(JOBS_FILE.read_text(encoding='utf8'))
        except (OSError, ValueError):
            JOBS_FILE.rename(JOBS_FILE.with_suffix('.broken'))
    for j, s in data.items():
        if s.get('status') in ('queued', 'running'):
            s.update(status='interrupted', error='服务重启，任务中断')
    for svg in OUT.glob('*.svg'):
        j = svg.stem
        if j in data:
            continue
        rec = dict(status='done', name=f'{j[:8]}（恢复）', recovered=True, created=svg.stat().st_mtime,
                   finished=svg.stat().st_mtime, svg=f'/file/{j}.svg', bytes=svg.stat().st_size)
        png = OUT / f'{j}.png'
        if png.is_file():
            rec['png'] = f'/file/{j}.png'
            try:
                with Image.open(png) as im:
                    rec['width'], rec['height'] = im.size
            except OSError:
                pass
        try:
            txt = svg.read_text(encoding='utf8', errors='ignore')
            rec['paths'] = txt.count('<path')
        except OSError:
            pass
        data[j] = rec
    for src in UP.glob('*.source'):
        if src.stem not in data:
            data[src.stem] = dict(status='interrupted', name=f'{src.stem[:8]}（恢复）', recovered=True,
                                  created=src.stat().st_mtime, error='没有结果文件')
    for j, s in sorted(data.items(), key=lambda kv: kv[1].get('created', 0)):
        jobs[j] = s
    with lock:
        _save()


def _queue_position(job):
    with lock:
        waiting = [j for j, s in jobs.items() if s['status'] == 'queued']
    return waiting.index(job) if job in waiting else 0


def work(job, raw, preset, overrides, fine=False):
    """在**子进程**里跑转换。

    矢量场 CNN 与平面分割计算会长时间占用原生计算线程并持有 GIL。
    改为独立子进程执行后，Web 服务全程保持低延迟响应，进度可实时刷新，且退出时自动完整回收显存。
    """
    src = UP / f'{job}.source'
    svgpath = OUT / f'{job}.svg'
    pngpath = OUT / f'{job}.png'
    hmpath = OUT / f'{job}.heatmap.png'
    try:
        if not src.is_file():
            raise RuntimeError('找不到上传的原图')
        cmd = [sys.executable, '-m', 'lumina_vec.worker', '--image', str(src), '--out', str(svgpath),
               '--preview', str(pngpath), '--heatmap', str(hmpath), '--preset', preset,
               '--overrides', json.dumps(overrides, ensure_ascii=False),
               '--fine', '1' if fine else '0']
        if WEIGHTS.is_file():
            cmd += ['--weights', str(WEIGHTS)]
        with runner:
            _set(job, status='running', started=time.time())
            env = dict(os.environ)
            env['PYTHONIOENCODING'] = 'utf-8'
            env.pop('PYTHONPATH', None)
            proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env=env, text=True, encoding='utf-8', errors='replace', bufsize=1)
            result = {}
            for line in proc.stdout:
                if line.startswith('#PROGRESS '):
                    try:
                        _, frac, label = line.rstrip('\n').split(' ', 2)
                        _set(job, progress=round(min(1.0, float(frac)) * 0.95, 4), stage=label.strip())
                    except ValueError:
                        pass
                elif line.startswith('#RESULT '):
                    try:
                        result = json.loads(line[len('#RESULT '):])
                    except ValueError:
                        result = {}
            proc.wait()
            err = (proc.stderr.read() or '').strip()
            if proc.returncode != 0 or not svgpath.is_file():
                raise RuntimeError((err[-400:] if err else f'转换进程退出码 {proc.returncode}'))
            w = result.get('width'); h = result.get('height')
            png_ok = pngpath.is_file()
            hm_ok = hmpath.is_file()
            _thumb(job)
            _set(job, status='done', progress=1.0, stage='完成', svg=f'/file/{job}.svg',
                 **({'png': f'/file/{job}.png', 'png_kind': 'raster'} if png_ok else {}),
                 **({'heatmap': f'/file/{job}.heatmap.png'} if hm_ok else {}),
                 width=w, height=h, paths=result.get('paths'), line_paths=result.get('line_paths', 0),
                 fill_paths=result.get('fill_paths', 0), vector_field=bool(result.get('vector_field')),
                 repaired_hairs=result.get('repaired_hairs', 0),
                 colors=result.get('colors'), bytes=result.get('bytes') or svgpath.stat().st_size,
                 seconds=result.get('seconds'), version=__version__, preset=result.get('preset'),
                 grain=result.get('grain'), params=result.get('params'), finished=time.time())
    except Exception as e:
        _set(job, status='error', error=str(e)[:500])
    finally:
        _free_gpu()


def _free_gpu():
    """Release any cached ONNX/GPU memory (worker subprocesses already reclaim all VRAM on exit)."""
    try:
        import gc
        from lumina_vec.feature_routing import clear_session_cache
        clear_session_cache()
        gc.collect()
    except Exception:
        pass


THUMB_PX = 200


def _thumb(job):
    """Small JPEG for the history list (composites RGBA previews onto a light neutral backdrop)."""
    src = OUT / f'{job}.png'; dst = OUT / f'{job}.thumb.jpg'
    if dst.is_file() or not src.is_file():
        return dst if dst.is_file() else None
    try:
        with Image.open(src) as im:
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                rgba = im.convert('RGBA')
                bg = Image.new('RGBA', rgba.size, (242, 242, 246, 255))
                im = Image.alpha_composite(bg, rgba).convert('RGB')
            else:
                im = im.convert('RGB')
            im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
            im.save(dst, 'JPEG', quality=82)
        return dst
    except OSError:
        return None


@app.get('/')
def index():
    return INDEX.read_text(encoding='utf8')


@app.post('/api/process')
def process():
    up = request.files.get('image')
    preset = request.form.get('preset', 'print-regions').strip() or 'print-regions'
    if not up or not up.filename:
        return jsonify(error='请选择图片'), 400
    if preset == 'auto':
        preset = 'print-regions'
    if preset not in PRESETS:
        return jsonify(error='预设无效'), 400
    overrides = {}
    bands = request.form.get('bands', '').strip()
    if bands:
        if not bands.isdigit() or not (8 <= int(bands) <= 256):
            return jsonify(error='颜色数需在 8–256 之间'), 400
        overrides['bands'] = int(bands)
    fine = request.form.get('fine', '').strip() == '1'
    gradient_mode = request.form.get('gradient_mode', '').strip()
    if gradient_mode in ('layered', 'clean'):
        overrides['gradient_mode'] = gradient_mode
    if request.form.get('redraw_source') in ('0', '1'):
        overrides['redraw_source'] = request.form.get('redraw_source') == '1'
    if request.form.get('smooth_lines') in ('0', '1'):
        overrides['smooth_lines'] = request.form.get('smooth_lines') == '1'
    if request.form.get('feature_routing') in ('0', '1'):
        overrides['feature_routing'] = request.form.get('feature_routing') == '1'
    if request.form.get('color_merge') in ('0', '1'):
        overrides['color_merge'] = request.form.get('color_merge') == '1'
    sr_scale = request.form.get('sr_scale', '').strip()
    if sr_scale in ('2', '5', '10', '2.0', '5.0', '10.0'):
        overrides['sr_scale'] = float(sr_scale)
    raw = up.read(); job = uuid.uuid4().hex
    (UP / f'{job}.source').write_bytes(raw)
    with lock:
        jobs[job] = {'status': 'queued', 'name': Path(up.filename).name[:120], 'created': time.time(),
                     'request': dict(preset=preset, overrides=overrides, fine=fine)}
        _save()
    threading.Thread(target=work, args=(job, raw, preset, overrides, fine), daemon=True).start()
    return jsonify(job=job, status='queued')


@app.post('/api/jobs/<job>/retry')
def retry(job):
    """Run an interrupted or failed job again with the settings it was submitted with."""
    src = UP / f'{job}.source'
    with lock:
        s = jobs.get(job)
        if not s:
            return jsonify(error='任务不存在'), 404
        if s['status'] in ('queued', 'running'):
            return jsonify(error='任务还在进行'), 409
        req = s.get('request')
        if not req or not src.is_file():
            return jsonify(error='缺少原图或转换设置，无法重试'), 409
        for k in ('error', 'svg', 'png', 'started', 'finished', 'seconds', 'paths', 'params'):
            s.pop(k, None)
        s.update(status='queued', created=time.time())
        jobs.move_to_end(job)
        _save()
    threading.Thread(target=work, args=(job, src.read_bytes(), req['preset'], req['overrides'], req['fine']),
                     daemon=True).start()
    return jsonify(job=job, status='queued')


@app.delete('/api/jobs/<job>')
def delete_job(job):
    """Delete a job's record and its files (original upload, SVG, PNG preview)."""
    with lock:
        s = jobs.get(job)
        if not s:
            return jsonify(error='任务不存在'), 404
        if s['status'] in ('queued', 'running'):
            return jsonify(error='任务还在进行，不能删除'), 409
        jobs.pop(job)
        _save()
    for f in (UP / f'{job}.source', OUT / f'{job}.svg', OUT / f'{job}.png',
              OUT / f'{job}.thumb.jpg', OUT / f'{job}.heatmap.png'):
        f.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get('/api/status/<job>')
def status(job):
    with lock:
        result = dict(jobs[job]) if job in jobs else None
    if not result:
        return jsonify(error='任务不存在'), 404
    if result['status'] == 'queued':
        result['ahead'] = _queue_position(job)
    if result['status'] == 'running':
        result['elapsed'] = round(time.time() - result['started'], 1)
    return jsonify(result)


@app.get('/api/jobs')
def list_jobs():
    """Newest first, paged: ?limit=20&offset=0. Returns {items, total, offset}."""
    limit = min(max(request.args.get('limit', 20, type=int), 1), 200)
    offset = max(request.args.get('offset', 0, type=int), 0)
    with lock:
        allj = list(reversed(jobs.items()))
        page = allj[offset:offset + limit]
        items = [dict(id=j, retry=bool(s.get('request')),
                      thumb=f'/thumb/{j}' if s.get('png') else None,
                      **{k: s.get(k) for k in ('status', 'name', 'png', 'svg', 'preset', 'seconds', 'paths',
                                               'error', 'created')})
                 for j, s in page]
    return jsonify(items=items, total=len(allj), offset=offset)


@app.get('/thumb/<job>')
def thumb(job):
    if not re.fullmatch(r'[0-9a-f]{32}', job):
        abort(404)
    p = _thumb(job)
    if not p:
        abort(404)
    resp = send_file(p, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


@app.get('/file/<name>')
def file(name):
    if Path(name).name != name or Path(name).suffix not in ('.svg', '.png', '.source'):
        abort(404)
    base = UP if name.endswith('.source') else OUT
    path = base / name
    if not path.is_file() and name.endswith('.heatmap.png'):
        job_id = name[:-len('.heatmap.png')]
        src_p = UP / f'{job_id}.source'
        if src_p.is_file():
            try:
                import numpy as np
                from lumina_vec.feature_routing import render_vector_heatmap_png
                with Image.open(src_p) as im:
                    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                        rgba = im.convert('RGBA')
                        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                        rgb_im = Image.alpha_composite(bg, rgba).convert('RGB')
                    else:
                        rgb_im = im.convert('RGB')
                    rgb_np = np.asarray(rgb_im)
                render_vector_heatmap_png(rgb_np, path)
            except Exception:                # noqa: BLE001
                pass
            finally:
                _free_gpu()
    if not path.is_file():
        abort(404)
    layer = request.args.get('layer', '').strip()
    if name.endswith('.svg') and layer in ('line', 'fill'):
        import xml.etree.ElementTree as ET
        ns_uri = 'http://www.w3.org/2000/svg'
        ET.register_namespace('', ns_uri)
        root = ET.fromstring(path.read_text(encoding='utf8'))
        keep = {'region-boundaries'} if layer == 'line' else {'region-colours', 'line-art-layer'}
        for child in list(root):
            gid = child.attrib.get('id', '')
            if child.tag.endswith('g') and gid and gid not in keep:
                root.remove(child)
        return app.response_class(ET.tostring(root, encoding='unicode'), mimetype='image/svg+xml')
    return send_file(path, as_attachment=request.args.get('download') == '1', download_name=name)


_load()


def _cnn_status():
    wdir = HERE / 'weights'
    files = {n: (wdir / n).is_file() for n in (
        'surface_style.onnx', 'feature_part.onnx', 'defect_repair_net.onnx',
        'line_net.onnx', 'line_vector_net.onnx', 'element_grad.onnx', 'realesr_anime.onnx',
        'anime_part_seg.onnx',
    )}
    ready = files['surface_style.onnx'] and files['feature_part.onnx']
    vector_ready = files['line_vector_net.onnx']
    defect_ready = files['defect_repair_net.onnx']
    dml = False
    try:
        import onnxruntime as ort
        dml = 'DmlExecutionProvider' in ort.get_available_providers()
    except Exception:                        # noqa: BLE001
        pass
    return dict(
        cnn_ready=ready,
        vector_ready=vector_ready,
        defect_ready=defect_ready,
        cnn_weights=files,
        directml=dml,
    )


@app.get('/healthz')
def health():
    cnn = _cnn_status()
    return jsonify(status='ok', version=__version__, weights=WEIGHTS.is_file() or cnn['cnn_ready'],
                   pth_weights=WEIGHTS.is_file(), cuda=_cuda_name(), presets=[*PRESETS], **cnn)


def _cuda_name():
    """Hardware GPU name and VRAM summary via DXGI/nvidia-smi without requiring PyTorch."""
    try:
        from lumina_vec.vram_guard import query_gpu_vram_mb
        name, total_mb, avail_mb = query_gpu_vram_mb()
        if name:
            return f'{name} (VRAM {total_mb / 1024:.1f} GB / 可用 {avail_mb / 1024:.1f} GB)'
    except Exception:                        # noqa: BLE001 - fallback cleanly to CPU info
        pass
    return None


def main():
    """Native entry point (no Docker): `python webui.py --port 8091`."""
    import argparse
    ap = argparse.ArgumentParser(description='Lumina-IMTVec web UI, run natively')
    ap.add_argument('--host', default=os.environ.get('LVEC_HOST', '0.0.0.0'),
                    help='bind address (default 0.0.0.0 so LAN devices can reach it)')
    ap.add_argument('--port', type=int, default=int(os.environ.get('LVEC_PORT', '8091')))
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    gpu = _cuda_name()
    cnn = _cnn_status()
    print(f'Lumina-IMTVec {__version__}')
    print(f'  url    : http://{a.host}:{a.port}   (LAN: http://<this-ip>:{a.port})')
    print(f'  data   : {ROOT}')
    print(f'  weights: {HERE / "weights"}  (ONNX ready={cnn["cnn_ready"]})')
    print(f'  cnn    : ready={cnn["cnn_ready"]} directml={cnn["directml"]} {cnn["cnn_weights"]}')
    print(f'  device : {gpu or ("ONNX DirectML" if cnn["directml"] else "CPU")}')
    if not cnn['cnn_ready']:
        print('  ! ONNX 权重缺失，请确认仓库 weights/ 目录下的 *.onnx 模型文件完整。')
    try:                                     # a real WSGI server if present: the dev server is fine
        from waitress import serve           # for a few users but serialises badly under load
        print('  server : waitress')
        serve(app, host=a.host, port=a.port)
    except ImportError:
        print('  server : flask dev server (pip install waitress 可换成更稳的服务)')
        app.run(host=a.host, port=a.port, threaded=True, debug=a.debug)


if __name__ == '__main__':
    main()

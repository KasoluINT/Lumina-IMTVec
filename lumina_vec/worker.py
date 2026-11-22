"""把一次转换放到独立进程里跑。

为什么要独立进程：描摹（vtracer 的 Rust 调用 + 大量 numpy/opencv 循环）是一次很长的原生调用，
长时间持有 GIL。在 Web 服务进程里直接跑，会把 waitress 的请求线程全部堵死——实测转换期间
/api/status 和 /healthz 都会超时（>3 s），网页就一直停在"排队中"，直到转换结束才恢复。

用法（webui 内部调用，命令行也能直接用）：

    python -m lumina_vec.worker --image in.png --out out.svg --preset auto \
        --overrides '{"structure":"lumina"}' [--preview preview.png] [--fine 1] [--weights w.pth]

标准输出约定（webui 逐行解析）：
    #PROGRESS <0..1> <阶段名>
    #RESULT {json}
错误走 stderr 并以非 0 退出码结束。
"""
import argparse
import json
import os
import re
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="run one conversion (child process of the web UI)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default=None)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--fine", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--heatmap", default=None)
    ap.add_argument("--preview-max", type=int, default=1600)
    a = ap.parse_args()
    if a.heatmap:
        os.environ["LUMINA_HEATMAP_OUT"] = a.heatmap
    else:
        os.environ.pop("LUMINA_HEATMAP_OUT", None)

    from PIL import Image
    from .pipeline import auto_params, fine_params, grain_level, vectorize
    from .presets import PRESETS

    def prog(frac, label):
        print("#PROGRESS %.4f %s" % (max(0.0, min(1.0, float(frac))), label), flush=True)

    img = Image.open(a.image)
    img.load()
    w, h = img.size
    over = json.loads(a.overrides or "{}")
    if a.preset == "auto":
        name, grain, params = auto_params(img)
    else:
        name, grain, params = a.preset, grain_level(img), dict(PRESETS[a.preset])
    if a.fine:
        params = fine_params(img, params)
    params.update(over)
    params["fine"] = bool(a.fine)

    t = time.time()
    svg, work = vectorize(img, a.weights, ai=True, progress=prog,
                          **{k: v for k, v in params.items() if k != "fine"})
    with open(a.out, "w", encoding="utf8") as f:
        f.write(svg)

    png = False
    if a.preview:
        try:
            im = work if work.mode == "RGBA" else work.convert("RGB")
            if max(w, h) > a.preview_max:
                k = a.preview_max / max(w, h)
                im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
            im.save(a.preview, "PNG")
            png = True
        except Exception:                      # noqa: BLE001 - 预览失败不影响结果
            png = False

    colors = len(set(m.lower() for m in re.findall(r'(?:fill|stroke)="(#[0-9a-fA-F]{6})"', svg)))
    import xml.etree.ElementTree as _ET
    line_paths, fill_paths = 0, 0
    try:
        _r = _ET.fromstring(svg)
        for _g in _r.iter():
            if _g.tag.endswith("g"):
                gid = _g.attrib.get("id", "")
                if gid == "region-boundaries":
                    line_paths += sum(1 for _e in _g.iter() if _e.tag.endswith("path"))
                elif gid in ("region-colours", "line-art-layer"):
                    fill_paths += sum(1 for _e in _g.iter() if _e.tag.endswith("path"))
    except Exception:
        pass
    print("#RESULT " + json.dumps(dict(
        paths=svg.count("<path"), colors=colors, bytes=os.path.getsize(a.out),
        line_paths=line_paths, fill_paths=fill_paths,
        vector_field=('cnn-vector-field-layer' in svg),
        boundary_paths=svg.count('class="region-gap-filler"'),
        filled_regions=len(re.findall(r'<path\b[^>]*\bfill="#[0-9a-fA-F]{6}"', svg)),
        seconds=round(time.time() - t, 1), preset=name, grain=round(float(grain), 2),
        params=params, width=w, height=h, repaired_hairs=svg.count('id="hair-'), png=png,
        heatmap=bool(a.heatmap and os.path.isfile(a.heatmap)))),
        flush=True)


if __name__ == "__main__":
    main()

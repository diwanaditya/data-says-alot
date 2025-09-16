from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# try imports, but don't crash with unreadable trace - give helpful hints
try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    print('Install Pillow: pip install Pillow')
    raise

try:
    import exifread
except Exception:
    print('Install exifread: pip install exifread')
    raise

try:
    from PyPDF2 import PdfReader
except Exception:
    print('Install PyPDF2: pip install PyPDF2')
    raise

try:
    import numpy as np
except Exception:
    print('Install numpy: pip install numpy')
    raise

try:
    from sklearn.ensemble import IsolationForest
except Exception:
    print('Install scikit-learn: pip install scikit-learn')
    raise

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    print('Install matplotlib: pip install matplotlib')
    raise

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
except Exception:
    print('Install reportlab: pip install reportlab')
    raise

try:
    import joblib
except Exception:
    print('Install joblib: pip install joblib')
    raise

# ---------------- utilities

def now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def compute_hashes(path: str) -> Dict[str, str]:
    import hashlib
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
    return {'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}


# ---------------- ingestion & simple sniff

class Ingest:
    IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'heic', 'webp', 'tiff'}
    VIDEO_EXTS = {'mp4', 'mov', 'mkv', 'avi', 'webm'}
    DOC_EXTS = {'pdf'}

    def __init__(self, path: str):
        self.root = os.path.abspath(path)
        self.is_dir = os.path.isdir(self.root)

    def discover(self) -> List[str]:
        out = []
        if self.is_dir:
            for r, d, files in os.walk(self.root):
                for fn in files:
                    p = os.path.join(r, fn)
                    if not os.path.islink(p):
                        out.append(p)
        else:
            out = [self.root]
        out.sort()
        return out

    @staticmethod
    def category(path: str) -> str:
        ext = os.path.splitext(path)[1].lower().strip('.')
        if ext in Ingest.IMAGE_EXTS:
            return 'image'
        if ext in Ingest.VIDEO_EXTS:
            return 'video'
        if ext in Ingest.DOC_EXTS:
            return 'pdf'
        mime, _ = mimetypes.guess_type(path)
        if mime:
            if mime.startswith('image'):
                return 'image'
            if mime.startswith('video'):
                return 'video'
            if mime == 'application/pdf':
                return 'pdf'
        return 'unknown'


# ---------------- metadata extraction

class MetadataExtractor:
    def image_meta(self, path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {'path': path}
        out['size'] = os.path.getsize(path)
        out['hashes'] = compute_hashes(path)
        try:
            with open(path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                out['exif'] = {k: str(v) for k, v in tags.items()}
        except Exception as e:
            out['exif_error'] = str(e)

        try:
            img = Image.open(path)
            out['format'] = img.format
            out['mode'] = img.mode
            out['size_px'] = img.size
            thumb = ImageOps.exif_transpose(img).copy()
            thumb.thumbnail((800, 800))
            bio = io.BytesIO()
            thumb.save(bio, 'JPEG', quality=70)
            out['thumb_bytes'] = bio.getvalue()
        except Exception as e:
            out['img_error'] = str(e)

        gps = self._extract_gps(out.get('exif', {}))
        if gps:
            out['gps'] = gps
        return out

    def _extract_gps(self, exif: Dict[str, Any]) -> Optional[Dict[str, float]]:
        # simple parse
        if not exif:
            return None
        lat_k = None
        lon_k = None
        lat_ref_k = None
        lon_ref_k = None
        for k in exif:
            lk = k.lower()
            if 'gpslatitude' in lk:
                lat_k = k
            if 'gpslongitude' in lk:
                lon_k = k
            if 'gpslatituderef' in lk:
                lat_ref_k = k
            if 'gpslongituderef' in lk:
                lon_ref_k = k
        if not lat_k or not lon_k:
            return None
        try:
            lat_val = exif[lat_k]
            lon_val = exif[lon_k]
            lat_ref = exif.get(lat_ref_k, 'N')
            lon_ref = exif.get(lon_ref_k, 'E')
            return {'latitude': self._to_decimal(str(lat_val), lat_ref), 'longitude': self._to_decimal(str(lon_val), lon_ref)}
        except Exception:
            return None

    def _to_decimal(self, val: str, ref: str) -> float:
        parts = re.findall(r"[0-9]+/[0-9]+|[0-9]+\.?[0-9]*", val)
        nums = []
        for p in parts:
            if '/' in p:
                a, b = p.split('/')
                nums.append(float(a) / float(b))
            else:
                nums.append(float(p))
        if len(nums) >= 3:
            d, m, s = nums[:3]
            dec = d + m / 60.0 + s / 3600.0
        elif len(nums) == 2:
            d, m = nums
            dec = d + m / 60.0
        else:
            dec = nums[0]
        if ref.strip().upper() in ('S', 'W'):
            dec = -abs(dec)
        return dec

    def pdf_meta(self, path: str) -> Dict[str, Any]:
        out = {'path': path}
        out['size'] = os.path.getsize(path)
        out['hashes'] = compute_hashes(path)
        try:
            r = PdfReader(path)
            out['pdf_info'] = {k: str(v) for k, v in (r.metadata.items() if r.metadata else [])}
            out['num_pages'] = len(r.pages)
            raw = open(path, 'rb').read()
            if b'/JavaScript' in raw or b'/JS' in raw:
                out.setdefault('flags', []).append('pdf_js')
            if b'/EmbeddedFile' in raw:
                out.setdefault('flags', []).append('embedded_file')
            if raw.rfind(b'%%EOF') != len(raw) - 5:
                out.setdefault('flags', []).append('appended_data')
        except Exception as e:
            out['pdf_error'] = str(e)
        return out

    def video_meta(self, path: str) -> Dict[str, Any]:
        out = {'path': path}
        out['size'] = os.path.getsize(path)
        out['hashes'] = compute_hashes(path)
        try:
            cmd = ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', path]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0 and p.stdout:
                out['ffprobe'] = json.loads(p.stdout.decode('utf-8', errors='ignore'))
            else:
                out['ffprobe_error'] = p.stderr.decode('utf-8', errors='ignore')
        except FileNotFoundError:
            out['ffprobe_error'] = 'ffprobe missing'
        except Exception as e:
            out['ffprobe_error'] = str(e)
        return out


# ---------------- analyzer heuristics & ML stego

class Analyzer:
    def __init__(self, stego_model_path: Optional[str] = None):
        self.model = None
        self.model_path = stego_model_path
        if stego_model_path and os.path.exists(stego_model_path):
            try:
                self.model = joblib.load(stego_model_path)
            except Exception:
                self.model = None

    # simple static heuristics
    def appended_data(self, path: str) -> Dict[str, Any]:
        res = {}
        try:
            data = open(path, 'rb').read()
            if b'PK' in data:
                res['zip'] = True
            if b'ELF' in data:
                res['elf'] = True
            if b'MZ' in data:
                res['pe'] = True
        except Exception as e:
            res['error'] = str(e)
        return res

    # LSB ratio
    def lsb_ratio(self, path: str) -> Optional[float]:
        try:
            img = Image.open(path).convert('RGB')
            pixels = list(img.getdata())
            ones = 0
            total = 0
            for (r, g, b) in pixels:
                ones += (r & 1) + (g & 1) + (b & 1)
                total += 3
            return ones / total if total else None
        except Exception:
            return None

    def entropy(self, path: str) -> Optional[float]:
        try:
            arr = np.array(Image.open(path).convert('L'))
            hist, _ = np.histogram(arr, bins=256, range=(0, 255))
            prob = hist / hist.sum()
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log2(prob)))
        except Exception:
            return None

    def noise_score(self, path: str) -> Optional[float]:
        try:
            img = Image.open(path).convert('L')
            img_np = np.array(img)
            blurred = np.array(img.filter(ImageFilter.MedianFilter(size=3)))
            diff = img_np.astype(np.int32) - blurred.astype(np.int32)
            return float(np.mean(np.abs(diff)))
        except Exception:
            return None

    def feature_vector(self, path: str) -> Optional[List[float]]:
        lsb = self.lsb_ratio(path)
        ent = self.entropy(path)
        noise = self.noise_score(path)
        if lsb is None and ent is None and noise is None:
            return None
        return [lsb if lsb is not None else 0.0, ent if ent is not None else 0.0, noise if noise is not None else 0.0]

    # ML anomaly detection (unsupervised)
    def train_stego_model(self, clean_corpus_dir: str, out_path: str, n_estimators: int = 100) -> Dict[str, Any]:
        feats = []
        files = []
        for r, d, fs in os.walk(clean_corpus_dir):
            for fn in fs:
                p = os.path.join(r, fn)
                if Ingest.category(p) != 'image':
                    continue
                fv = self.feature_vector(p)
                if fv is not None:
                    feats.append(fv)
                    files.append(p)
        if not feats:
            return {'error': 'no features found'}
        X = np.array(feats)
        model = IsolationForest(n_estimators=n_estimators, contamination=0.01, random_state=42)
        model.fit(X)
        joblib.dump(model, out_path)
        self.model = model
        self.model_path = out_path
        return {'trained_on': len(feats), 'model_path': out_path}

    def predict_stego(self, path: str) -> Dict[str, Any]:
        fv = self.feature_vector(path)
        out = {'path': path}
        if fv is None:
            out['model_error'] = 'no features'
            return out
        out['features'] = fv
        if self.model is None:
            out['model_error'] = 'no model loaded'
            return out
        score = float(self.model.decision_function([fv])[0])
        pred = int(self.model.predict([fv])[0])
        out['anomaly_score'] = score
        out['anomaly_pred'] = 'suspicious' if pred == -1 else 'normal'
        return out


# ---------------- sandbox orchestration (firejail preferred)

def run_in_sandbox(cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    # try firejail, otherwise show instructions
    out = {'cmd': cmd}
    try:
        if shutil.which('firejail'):
            full = ['firejail', '--private', '--net=none', '--quiet'] + cmd
            p = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            out['returncode'] = p.returncode
            out['stdout'] = p.stdout.decode('utf-8', errors='ignore')
            out['stderr'] = p.stderr.decode('utf-8', errors='ignore')
        else:
            out['error'] = 'firejail not installed; run in VM/QEMU for dynamic analysis'
    except subprocess.TimeoutExpired:
        out['error'] = 'timeout'
    except Exception as e:
        out['error'] = str(e)
    return out


# ---------------- report generator with charts

class Report:
    def __init__(self, out_pdf: str, title: str = 'data says lot'):
        self.out_pdf = out_pdf
        self.title = title
        self.styles = getSampleStyleSheet()
        self.story = []

    def cover(self, meta: Dict[str, Any]):
        self.story.append(Paragraph(self.title, ParagraphStyle('T', fontName='Times-Roman', fontSize=18, leading=22)))
        self.story.append(Spacer(1, 6 * mm))
        self.story.append(Paragraph(f'Generated: {now_iso()}', self.styles['Normal']))
        self.story.append(Spacer(1, 6 * mm))
        tbl = [['Files analyzed', str(meta.get('file_count', 0))], ['Note', 'See appendix for machine JSON']]
        t = Table(tbl, hAlign='LEFT')
        t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Times-Roman')]))
        self.story.append(t)
        self.story.append(PageBreak())

    def add_file(self, info: Dict[str, Any], idx: int):
        h = Paragraph(f"{idx+1}. {os.path.basename(info.get('path',''))}", self.styles['Heading2'])
        self.story.append(h)
        self.story.append(Spacer(1, 4 * mm))
        # basic table
        tbl = [['Size', str(info.get('size', 'n/a'))], ['Category', info.get('category', 'n/a')]]
        if info.get('hashes'):
            tbl.append(['SHA256', info['hashes'].get('sha256')])
        t = Table(tbl)
        t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Times-Roman')]))
        self.story.append(t)
        self.story.append(Spacer(1, 4 * mm))
        # thumbnail
        if info.get('thumb_bytes'):
            bio = io.BytesIO(info['thumb_bytes'])
            img = RLImage(bio, width=120 * mm, height=None)
            self.story.append(img)
            self.story.append(Spacer(1, 4 * mm))
        # analyzer
        if info.get('analysis'):
            lines = '\n'.join([f"{k}: {v}" for k, v in info['analysis'].items()])
            self.story.append(Paragraph(lines, self.styles['Normal']))
        self.story.append(PageBreak())

    def add_chart(self, fig_path: str, caption: str = ''):
        img = RLImage(fig_path, width=160 * mm, height=None)
        self.story.append(img)
        if caption:
            self.story.append(Paragraph(caption, self.styles['Italic']))
        self.story.append(Spacer(1, 4 * mm))
        self.story.append(PageBreak())

    def appendix_json(self, data: List[Dict[str, Any]]):
        txt = json.dumps(data, indent=2)
        for i in range(0, len(txt), 4000):
            chunk = txt[i:i+4000]
            self.story.append(Paragraph('<pre>' + chunk + '</pre>', ParagraphStyle('Mono', fontName='Courier', fontSize=8)))
            self.story.append(Spacer(1, 2 * mm))

    def build(self):
        doc = SimpleDocTemplate(self.out_pdf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm)
        doc.build(self.story)


# ---------------- orchestration

def analyze(paths: List[str], analyzer: Analyzer, extractor: MetadataExtractor) -> Dict[str, Any]:
    results = []
    for p in paths:
        rec = {'path': p, 'category': Ingest.category(p)}
        try:
            if rec['category'] == 'image':
                meta = extractor.image_meta(p)
                rec.update(meta)
                rec['analysis'] = {}
                rec['analysis']['appended'] = analyzer.appended_data(p)
                rec['analysis']['lsb'] = analyzer.lsb_ratio(p)
                rec['analysis']['entropy'] = analyzer.entropy(p)
                rec['analysis']['noise'] = analyzer.noise_score(p)
                if analyzer.model:
                    rec['analysis']['ml'] = analyzer.predict_stego(p)
            elif rec['category'] == 'pdf':
                meta = extractor.pdf_meta(p)
                rec.update(meta)
                rec['analysis'] = analyzer.appended_data(p)
            elif rec['category'] == 'video':
                meta = extractor.video_meta(p)
                rec.update(meta)
                rec['analysis'] = analyzer.appended_data(p)
            else:
                rec['size'] = os.path.getsize(p)
                rec['hashes'] = compute_hashes(p)
                rec['analysis'] = analyzer.appended_data(p)
        except Exception as e:
            rec['error'] = str(e)
        results.append(rec)
    return {'generated': now_iso(), 'file_count': len(results), 'files': results}


# ---------------- testing & benchmarking

def run_tests():
    # basic smoke tests
    print('Running quick unit tests...')
    tmp = 'test_tmp'
    os.makedirs(tmp, exist_ok=True)
    # create a small image
    p = os.path.join(tmp, 't1.png')
    Image.new('RGB', (64, 64), color=(123, 222, 111)).save(p)
    ing = Ingest(p)
    assert ing.discover() == [os.path.abspath(p)]
    ex = MetadataExtractor()
    a = Analyzer()
    res = analyze([p], a, ex)
    assert res['file_count'] == 1
    print('Basic tests passed.')


def benchmark(paths: List[str], analyzer: Analyzer, extractor: MetadataExtractor):
    t0 = time.time()
    out = analyze(paths, analyzer, extractor)
    t1 = time.time()
    print(f'Analyzed {len(paths)} files in {t1-t0:.2f}s')
    # simple stego benchmark if model exists
    if analyzer.model:
        scores = []
        for f in paths:
            if Ingest.category(f) != 'image':
                continue
            pred = analyzer.predict_stego(f)
            scores.append(pred.get('anomaly_score'))
        print('Anomaly scores (sample):', scores[:10])
    return out


# ---------------- small helper to create charts

def make_lsb_hist(files: List[str], analyzer: Analyzer, out_png: str):
    vals = []
    for f in files:
        if Ingest.category(f) != 'image':
            continue
        v = analyzer.lsb_ratio(f)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    plt.figure(figsize=(6, 3))
    plt.hist(vals, bins=30)
    plt.title('LSB ratio distribution')
    plt.xlabel('LSB ratio')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return out_png


# ---------------- CLI

def build_cli():
    p = argparse.ArgumentParser(description='data says lot (human-style updated)')
    p.add_argument('-i', '--input', required=True, help='file or folder')
    p.add_argument('-o', '--output', required=True, help='pdf report path')
    p.add_argument('--json', help='save machine json')
    p.add_argument('--train-clean', help='train stego model on clean corpus and save model file')
    p.add_argument('--model', help='load existing stego model (.joblib)')
    p.add_argument('--run-tests', action='store_true')
    p.add_argument('--benchmark', action='store_true')
    return p


def main(argv: Optional[List[str]] = None):
    args = build_cli().parse_args(argv)
    if args.run_tests:
        run_tests()
        return

    extractor = MetadataExtractor()
    analyzer = Analyzer(stego_model_path=args.model)

    if args.train_clean:
        print('Training stego model...')
        r = analyzer.train_stego_model(args.train_clean, args.train_clean + '.joblib')
        print(r)
        analyzer = Analyzer(stego_model_path=args.train_clean + '.joblib')

    path = args.input
    if os.path.isdir(path):
        paths = Ingest(path).discover()
    else:
        paths = [path]

    if args.benchmark:
        out = benchmark(paths, analyzer, extractor)
    else:
        out = analyze(paths, analyzer, extractor)

    # build charts
    charts = []
    hist_png = 'lsb_hist.png'
    c = make_lsb_hist(paths, analyzer, hist_png)
    if c:
        charts.append((c, 'LSB ratio histogram'))

    # report
    rpt = Report(args.output, title='data says lot')
    rpt.cover({'file_count': out['file_count']})
    for idx, f in enumerate(out['files']):
        rpt.add_file(f, idx)
    for fig, cap in charts:
        rpt.add_chart(fig, cap)
    rpt.appendix_json(out['files'])
    rpt.build()

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as jf:
            json.dump(out, jf, indent=2)
        print('Wrote JSON to', args.json)

    print('Done. Report at', args.output)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Fatal:', e)
        traceback.print_exc()

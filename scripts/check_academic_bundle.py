"""Headless checks, PDF export and artifact readback for the local presentation.

Requires Playwright, Chromium and PyMuPDF in the checking environment. Checks the
offline report and PPTX contents/geometry; does not claim native Office rendering.
"""

import argparse
import hashlib
import json
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fitz
from playwright.sync_api import sync_playwright
from pptx import Presentation


@contextmanager
def preview(directory):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown(); server.server_close(); thread.join()


def main(output):
    if (output / "verification.json").exists() or (output / "report.pdf").exists():
        raise FileExistsError("verification/export already exists; use a fresh bundle")
    errors, requests = [], []
    with preview(output) as local_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                             args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto(local_url + "report.html", wait_until="networkidle")
        assert page.locator(".market-panel:visible").count() == 1
        assert page.locator('.market-panel[data-market="TAIEX"]').is_visible()
        page.select_option("#market", "TX_selected_monthly")
        assert page.locator('.market-panel[data-market="TX_selected_monthly"]').is_visible()
        before = page.locator("#frozen-prices").inner_text()
        morning = page.locator("#demo-text").inner_text()
        page.click("#close-button")
        assert page.locator("#demo-text").inner_text() != morning
        assert page.locator("#frozen-prices").inner_text() == before
        page.click("#morning-button")
        assert page.locator("#demo-text").inner_text() == morning
        assert page.locator("#frozen-prices").inner_text() == before
        page.select_option("#market", "TAIEX")
        assert page.locator("img").evaluate_all("imgs => imgs.every(i => i.complete && i.naturalWidth > 0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.evaluate("scrollTo(0,0)")
        page.screenshot(path=str(output / "report-preview.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(output / "mobile-preview.png"))
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.emulate_media(media="print")
        assert page.locator(".market-panel:visible").count() == 2
        page.pdf(path=str(output / "report.pdf"), print_background=True, prefer_css_page_size=True)
        browser.close()
    assert not errors, errors
    assert not [u for u in requests if u.startswith(("http:", "https:")) and not u.startswith(local_url)], \
        "offline report requested an external service"
    document = fitz.open(output / "report.pdf")
    contents = "\n".join(p.get_text() for p in document)
    assert len(document) >= 8
    for term in ("MAE", "RMSE", "TAIEX", "TX", "179", "合成", "TimesFM"):
        assert term in contents, f"PDF missing {term}"
    # Contact sheet supports visual inspection of pagination without replacing the PDF.
    from PIL import Image, ImageDraw
    thumb_w, thumb_h = 240, 350
    rows = (len(document) + 3) // 4
    contact = Image.new("RGB", (4 * thumb_w, rows * thumb_h), "#dce4ec")
    draw = ImageDraw.Draw(contact)
    for i, pdf_page in enumerate(document):
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(.35, .35), alpha=False)
        thumb = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        thumb.thumbnail((thumb_w - 12, thumb_h - 28))
        x, y = i % 4 * thumb_w + 6, i // 4 * thumb_h + 20
        contact.paste(thumb, (x, y)); draw.text((x, y - 16), f"Page {i + 1}", fill="#18364b")
    contact.save(output / "pdf-contact-sheet.png")
    slides = Presentation(output / "presentation.pptx")
    assert len(slides.slides) == 16
    for slide in slides.slides:
        assert slide.notes_slide.notes_text_frame.text.strip()
        for shape in slide.shapes:
            assert 0 <= shape.left < slides.slide_width and 0 <= shape.top < slides.slide_height
            assert shape.left + shape.width <= slides.slide_width + 2
            assert shape.top + shape.height <= slides.slide_height + 2
    result = {"offline_html_passed": True, "desktop_width": 1440, "mobile_width": 390,
              "horizontal_overflow": False, "market_selection_passed": True,
              "synthetic_frozen_prices_unchanged": True, "javascript_errors": errors,
              "external_network_requests": 0, "pdf_pages": len(document),
              "pptx_slides": len(slides.slides), "pptx_notes_and_geometry_checked": True,
              "native_office_rendering_checked": False, "model_fits": 0,
              "real_sheet_write": False, "new_market_accuracy_claim": False}
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    manifest = {"scope": "final locally verified delivery bundle", "verification": result,
                "files_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "delivery-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())

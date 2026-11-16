"""Render frozen evidence into scientific figures, an offline report and slides.

No training, model selection, network calls, or publication. Full plotted results
stay in the requested local output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

FIELDS = ("open", "high", "low", "close")
METHODS = ("without_night", "with_night_gap_only", "night_carry_with_mean_body")
NAMES = {"without_night": "Ridge：無夜盤", "with_night_gap_only": "Ridge：僅開盤加夜盤",
         "night_carry_with_mean_body": "預列夜盤直推基準"}
MARKETS = {"TAIEX": "TAIEX 加權指數", "TX_selected_monthly": "TX 盤前選月合約"}
LABELS = {"open": "開盤 O", "high": "最高 H", "low": "最低 L", "close": "收盤 C"}
COLORS = {"actual": "#ce4257", "with_night_gap_only": "#197b9b",
          "without_night": "#9163a8", "night_carry_with_mean_body": "#697585"}
NAVY, TEAL = "#18364b", "#137e82"


def escape(value):
    return html.escape(str(value), quote=True)


def figure_style():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": 0.18, "axes.unicode_minus": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def dates(rows):
    return [datetime.fromisoformat(r["date"]) for r in rows]


def segments(ax, rows, y, market, **kwargs):
    """Keep every point but avoid visually connecting different TX contracts."""
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or (market != "TAIEX" and rows[i]["contract"] != rows[i - 1]["contract"]):
            ax.plot(dates(rows[start:i]), y[start:i], **kwargs)
            kwargs = {**kwargs, "label": "_nolegend_"}
            start = i


def price_panel(ax, data, market, field, context=True):
    rows = data["daily"]["with_night_gap_only"]
    if context:
        past = data["context"]
        segments(ax, past, [r["actual"][field] for r in past], market,
                 color="#b4bcc5", lw=1.3, marker=".", ms=2, label="先前驗證區段的實際值")
        ax.axvspan(dates(past)[0], dates(rows)[0], color="#edf1f5", alpha=0.65)
    for method in ("night_carry_with_mean_body", "with_night_gap_only"):
        arm = data["daily"][method]
        segments(ax, arm, [r["predicted"][field] for r in arm], market,
                 color=COLORS[method], lw=1.25, alpha=0.85, label=NAMES[method])
    segments(ax, rows, [r["actual"][field] for r in rows], market,
             color=COLORS["actual"], lw=1.35, label="實際值")
    ax.axvline(dates(rows)[0], color=NAVY, ls="--", lw=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylabel("指數點數")
    ax.set_title(LABELS[field], loc="left")


def plot_results(markets, figures):
    for market, data in markets.items():
        for field in FIELDS:
            fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 1]})
            price_panel(top, data, market, field)
            top.legend(loc="upper left", fontsize=8, ncol=2)
            for method in ("with_night_gap_only", "night_carry_with_mean_body"):
                rows = data["daily"][method]
                error = [(r["predicted"][field] - r["actual"][field]) / r["previous_close"] * 10000
                         for r in rows]
                segments(bottom, rows, error, market, color=COLORS[method], lw=0.9)
            bottom.axhline(0, color=NAVY, lw=0.7)
            bottom.set_ylabel("帶符號誤差 bp")
            fig.suptitle(f"{MARKETS[market]}｜{LABELS[field]}：預測與實際", x=0.07, ha="left", fontsize=17)
            m = data["metrics"]["with_night_gap_only"]
            caption = (f"已見回溯區間 2026-01～09；n={m['scored']}；Ridge MAE={m[field + '_mae_points']:.2f}點／"
                       f"{m[field + '_mae_bps']:.2f}bp；RMSE={m['rmse_points'][field]:.2f}點。")
            fig.text(0.07, 0.045, caption, fontsize=9)
            fig.text(0.07, 0.012, "虛線為評估起點；灰底是先前驗證資料。非新前向測試；TX 換月處斷線，點位全保留。", fontsize=8)
            fig.tight_layout(rect=(0.02, 0.085, 0.99, 0.94))
            fig.savefig(figures / f"{market}-{field}.png", dpi=160)
            plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for ax, field in zip(axes.flat, FIELDS):
            price_panel(ax, data, market, field, context=False)
            ax.tick_params(axis="x", labelsize=8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle(MARKETS[market] + "｜四個價位分別檢視（179 日已見回溯）", fontsize=16)
        fig.tight_layout(rect=(0, 0.065, 1, 0.95))
        fig.savefig(figures / f"{market}-overview.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for j, method in enumerate(METHODS):
            m = data["metrics"][method]
            ax.bar([i + (j - 1) * .24 for i in range(4)], [m[f + "_mae_bps"] for f in FIELDS],
                   width=.23, color=COLORS[method], label=NAMES[method])
        ax.set_xticks(range(4), [LABELS[f] for f in FIELDS])
        ax.set_ylabel("MAE（bp，越低越好）")
        ax.set_title(MARKETS[market] + "｜同日期、同前收、同定義比較", loc="left")
        ax.legend(fontsize=9)
        fig.text(.08, .015, "每種方法 179／179 日；無失敗子集篩選。這是既有結果整理，未選擇新模型。", fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, 1))
        fig.savefig(figures / f"{market}-errors.png", dpi=160)
        plt.close(fig)


def box(ax, x, y, w, h, text, color, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.014,rounding_size=.02",
                               facecolor=color, edgecolor="#9163a8" if dashed else "white",
                               linestyle="--" if dashed else "-", lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color=NAVY)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                               color="#64788a", lw=1.6, linestyle="--" if dashed else "-",
                               connectionstyle="arc3,rad=0"))


def diagrams(figures):
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    for x, text in ((.02, "歷史日盤／完成夜盤\n來源、同合約前收"),
                    (.345, "清理與時間檢查\n固定模型／基準"),
                    (.67, "08:40 前凍結數字\nOHLC、版本、指紋")):
        box(ax, x, .68, .29, .20, text, "#dfedf5")
    box(ax, .02, .34, .29, .20, "截止前新聞／事件\n僅作背景與風險說明", "#eee6f6")
    box(ax, .345, .34, .29, .20, "晨報／展示介面\n只能引用凍結數字", "#eee6f6")
    box(ax, .67, .34, .29, .20, "收盤後官方實際行情\n同日、同市場、同合約", "#fcebd9")
    box(ax, .345, .025, .29, .20, "逐價位評估與覆盤\nMAE、RMSE、方向、缺漏", "#dff0e9")
    box(ax, .02, .025, .29, .20, "新聞輔助獨立方向 B2\n已備評分介面，待實測", "#eee6f6", dashed=True)
    arrow(ax, (.165, .34), (.165, .225), dashed=True)
    arrow(ax, (.31, .125), (.345, .125), dashed=True)
    for a, b in [((.31, .78), (.345, .78)), ((.635, .78), (.67, .78)),
                 ((.31, .44), (.345, .44)), ((.815, .68), (.49, .54)),
                 ((.49, .34), (.49, .225)), ((.815, .34), (.635, .125))]:
        arrow(ax, a, b)
    fig.suptitle("專案總流程｜量化為基礎，AI 輔助解釋", fontsize=18, x=.06, ha="left")
    fig.text(.06, .01, "實線為既有資料流；虛線為獨立方向比較。新排程、真實盤前保存、新聞增益及 Sheet 讀回待驗證。", fontsize=9)
    fig.savefig(figures / "project-flow.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    for x, text in ((.12, "較長歷史\n模型估計期間"), (.48, "D−1／完成夜盤\n推論用資訊"),
                    (.73, "D日08:30\n資料截止"), (.9, "D日收盤後\n才取得答案")):
        ax.text(x, .90, text, ha="center", va="center", fontsize=11)
    arrow(ax, (.04, .76), (.98, .76))
    box(ax, .04, .53, .60, .13, "量化：估計資料範圍與每次輸入視窗需分開說明", "#dfedf5")
    box(ax, .36, .33, .36, .13, "新聞：較短期間，發布與取得均需核對", "#eee6f6")
    box(ax, .70, .12, .27, .13, "08:40前凍結 → 15:30收盤流程", "#dff0e9")
    ax.plot([.73, .73], [.12, .78], ls="--", color=NAVY)
    fig.suptitle("資訊時間軸｜先有預測，之後才能看到當日答案", fontsize=18, x=.06, ha="left")
    fig.text(.06, .04, "本次圖表：原始訓練至2025/09、驗證2025Q4、已見評估2026/01–09；依預宣告規則每日擴展擬合。", fontsize=10)
    fig.savefig(figures / "information-timeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def tables(evidence):
    price, direction, errors = [], [], []
    for market, d in evidence["markets"].items():
        for method in METHODS:
            m = d["metrics"][method]
            price.append([MARKETS[market], NAMES[method], *[f"{m[f + '_mae_bps']:.2f}" for f in FIELDS],
                          f"{m['scored']}/179"])
            direction.append([MARKETS[market], NAMES[method], f"{m['gap_accuracy']:.2%}",
                              f"{m['close_accuracy']:.2%}", f"{m['always_up_close_accuracy']:.2%}"])
        for r in d["largest_close_errors"]:
            errors.append([MARKETS[market], r["date"], r["contract"] if market != "TAIEX" else "—",
                           f"{r['close_ae_points']:.2f}", f"{r['close_ae_bps']:.2f}", "僅列誤差，不作消息因果歸因"])
    return price, direction, errors


def build_html(evidence, demo, output):
    price, direction, errors = tables(evidence)
    sections = []
    for market, d in evidence["markets"].items():
        figs = "".join(f'<figure><img src="figures/{market}-{f}.png" alt="{escape(MARKETS[market])} {LABELS[f]}預測實際對照"><figcaption>{LABELS[f]}：圖下方為相對同合約／同標的前收的帶符號誤差。</figcaption></figure>' for f in FIELDS)
        details = []
        for method in METHODS:
            m = d["metrics"][method]
            for f in FIELDS:
                details.append([NAMES[method], LABELS[f], f"{m[f + '_mae_points']:.2f}",
                                f"{m['rmse_points'][f]:.2f}", f"{m[f + '_mae_bps']:.2f}",
                                f"{m[f + '_rmse_bps']:.2f}", m["scored"], m["missing_or_failed"]])
        sections.append(f'<section class="market-panel" data-market="{market}"><h2>{MARKETS[market]}｜O/H/L/C 對照</h2><p>179 日已見回溯。灰底為先前 20 個驗證日；不是這次新增訓練或新前向測試。TX 換月時不連接線段，但保留全部點。</p>{figs}<div class="price-details"><h3>完整價格誤差表</h3>{table(["方法","目標","MAE 點","RMSE 點","MAE bp","RMSE bp","n","缺漏"],details)}</div></section>')
    pilot = evidence["inventory"]["timesfm_pilot"]
    demo_record = demo["morning_export"]["records"][0]
    numbers = demo_record["frozen_numerical"]["prices"]
    demo_rows = demo["local_readback_rows"]
    demo_json = json.dumps({"morning": demo_record["values"], "close": demo_rows[0], "prices": numbers},
                           ensure_ascii=False).replace("<", "\\u003c")
    body = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>台股盤前 OHLC 專題｜研究成果與展示</title>
<style>
:root{{--navy:{NAVY};--teal:{TEAL}}}*{{box-sizing:border-box}}body{{margin:0;background:#eff3f6;color:var(--navy);font-family:"Noto Sans CJK TC","Microsoft JhengHei",sans-serif;line-height:1.7}}main{{max-width:1120px;margin:auto;padding:38px 28px}}header{{background:var(--navy);color:white;padding:40px;border-radius:18px}}h1{{font-size:32px;line-height:1.35;margin:8px 0 16px}}h2{{margin-top:0;font-size:25px}}h3{{font-size:19px}}section{{background:white;border-radius:14px;padding:28px;margin:24px 0;overflow-wrap:anywhere}}.eyebrow{{letter-spacing:.12em;color:#a6dfe2;font-size:14px}}.badges{{display:flex;gap:10px;flex-wrap:wrap}}.badge{{background:#dceef0;color:#115c63;padding:6px 12px;border-radius:8px;font-weight:bold}}.warning{{background:#fff1d6;border-left:5px solid #cb8b20;padding:15px 20px;border-radius:7px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.card{{border:1px solid #d7e0e8;border-radius:10px;padding:18px}}figure{{margin:22px 0;break-inside:avoid}}img{{width:100%;height:auto;display:block}}figcaption,.muted{{font-size:13px;color:#596c7b}}table{{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow-x:auto}}th,td{{border-bottom:1px solid #d8e2e8;padding:9px;text-align:left;vertical-align:top}}th{{background:#e7f0f5;white-space:nowrap}}button,select{{font:inherit;border:1px solid #7894a6;border-radius:7px;background:white;padding:9px 14px;cursor:pointer}}button.active{{background:var(--teal);color:white}}.controls{{position:sticky;top:0;z-index:2;background:#edf3f7;padding:12px 0}}.demo-price{{font-size:24px;font-weight:bold;color:var(--teal)}}footer{{font-size:12px;margin:25px}}a{{color:#137b92}}.small{{font-size:13px}}@media(max-width:640px){{main{{padding:16px}}header,section{{padding:20px}}h1{{font-size:26px}}.grid{{grid-template-columns:1fr}}table{{font-size:11px}}}}@media print{{@page{{size:A4;margin:13mm}}body{{background:white;font-size:10pt}}main{{padding:0;max-width:none}}header{{border-radius:0;padding:20px}}section{{padding:12px 0;border-radius:0;break-before:page;margin:0}}section:first-of-type{{break-before:auto}}.controls,.no-print{{display:none!important}}.market-panel[hidden]{{display:block!important}}figure{{break-inside:avoid}}.price-details{{break-inside:avoid}}tr{{break-inside:avoid}}table{{display:table;font-size:8pt}}th,td{{padding:5px}}h1{{font-size:24pt}}h2{{font-size:17pt}}.grid{{gap:12px}}}}
</style><main><header><div class="eyebrow">ACADEMIC RESEARCH · 2026-11-16</div><h1>台股盤前 OHLC 預測<br>量化為基礎，AI 輔助解釋</h1><p>資料可追溯、預測先凍結、收盤再驗證。以完整研究流程與真實限制呈現成果。</p><div class="badges"><span class="badge">TAIEX 179 日</span><span class="badge">TX 179 日</span><span class="badge">三種固定方法</span><span class="badge">本次模型重訓 0 次</span></div></header>
<section><h2>研究結論與證據範圍</h2><div class="warning"><strong>這是已凍結的回溯結果整理，評估標籤先前已看過；歷史發布時點未核實。</strong><br>不是新增獨立測試，不是正式前向準確率，也沒有交易獲利主張。</div><p>只為開盤頭加入夜盤後，相對無夜盤版，TAIEX／TX 收盤 MAE 改善約 22.19%／28.76%；但兩者仍輸給預列的簡單夜盤直推基準約 6.41%／2.85%。所有方法與負面結果都保留，不因這次展示重選模型。</p><p>TAIEX 的開盤幅度沒有因加入夜盤而改善；「某個方向比較準」不能代表每個價位的誤差都較小。四個價位分開呈現，方向與價格距離分開評估。</p></section>
<section><h2>專案總流程</h2><img src="figures/project-flow.png" alt="歷史行情、量化、凍結、AI解釋、官方收盤與驗證流程"><p>量化模型輸出數字，AI 依凍結結果補充背景；新聞不改數字或機率。新流程程式與本地轉接已有測試；新排程、真實盤前遠端保存與 Google Sheet 讀回仍需執行證據。</p><h3>資料時間軸</h3><img src="figures/information-timeline.png" alt="歷史模型資料、短期新聞與盤前資訊截止時間軸"></section>
<section><h2>資料、方法與公平比較</h2><p>正式現貨 TAIEX 與盤前選定月合約 TX 分開。期貨日夜盤分開、使用同合約前日盤最後成交價，排除週別／價差；夜盤按下一一般交易日對齊，保留週末資料年齡。當日日盤 O/H/L/C 只能作事後答案。</p><p>最初訓練至 2025-09-30、驗證 2025Q4、已見評估 2026-01～09；依預宣告規則每日擴展擬合，只使用當日以前共同樣本。此次僅讀取既有輸出與來源核對，沒有任何模型擬合。</p>{table(["方法","開盤頭","盤中頭／影線"],[[NAMES[METHODS[0]],"前日 gap／close 報酬","前日 body／range；過去平均非負影線"],[NAMES[METHODS[1]],"相同特徵＋完成夜盤報酬","與無夜盤版完全相同"],[NAMES[METHODS[2]],"直接使用完成夜盤報酬","過去平均 body 與影線"]])}<h3>模型清冊</h3><p>另找到 TimesFM 的 <strong>output-head-only</strong> 小樣本試驗紀錄：訓練 {pilot.get('training_windows','待核對')} 個視窗、驗證 {pilot.get('validation_windows','待核對')} 個視窗，有梯度與參數更新紀錄；它不是完整 179 日同口徑比較，也不是全模型微調完成。相關衍生權重不隨成果包散布。ARIMA 尚未完成比較，列為後續選做。</p></section>
<section><h2>指標定義</h2><div class="grid"><div class="card"><h3>連續價格誤差</h3><p>MAE = 平均絕對誤差<br>RMSE = 平均平方誤差的平方根<br>O／H／L／C 各自列點數與 bp。</p></div><div class="card"><h3>方向與容許誤差分開</h3><p>相對同標的／同合約前收，±5bp 內為平，其餘為漲／跌。<br>「100 點內」屬容許誤差命中率，不等於方向命中率；本次未事後選擇這個門檻。</p></div></div><p>每市場各方法同為 179／179 日，失敗 0，分母沒有挑選較好子集。真實前向與回溯不得混算；程式測試通過也不代表預測準確。</p><h3>價格 MAE 比較（bp，越低越好）</h3>{table(["市場","方法","O","H","L","C","樣本"],price)}<h3>方向比較（同一 5bp 中性門檻）</h3>{table(["市場","方法","開盤方向命中","收盤方向命中","永遠猜漲：收盤"],direction)}</section>
<div class="controls"><label for="market">圖表市場：</label><select id="market"><option value="TAIEX">TAIEX 加權指數</option><option value="TX_selected_monthly">TX 盤前選月合約</option></select><span class="muted">列印 PDF 包含兩個市場。</span></div>{''.join(sections)}
<section><h2>大誤差案例｜先查資料與單位，再討論原因</h2><p>依「僅開盤加入夜盤」模型的收盤絕對 bp 誤差排序，各市場列前五日。這是事後診斷，不用來重選門檻或證明某事件造成誤差。先核對日期、合約、前收、價格尺度及轉換，再提出待驗證的行情假說。</p>{table(["市場","日期","TX 合約","C 絕對誤差點","C 絕對誤差bp","解讀限制"],errors)}</section>
<section><h2>量化＋AI：如何避免假比較</h2>{table(["比較組","價格 MAE／RMSE","方向／用途"],[["A：固定量化預測","可以比較","以同一中性門檻評估"],["B1：同一組量化數字＋AI解釋","與A完全相同","數字忠實、來源與風險背景；不是新的價格增益"],["B2：量化＋新聞的獨立方向決策","沒有單點價格就不適用","已備獨立決策與同口徑評分介面；尚缺真實盤前新聞決策紀錄"]])}<p>不把 LLM 自述信心當作已校準機率，也不拿原本沒有輸出的區間中點假造價格預測。</p></section>
<section id="synthetic-demo"><h2>完整操作流程示範</h2><div class="warning"><strong>合成資料示範，不是市場預測；沒有寫入 Google Sheet。</strong></div><p>切換盤前／收盤檢視時，原先凍結的數字保持不變；覆盤只加入實際結果與誤差，禁止改寫盤前假設。</p><div class="grid"><div class="card"><h3>凍結量化數字（合成）</h3><div class="demo-price" id="frozen-prices"></div><p>顯示至小數2位；原數值與指紋保持不變。AI 可補充背景，不得改數字。</p></div><div class="card"><div><button id="morning-button" class="active">合成盤前晨報</button> <button id="close-button">合成收盤覆盤</button></div><p id="demo-text"></p></div></div><p class="muted">原程式包含數字與敘事一致、日期／合約、盤前欄位不變與覆盤欄位讀回測試。示範本身不當成新前向績效。</p></section>
<section><h2>產品價值與 SWOT</h2><div class="grid"><div class="card"><h3>優勢</h3>來源可追溯、數字先凍結、逐價位比較、盤後檢討與失敗留存。</div><div class="card"><h3>限制</h3>尚未優於強夜盤基準；歷史發布時點未核實；缺真實新版本全流程證據。</div><div class="card"><h3>機會</h3>作研究與學習工具，將量化、背景資訊及覆盤集中，降低事後改寫假設的空間。</div><div class="card"><h3>風險</h3>資料延遲、行情變化、過度調參、授權限制與使用者把研究數字誤當保證。</div></div></section>
<section><h2>完成度與接續工作</h2><p><strong>已完成本輪：</strong>總流程圖、資訊時間軸、兩市場八張逐價位圖、完整價格／方向比較、誤差案例、模型清冊、簡報與離線操作示範。</p><p><strong>仍需真實驗證：</strong>新流程啟用後的盤前遠端保存、同日兩市場收盤資料、正式或 staging Sheet 寫入與讀回；累積足夠新樣本後才能討論前向準確率與區間覆蓋。</p><p><strong>後續選做：</strong>ARIMA、預先宣告的新聞方向決策、情境分類與模型改善。不能因展示而改寫舊研究，或把已見資料當新測試。</p><h3>重現與來源</h3><p>先執行 prepare_academic_results 驗證來源及既有誤差，再執行 build_academic_bundle。原始行情／每日明細保留 ignored；公開的 price-comparison.csv 只有彙總。各檔案指紋、版本及輸出清冊見 manifest.json。</p></section><footer>2026-11-16｜學術研究成果整理。所有圖與數值來自既有凍結研究的來源核對與重新計算；未呼叫模型、未新增訓練或新聞決策。</footer></main>
<script>const demo={demo_json};const market=document.getElementById('market');function choose(){{document.querySelectorAll('.market-panel').forEach(p=>p.hidden=p.dataset.market!==market.value)}}market.addEventListener('change',choose);choose();document.getElementById('frozen-prices').textContent=Object.entries(demo.prices).map(([k,v])=>k+' '+Number(v).toFixed(2)).join(' ／ ');function stage(close){{document.getElementById('demo-text').textContent=close?demo.close.slice(6,14).join('；'):'量化數字與名目邊際區間已凍結。'+demo.morning[3]+'；'+demo.morning[4];document.getElementById('morning-button').classList.toggle('active',!close);document.getElementById('close-button').classList.toggle('active',close)}}document.getElementById('morning-button').onclick=()=>stage(false);document.getElementById('close-button').onclick=()=>stage(true);stage(false);</script></html>'''
    (output / "report.html").write_text(body, encoding="utf-8")


def slides(evidence, output):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    notes = []

    def text(slide, x, y, w, h, value, size=22, color=NAVY, bold=False):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame; tf.word_wrap = True
        for i, line in enumerate(value.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.name = "Microsoft JhengHei"; p.font.size = Pt(size)
            p.font.bold = bold; p.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
            p.space_after = Pt(10)
        return shape

    def add(title, note, body=None, picture=None, small=None):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)
        text(slide, .65, .35, 12, .7, title, 28, bold=True)
        text(slide, .65, 6.97, 11.5, .32, "已見回溯研究｜非新前向證據｜2026-11-16", 10, color="#62798a")
        text(slide, 12, 6.97, .5, .32, str(len(prs.slides)), 10)
        if picture:
            with Image.open(output / "figures" / picture) as image:
                image_w, image_h = image.size
            scale = min(11.65 / image_w, 5.45 / image_h)
            width, height = image_w * scale, image_h * scale
            slide.shapes.add_picture(str(output / "figures" / picture), Inches(.85), Inches(1.2),
                                     width=Inches(width), height=Inches(height))
        if body:
            text(slide, .8, 1.55, 11.7, 4.9, body, 23)
        if small:
            text(slide, .8, 6.45, 11.7, .4, small, 12)
        slide.notes_slide.notes_text_frame.text = note
        notes.append((title, note))
        return slide

    add("台股盤前 OHLC 預測與 AI 輔助覆盤", "本專題建立盤前資訊到收盤驗證的完整研究流程。今天展示的是既有結果整理，不是新增模型準確率或交易服務。",
        "研究目的：在開盤前形成可驗證的數值預測\n量化模型提供數字，AI 協助整理背景與風險\n盤前先凍結，收盤後以相同定義驗證\n本輪重訓 0 次；兩市場各 179 個共同日期")
    add("研究問題與交付範圍", "先分清模型好壞與產品流程。價格預測是否有效，需要基準與誤差；AI背景是否有用途，需要另外定義可驗證任務。",
        "① O／H／L／C 哪一項誤差較小？\n② 夜盤資訊相對相同模型是否有幫助？\n③ 回歸模型能否超過簡單夜盤基準？\n④ 如何保留盤前假設，完成一致的盤後驗證？\nAI 解釋目前不改價格，不能宣稱價格預測因文字而進步。")
    add("專案總流程", "藍色為量化，紫色為文字輔助，橘色為事後答案，綠色為驗證。數字先凍結，新聞不改它。新流程真實啟用與Sheet讀回仍待驗證。", picture="project-flow.png")
    add("資料與資訊時間軸", "模型估計使用的歷史期間與每次推論的視窗要分開。夜盤和新聞必須在截止前可取得，當日行情只能於事後作答案。", picture="information-timeline.png")
    add("資料前處理與時序隔離", "對期貨先在同合約內算報酬，再按盤前已完成夜盤決定月份。不能用當日日盤量選主力，不能用今天下載時間假裝歷史發布證明。",
        "官方 TAIEX；TX 指定月份、日夜盤分開\n前收採同合約日盤最後成交價，排除結算價\n夜盤對齊下一一般交易日，保留週末資料年齡\n最初訓練至 2025/09；驗證 2025Q4；已見評估 2026/01–09\n每日擴展擬合規則已事先固定；此次只畫既有輸出")
    add("模型清冊：不同證據不能混算", "已找到TimesFM輸出頭的小樣本試驗，應承認它有參數更新紀錄，也應說清楚只有8個訓練與8個驗證視窗。它不能加入目前179日的共同方法排名。",
        "主比較：無夜盤 Ridge／開盤加夜盤 Ridge／夜盤直推基準\nTimesFM：輸出頭小樣本試驗，8 訓練＋8 驗證視窗\nTimesFM 尚無同口徑 179 日結果；不是全模型微調完成\nARIMA：尚未評估，列後續選做\nAI：目前為凍結數字的解釋層，非新的價格模型")
    add("評分定義：三種問題分開", "四個價位要分開評分，不代表一定要拆成四個模型。方向對也可能幅度錯；100點容許誤差不是方向命中率。門檻不依最後測試結果挑選。",
        "價格距離：每個價位各列 MAE／RMSE，點數及 bp\n方向命中：相對前收，±5bp 內為平，其餘漲／跌\n容許誤差：例如 100 點內，必須另說明理由\n分母：每市場各方法 179／179；本研究失敗 0\n歷史回溯、真實前向及工程測試分開報告")
    add("TAIEX：四個價位分開看", "圖中紅線為實際，藍線為僅開盤加夜盤的Ridge，灰線為簡單夜盤基準。價格走勢貼近也可能只是共享上升趨勢，仍要看誤差與相對基準。", picture="TAIEX-overview.png")
    add("TX：同合約計算、換月斷線", "TX以盤前規則選擇月合約，每個報酬都使用該合約自己的前收。換月處不連線，全部179個點仍保留；不是事後調整過的主力連續資料。", picture="TX_selected_monthly-overview.png")
    add("TAIEX：加入夜盤仍未超越簡單基準", "相對無夜盤回歸，收盤MAE改善22.19%；但相對預列簡單夜盤基準仍差6.41%。開盤幅度相對無夜盤還惡化約3.65%，負面結果要保留。", picture="TAIEX-errors.png")
    add("TX：夜盤改善明顯，基準依然更強", "相對無夜盤回歸，收盤MAE改善28.76%，開盤改善63.33%；但收盤相對夜盤直推基準仍差2.85%。研究目標尚未通過。", picture="TX_selected_monthly-errors.png")
    _, direction, _ = tables(evidence)
    slide = add("方向命中率也要比簡單猜測", "方向使用同一個5bp中性門檻及共同日期。不能只報72%的數字，還應把夜盤基準與永遠猜漲一起列出。這不代表交易獲利。")
    shape = slide.shapes.add_table(7, 5, Inches(.65), Inches(1.55), Inches(12), Inches(4.65)).table
    for i, row in enumerate([["市場", "方法", "開盤方向", "收盤方向", "永遠猜漲：收盤"], *direction]):
        for j, value in enumerate(row):
            cell = shape.cell(i, j); cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Microsoft JhengHei"; p.font.size = Pt(12 if j == 1 else 13)
    add("量化＋AI：先定義公平比較", "只改寫文字而不改價格，價格誤差當然與量化相同。若要研究新聞的方向增益，需要另設盤前凍結的方向任務及未見樣本，不能把本輪解釋層當成完成那個比較。",
        "A：量化數字 → 可計算價格與方向\nB1：同一數字＋AI 解釋 → 價格誤差完全相同\nB1 可驗證：數字忠實度、來源追溯、風險說明\nB2：新聞輔助獨立方向決策 → 介面已備，真實比較待執行\n區間不是方向機率，LLM 自述信心也未經校準")
    add("完整操作展示：盤前與收盤不混寫", "開啟離線report.html，先切換兩市場圖表，再到合成示範切換晨報與覆盤。請明說這是合成資料，不是盤前實測或Google Sheet寫入證據。",
        "離線成果頁：市場切換、四價曲線、分項誤差及來源\n合成示範：凍結數值 → 晨報 → 收盤結果與覆盤\n切換前後，原 OHLC 與盤前欄位保持不變\n正式或 staging Sheet 的真實寫入／讀回仍待驗證\n不用合成示範充當預測績效")
    add("產品定位與 SWOT", "價值在於把可追溯的數字、背景與覆盤集中。不能在尚未證明時宣傳超額報酬或穩定高準確率。產品與研究限制都要一起呈現。",
        "使用者：需要整理盤前假設與盤後檢討的研究者／學習者\n優勢：來源追溯、數字凍結、分項評估、失敗留存\n弱點：未超過強基準，尚缺完整真實前向證據\n機會：量化與文字一致的研究工作台\n風險：資料延遲、行情變化、過度調參與授權限制")
    add("研究結論與後續工作", "本輪完成教授要求的流程說明、分目標圖表與比較及產品示範。準確率未達標不是要隱藏的結果。下一步先完成真實啟用與前向驗證，再研究ARIMA或新聞方向策略。",
        "夜盤對回歸有幫助，但簡單夜盤基準仍較強\n完成：流程圖、時間軸、八張價位圖、比較表與展示包\n尚待：真實盤前遠端保存 → 同日收盤 → Sheet 讀回\n之後才研究：ARIMA、新聞方向決策、情境分類與模型改善\n不重用已看過的評估資料宣稱新樣本外進步")
    prs.save(output / "presentation.pptx")
    script = "# 專題簡報講稿\n\n每頁約 30–45 秒，依實際報告時間刪減。圖表為已見回溯，示範為合成資料。\n\n"
    for i, (title, note) in enumerate(notes, 1):
        script += f"## {i}. {title}\n\n{note}\n\n"
    (output / "speaker-notes.md").write_text(script, encoding="utf-8")


def main(evidence_path, demo_path, output):
    if output.exists():
        raise FileExistsError("use a fresh output directory")
    evidence = json.loads(evidence_path.read_text())
    demo = json.loads(demo_path.read_text())
    if evidence["model_refit_performed"] or evidence["new_independent_test"]:
        raise ValueError("renderer only accepts the frozen retrospective presentation scope")
    if demo.get("fixture_kind") != "SYNTHETIC_INTEGRATION_ONLY":
        raise ValueError("demo must explicitly be synthetic")
    output.mkdir(parents=True)
    figures = output / "figures"; figures.mkdir()
    figure_style(); diagrams(figures); plot_results(evidence["markets"], figures)
    build_html(evidence, demo, output); slides(evidence, output)
    (output / "synthetic-demo.json").write_text(demo_path.read_text(), encoding="utf-8")
    (output / "README.md").write_text(
        "# 專題成果包\n\n先開 report.html（離線、可切換市場及合成晨報／覆盤）。\n"
        "presentation.pptx 為16頁可編輯簡報；speaker-notes.md 為逐頁講稿。\n"
        "figures/ 含兩張流程／時間圖、八張逐價位圖、兩張四價總覽及兩張誤差比較圖。\n"
        "數值來自既有179日已見回溯，不是新增前向測試；沒有模型重訓。\n"
        "TimesFM小樣本輸出頭試驗另列，沒有混入主比較，也沒有散布衍生權重。\n"
        "合成展示不代表Google Sheet實際寫入，新聞解釋也不代表新的價格增益。\n",
        encoding="utf-8")
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(), "python": platform.python_version(),
        "packages": {n: version(n) for n in ("matplotlib", "python-pptx")},
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "demo_sha256": hashlib.sha256(demo_path.read_bytes()).hexdigest(),
        "model_refit_performed": False, "new_independent_test": False,
        "full_daily_json_exported": False, "derivative_model_weights_exported": False,
        "files_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "figures": len(list(figures.glob("*.png"))),
                      "slides": 16, "model_fits": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--demo", type=Path, default=Path("examples/synthetic-shadow-bridge.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.evidence, args.demo, args.output)

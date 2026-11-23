"""Render TimesFM comparison, threshold sensitivity and audited error cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from scripts.build_academic_bundle import figure_style, segments

FIELDS = ("open", "high", "low", "close")
TFM = ("timesfm3_price_only", "timesfm3_night_residual")
LABELS = {"open": "開盤 O", "high": "最高 H", "low": "最低 L", "close": "收盤 C"}
NAMES = {TFM[0]: "TimesFM3：僅歷史價格", TFM[1]: "TimesFM3：夜盤殘差",
         "without_night": "Ridge：無夜盤", "with_night_gap_only": "Ridge：開盤加夜盤",
         "night_carry_with_mean_body": "簡單夜盤直推基準"}
COLORS = {TFM[0]: "#9163a8", TFM[1]: "#197b9b", "actual": "#ce4257", "baseline": "#697585"}


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def main(tfm_directory, diagnostics, frozen_directory, output):
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"; figures.mkdir()
    summary = json.loads((tfm_directory / "summary.json").read_text())
    daily = json.loads((tfm_directory / "daily-private.json").read_text())
    old = json.loads((frozen_directory / "daily-private.json").read_text())
    audit = json.loads(diagnostics.read_text())
    figure_style()
    comparison = []
    chart_sections = []
    for market, arms in summary["markets"].items():
        data = daily[market]
        baseline = old[market]["evaluation_seen"]["night_carry_with_mean_body"]
        dates = [r["date"] for r in baseline]
        for method in TFM:
            if [r["date"] for r in data[method]] != dates or not arms[method]["complete"]:
                raise ValueError("refuse incomplete or unmatched TimesFM comparison")
            if any(r["actual"] != base["actual"] for r, base in zip(data[method], baseline)):
                raise ValueError("TimesFM and baseline actuals differ")
        metrics = {**arms[TFM[0]]["baseline_metrics"], **{m: arms[m]["metrics"] for m in TFM}}
        for method, m in metrics.items():
            comparison.append([market, NAMES[method], *[round(m[f + "_mae_bps"], 3) for f in FIELDS],
                               f"{m['close_accuracy']:.2%}", m["scored"]])
        for field in FIELDS:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
            for method in TFM:
                rows = data[method]
                segments(axes[0], rows, [r["predicted"][field] for r in rows], market,
                         color=COLORS[method], lw=1.2, label=NAMES[method])
                errors = [(r["predicted"][field] - r["actual"][field]) / r["previous_close"] * 10000
                          for r in rows]
                segments(axes[1], rows, errors, market, color=COLORS[method], lw=.85)
            segments(axes[0], baseline, [r["predicted"][field] for r in baseline], market,
                     color=COLORS["baseline"], lw=1.1, label="夜盤基準")
            segments(axes[0], baseline, [r["actual"][field] for r in baseline], market,
                     color=COLORS["actual"], lw=1.1, label="實際值")
            axes[0].set(ylabel="指數點數"); axes[0].legend(fontsize=9, ncol=2)
            axes[1].set(ylabel="帶符號誤差 bp"); axes[1].axhline(0, color="#18364b", lw=.7)
            axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            fig.suptitle(f"{market}｜{LABELS[field]}｜TimesFM3固定設定，179日已見回溯", fontsize=16)
            fig.text(.07, .015, "64點context、horizon=1、官方權重固定；零參數更新。TX換月處斷線。歷史可用時間／預訓練重疊未驗證。", fontsize=9)
            fig.tight_layout(rect=(0, .045, 1, .95))
            fig.savefig(figures / f"{market}-{field}.png", dpi=150); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for ax, field in zip(axes.flat, FIELDS):
            for method in TFM:
                segments(ax, data[method], [r["predicted"][field] for r in data[method]], market,
                         color=COLORS[method], lw=1.0, label=NAMES[method])
            segments(ax, baseline, [r["actual"][field] for r in baseline], market,
                     color=COLORS["actual"], lw=1.0, label="實際值")
            ax.set_title(LABELS[field], loc="left")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{market}｜TimesFM3四價總覽，179日已見回溯", fontsize=16)
        fig.tight_layout(rect=(0, .06, 1, .95))
        fig.savefig(figures / f"{market}-overview.png", dpi=150); plt.close(fig)
        chart_sections.append(f'<section><h2>{market}：四價分項</h2>' + ''.join(
            f'<figure><img src="figures/{market}-{f}.png"><figcaption>{LABELS[f]}；下方是帶符號價格誤差，不是方向命中率。</figcaption></figure>'
            for f in FIELDS) + '</section>')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, market in zip(axes, daily):
        for phase, color, label in (("validation", "#9163a8", "驗證期"), ("evaluation_seen", "#197b9b", "已見評估期")):
            rows = [r for r in audit["direction_sensitivity"] if r["market"] == market and r["phase"] == phase
                    and r["method"] == "with_night_gap_only"]
            ax.plot([r["threshold_bps"] for r in rows], [100 * r["close"]["direction_accuracy"] for r in rows],
                    marker="o", color=color, label=label)
        ax.axvline(5, ls="--", color="#18364b"); ax.set(title=market, xlabel="中性門檻 ±bp（固定網格，不選最佳）")
        ax.legend()
    axes[0].set_ylabel("收盤方向命中率 %")
    fig.suptitle("門檻敏感度｜原開盤加夜盤 Ridge，維持原5bp政策", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(figures / "threshold-sensitivity.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    cases = audit["audited_error_cases"]
    y = np.arange(len(cases))
    ax.barh(y - .17, [c["gap_error_log_bp"] for c in cases], height=.33, color="#9163a8", label="開盤gap誤差")
    ax.barh(y + .17, [c["body_error_log_bp"] for c in cases], height=.33, color="#197b9b", label="盤中body誤差")
    ax.set_yticks(y, [c["market"].replace("_selected_monthly", "") + " " + c["date"] for c in cases])
    ax.axvline(0, color="#18364b", lw=.7); ax.set_xlabel("精確log誤差 × 10000（兩項相加為收盤log誤差，非事件因果）")
    ax.legend(); ax.set_title("十個大誤差案例：九個以body誤差較大", loc="left")
    fig.tight_layout(); fig.savefig(figures / "error-decomposition.png", dpi=150); plt.close(fig)
    case_rows = [[c["market"], c["date"], round(c["close_absolute_error_points"], 2),
                  round(c["close_absolute_error_bps"], 2), c["dominant_component"],
                  "全部通過" if all(c["checks"].values()) else "有缺漏"] for c in cases]
    direction_rows = [[r["market"], r["phase"], r["threshold_bps"], f"{r['close']['direction_accuracy']:.2%}",
                       f"{r['close']['actual_flat_fraction']:.2%}"] for r in audit["direction_sensitivity"]
                      if r["method"] == "with_night_gap_only"]
    tolerance_rows = [[r["market"], r["phase"], f"{r['hit_fraction']:.2%}"] for r in audit["tolerance_hit_rates"]
                      if r["method"] == "with_night_gap_only" and r["field"] == "close"
                      and r["unit"] == "points" and r["threshold"] == 100]
    body = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>研究缺口推進報告</title><style>body{font-family:"Noto Sans CJK TC",sans-serif;color:#18364b;background:#f1f5f8;margin:0;line-height:1.7}main{max-width:1080px;margin:auto;padding:30px}section{background:white;padding:28px;margin:22px 0;border-radius:12px}h1{font-size:30px}h2{font-size:23px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border-bottom:1px solid #d8e3eb;text-align:left}th{background:#e9f2f6}img{width:100%}figure{margin:18px 0;break-inside:avoid}figcaption{font-size:12px;color:#62798a}.warning{background:#fff1d6;padding:15px;border-left:5px solid #d29830}@media print{@page{size:A4;margin:13mm}body{background:white;font-size:10pt}main{padding:0}section{break-before:page;margin:0;padding:0}section:first-child{break-before:auto}table{font-size:8pt}td,th{padding:5px}tr{break-inside:avoid}figure{break-inside:avoid}}</style><main>'
    body += '<section><h1>教授要求缺口｜第二輪實證與流程推進</h1><p>2026-11-16，Asia/Taipei</p><div class="warning">固定模型版本與設定；兩市場各179日，兩種TimesFM3變體完整。這是已見歷史研究，預訓練重疊與歷史發布時間未核實，不是新增前向準確率。</div><p>本輪完成：TimesFM完整分項比較、門檻理由與敏感度、十個大誤差來源／模型重現查核，以及獨立新聞研究的遠端時鐘收據與有限期試跑準備。真實新聞增益需要未來交易日的決策與actual，不能今天倒填。</p><h2>主要結果</h2>'
    body += table(["市場", "方法", "O MAE bp", "H MAE bp", "L MAE bp", "C MAE bp", "C方向", "n"], comparison)
    body += '<p>TimesFM夜盤殘差版收盤MAE為106.39／108.39bp，仍比簡單夜盤基準差3.03%／1.60%；TAIEX開盤誤差也高。沒有依結果改視窗、模型或門檻。輸出頭小試驗仍另列，這次是官方權重zero-shot，不是擴大微調。</p></section>'
    body += '<section><h2>TimesFM方法、版本與限制</h2><p>固定官方google/timesfm-3.0-pytorch，revision 43046b85ec22d584a13f8098c2ed39c889e129c2；context=64、horizon=1、CPU4執行緒、use_znorm=True，僅用各次過去context。四條輸入為各價位相對同合約／標的前收的log比率。價格版只用歷史；夜盤版先扣除每個歷史日的夜盤log報酬，再把當次已完成夜盤加回。</p><p>以exp還原保持正值，再用H=max(rawH,O,C)、L=min(rawL,O,C)投影；投影次數保留於JSON。沒有更新參數、搜尋超參數或把量化區間當方向機率。不同資訊集合分開標示，不能宣稱這是只改演算法的受控比較。</p>'
    body += f'<p>實際執行 {summary["runtime_seconds"]:.1f} 秒，尖峰RSS {summary["peak_rss_mib"]:.1f} MiB；沒有購買運算。權重與衍生checkpoint不隨報告散布。</p></section>'
    body += ''.join(chart_sections)
    body += '<section><h2>門檻依據：定義、尺度與敏感度</h2><p>5bp是既有研究的中性帶，用來避免把極小漲跌直接當方向；bp隨指數尺度正規化。它是可重現的研究定義，不是已證明的經濟最適值，也不是交易成本模型。本輪事先固定0、2、5、10、20bp網格作描述性檢查，驗證與已見評估分開，沒有選出新的最佳門檻。</p><img src="figures/threshold-sensitivity.png">'
    body += table(["市場", "期間", "門檻bp", "收盤方向命中", "實際平盤比例"], direction_rows)
    body += '<h2>100點內命中與方向正確是兩件事</h2>' + table(["市場", "期間", "收盤誤差≤100點命中率"], tolerance_rows)
    body += '<p>已見評估中，Ridge的5bp方向命中約72%，但100點內命中僅約17–18%。不能把兩者都叫同一個正確率。門檻敏感性呈現完整網格，不用本輪結果重選政策。</p></section>'
    body += '<section><h2>大誤差查核：先排除可檢查的工程問題</h2>' + table(["市場", "日期", "C誤差點", "C誤差bp", "較大誤差段", "來源／模型檢查"], case_rows)
    body += '<img src="figures/error-decomposition.png"><p>十案均核對官方日盤OHLC、自己的前收／同合約、日期與lag、原alpha=0.1模型重算及價格重建；10次重現相當於20個原Ridge head擬合核對，沒有新參數搜尋。九案body誤差較大；這是精確代數分解，不能因此宣稱已找到行情或消息原因。歷史發布時點仍未證明，其他未檢查問題不因此被全面排除。</p></section>'
    body += '<section><h2>新聞配對與真實流程：啟動條件及證據</h2><p>已建立獨立研究ledger Issue #4，留言寫入／讀回的實際測試成功。盤前payload包含原數字、獨立新聞方向、指令與程式版本；GitHub留言created_at／updated_at及內容hash證明遠端保存時間。新留言不改舊留言；逾時、缺新聞、缺actual與重複均保留。預計11/17–11/30以10個工作日試跑，08:15盤前、15:30收盤；是否已啟用以trial-status.json的真實工具回讀為準。</p><p>原量化基準固定night_carry，新聞最近72小時、最多5則，資料08:30截止、08:40前另存方向；模型精確後端若未公開就明列。真實配對目前仍為0，10日也不保證得到改善結論；少於30筆不給穩健區間。原V0／V1排程與正式Sheet不切換。</p><p>下午的GitHub收據測試不是盤前預測。後續有效樣本必須先有真實盤前留言，再有同日官方actual與對應覆盤；不得事後補做。</p></section></main></html>'
    (output / "report.html").write_text(body, encoding="utf-8")
    with (output / "timesfm-comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["market", "method", "O_MAE_bp", "H_MAE_bp", "L_MAE_bp", "C_MAE_bp", "C_direction", "n"])
        writer.writerows(comparison)
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    def slide(title, lines=None, image=None, note=""):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        def text(x,y,w,h,value,size):
            frame=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame
            frame.word_wrap=True
            for i,line in enumerate(value.split("\n")):
                p=frame.paragraphs[0] if i==0 else frame.add_paragraph();p.text=line
                p.font.name="Microsoft JhengHei";p.font.size=Pt(size);p.font.color.rgb=RGBColor(24,54,75)
                p.space_after=Pt(12)
        text(.6,.35,12,.8,title,28)
        if lines:text(.8,1.6,11.7,4.8,lines,22)
        if image:s.shapes.add_picture(str(figures/image),Inches(.8),Inches(1.25),width=Inches(11.7),height=Inches(5.3))
        text(.7,7.0,11.8,.3,"已見回溯；新聞真實增益待樣本｜2026-11-16",10)
        s.notes_slide.notes_text_frame.text=note or title
    slide("四項缺口：這一輪推進到哪裡", "TimesFM：完成兩市場各179日兩變體比較\n門檻：理由、尺度與完整敏感度網格\n誤差：十案官方來源／同合約／模型重算通過\n新聞：獨立遠端收據與10日試跑流程\n真實增益仍需要未來盤前決策與同日actual",note="這輪增加實際研究證據，而非重述計畫。新聞資料不能事後生成成有效盤前預測。")
    slide("TimesFM設定固定，沒有重新微調", "官方權重與revision固定；64點context、一步預測\n歷史價格版／夜盤殘差版分開\n兩市場、兩方法均179日，無失敗子集篩選\n本次零參數更新；不混稱先前8個驗證視窗小試驗\n預訓練重疊未知、歷史發布未核實，僅回溯結論")
    slide("TAIEX：完整分價位結果",image="TAIEX-overview.png",note="夜盤殘差收盤106.39bp仍輸簡單基準3.03%；開盤也有很大差異，不能只報收盤改善。")
    slide("TX：完整分價位結果",image="TX_selected_monthly-overview.png",note="夜盤殘差收盤108.39bp仍輸簡單基準1.60%。換月處斷線，各報酬使用自己合約的前收。")
    slide("門檻不是自然常數：敏感性完整呈現",image="threshold-sensitivity.png",note="維持5bp，不依這一輪結果選最佳；100點容許誤差與方向命中分開。")
    slide("十個大誤差：九個body誤差較大",image="error-decomposition.png",note="來源與原模型重現都通過。這是代數上的誤差位置，不能當作消息或行情的因果證明。")
    slide("新聞與量化的獨立研究試跑", "固定原數字，只另存新聞收盤方向\n08:30前新聞，08:40前遠端留言保存\nGitHub伺服器時間與精確內容回讀，hash單獨不夠\n缺新聞、延遲、缺actual與重複全部保留\n不切換原排程或Sheet；目前真實配對0筆")
    slide("接續驗收：實際執行比宣稱完成更重要", "以trial-status.json確認新任務是否啟用\n先取得真正盤前收據，再收盤驗證／累計\n10日先驗證流程與描述性結果，不保證改善\n保留負面結果與新樣本；不重調已見評估\nTimesFM全模型微調、ARIMA與事件因果另列後續")
    prs.save(output / "followup-presentation.pptx")
    manifest = {"source_plan_sha256": hashlib.sha256((tfm_directory/"plan.json").read_bytes()).hexdigest(),
                "source_summary_sha256": hashlib.sha256((tfm_directory/"summary.json").read_bytes()).hexdigest(),
                "diagnostic_sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
                "files_sha256": {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in output.rglob('*') if p.is_file()},
                "model_weight_exported":False,"training_performed_by_renderer":False}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({"figures":len(list(figures.glob('*.png'))),"slides":len(prs.slides),"output":str(output)}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timesfm',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--frozen',type=Path,default=Path('reports/generated/night-ablation-v1-verified'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.timesfm,a.diagnostics,a.frozen,a.output)

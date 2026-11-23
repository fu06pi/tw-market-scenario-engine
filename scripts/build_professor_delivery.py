"""Build a source-backed professor checklist and consolidated research report.

Reads frozen summaries and GitHub API readbacks only; never trains or predicts.
Raw comments remain private. Public output contains aggregate results and links.
"""

import argparse
import csv
import hashlib
import html
import json
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

from tw_scenario.forward_trial import (
    MARKER,
    aggregate_trial_comments,
    assess_comment,
    parse_comment,
)
from tw_scenario.shadow import now

COMMIT = "398d3b7b35f8d007178037ccbc8e06b6ee5d19a6"
PROTOCOL = "09a81ae11e23a79f0a59081f92b62091d0890662e6d6ac4a1363b3d0c449c44c"
REPO = "fu06pi/tw-market-scenario-engine"
REPORT_DATE = "2026-11-23"
LABELS = {"open": "開盤", "high": "最高", "low": "最低", "close": "收盤"}
METHODS = {"without_night": "Ridge：無夜盤", "with_night_gap_only": "Ridge：開盤加夜盤",
           "night_carry_with_mean_body": "簡單夜盤基準"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    cell = lambda x: html.escape(str(x))
    return "<table><thead><tr>" + "".join(f"<th>{cell(h)}</th>" for h in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in r) +
                                         "</tr>" for r in rows) + "</tbody></table>"


def main(comments_path, latest_failure, assets, output):
    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    protocol = root / "config/forward_news_trial_v1.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL
    config = read(protocol)
    for key, digest in [("checkpoint", "checkpoint_sha256"), ("news_prompt", "news_prompt_sha256")]:
        assert hashlib.sha256((root / config[key]).read_bytes()).hexdigest() == config[digest]
    comments = read(comments_path)
    receipts = []
    for metadata in comments:
        if metadata.get("body", "").startswith(MARKER):
            payload = parse_comment(metadata["body"])
            assess_comment(metadata, REPO, 4, PROTOCOL, COMMIT)
            receipts.append((metadata, payload))
    aggregates = [(m, p) for m, p in receipts if p.get("kind") == "aggregate"]
    last_meta, last = max(aggregates, key=lambda pair: pair[0]["created_at"])
    assert last_meta["created_at"] == last_meta["updated_at"]
    last_date = date.fromisoformat(last["target_date"])
    first = date.fromisoformat(config["start_date"])
    expected = [(first + timedelta(days=i)).isoformat() for i in range((last_date-first).days+1)
                if (first + timedelta(days=i)).weekday() < 5]
    cumulative = aggregate_trial_comments(comments, REPO, 4, PROTOCOL, COMMIT, expected)
    assert cumulative == last["comparison"], "Replay differs from published cumulative result"
    late_meta = read(latest_failure)
    late = parse_comment(late_meta["body"])
    assess_comment(late_meta, REPO, 4, PROTOCOL, COMMIT)
    assert late["kind"] == "late_failure" and late["target_date"] == REPORT_DATE
    assert late_meta["created_at"] == late_meta["updated_at"]
    rows = list(csv.DictReader((root / "reports/academic/2026-11-16/price-comparison.csv")
                              .read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 24 and all(r["n"] == "179" and r["failed_or_missing"] == "0" for r in rows)
    timesfm = read(root / "reports/followup/2026-11-16/timesfm3-summary.json")
    assert timesfm["parameter_updates"] == 0 and not timesfm["new_independent_test"]
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    for market in ("TAIEX", "TX_selected_monthly"):
        for field in LABELS:
            source = assets / "figures" / f"{market}-{field}.png"
            shutil.copy2(source, output / "figures" / source.name)
    for name in ("project-flow.png", "information-timeline.png"):
        shutil.copy2(assets / "figures" / name, output / "figures" / name)
    shutil.copy2(root / "reports/academic/2026-11-16/price-comparison.csv", output / "price-comparison.csv")
    (output / "evidence").mkdir()
    for name in ("timesfm3-summary.json", "direction-sensitivity.csv", "diagnostic-plan.json"):
        shutil.copy2(root / "reports/followup/2026-11-16" / name, output / "evidence" / name)
    summary = {"as_of": now().isoformat(), "last_completed_close_date": last["target_date"],
               "last_close_aggregate_url": last_meta["html_url"], "replay_matches_published": True,
               "comparison": cumulative, "today_preopen": {"date": late["target_date"],
               "status": "late_failure", "receipt_url": late_meta["html_url"],
               "server_created_at": late_meta["created_at"], "new_valid_forecasts": 0},
               "model_fits_this_delivery": 0, "real_sheet_write": False,
               "trial_code_or_protocol_changed": False, "PR_modified": False}
    (output / "evidence/forward-status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    checklist = [
        ("專題整體流程圖", "完成", "資料→檢查→量化凍結→新聞→晨報→官方actual→驗證；圖與方法說明"),
        ("資料與新聞時間範圍", "完成", "2025Q4驗證、2026/01–09回溯、64點context、72h新聞窗與08:30／08:40"),
        ("OHLC各項MAE／RMSE", "完成", "兩市場×三方法×四價位，24列點數與bp表；TimesFM另列"),
        ("預測與實際比較圖", "完成", "兩市場各四張；左側2025/12背景為驗證期，不能稱新訓練"),
        ("純量化 vs 量化＋新聞", "部分完成", "共同收盤方向指標及實際配對已完成；每市場僅1筆，增益無法定論"),
        ("評估門檻理由", "完成", "100點為描述性容許值、非最佳標準；±5bp方向分列並做固定網格敏感度"),
        ("研究報告", "完成", "本報告整合動機、資料、模型、驗證、失敗、結論與限制"),
        ("系統展示", "完成展示素材", "既有離線互動晨報覆盤及16頁簡報；正式Sheet整合未完成"),
        ("產品定位／使用情境／SWOT", "完成", "學術研究、學生與自主盤前覆盤；價值與限制同列"),
        ("ARIMA對照", "未完成／選做", "待核心前向證據足夠後，相同日期及資訊集合比較"),
        ("策略切換與進一步調參", "未執行／後續", "不使用已見評估挑參數；先處理解析可靠性與累積樣本"),
        ("美化流程與展示", "已有基本版／後續", "保留可讀圖表與離線介面，不延誤實證工作"),
    ]
    pages = []
    def add(title, content):
        pages.append(f'<section><h1>{html.escape(title)}</h1>{content}</section>')
    add("盤前台股輔助判斷系統｜研究報告與驗收", f'<p class="eyebrow">資料截點 {REPORT_DATE} · Asia/Taipei</p>'
        '<p class="lead">以量化模型預測大盤 OHLC，再以盤前新聞形成獨立方向判斷，收盤後比較加入新聞是否有幫助。</p>'
        '<div class="note">歷史模型尚未勝過強夜盤基準；新聞每市場僅1筆真實配對，無法證明改善。介面展示完成不等於正式系統遷移完成。</div>'
        + table(["要求", "狀態", "證據／邊界"], checklist))
    add("研究動機、目的與整體流程", '<p>盤前資訊分散，價格預測與文字解釋容易混在一起，也容易在收盤後改寫原始判斷。本系統將可追溯資料、固定量化數字、新聞證據與盤後評分集中，提供可重現的研究及學習流程。</p>'
        '<p>研究問題一：純量化OHLC各項誤差如何？問題二：同一盤前條件下，新聞是否改善收盤方向？問題三：流程能否保存當時真正可取得的內容？</p>'
        '<img src="figures/project-flow.png" alt="資料到量化、新聞、收盤驗證的整體流程圖">'
        '<p class="caption">流程圖為11/16設計版本；圖中新聞研究進度以本報告最新實際紀錄為準。未以此圖宣稱Sheet已整合。</p>')
    add("資料區間與盤前時間限制", '<img src="figures/information-timeline.png" alt="歷史資料與盤前截止時間軸">'
        + table(["資料／階段", "時間與口徑"], [
            ["歷史來源", "官方期交所TX月CSV範圍2024/12～2026/09；現貨依原研究共同資料覆蓋，並非宣稱2018起已完成"],
            ["初始訓練／驗證", "訓練截至2025/09/30；驗證2025/10/01～12/31；Ridge預宣告每日擴展，只用當時以前資料"],
            ["歷史評估", "2026/01/02～09/30各179日，標籤已見；不是新的盲測"],
            ["TimesFM", "固定官方3.0權重；每次64點過去context、預測1日；與長期權重預訓練範圍不同"],
            ["當日D量化輸入", "D以前日盤及已完成夜盤；期貨使用同一被選月份合約自己的前收"],
            ["新聞", "決策前72小時內、最多5則；發布與真實取得均不晚於D日08:30"],
            ["保存／驗證", "08:40前遠端保存精確payload；收盤後官方同日同合約actual；延遲及失敗保留"]]))
    add("方法與評分標準", table(["方法", "輸出／適用指標"], [
        ["Ridge無夜盤／開盤加夜盤", "相同共同日期與過去資料；只有開盤頭增加完成夜盤特徵"],
        ["簡單夜盤基準", "夜盤直推開盤，過去平均盤中body與影線；保持基準作研究預設"],
        ["TimesFM兩種zero-shot", "歷史價格與夜盤殘差兩版本；716次預測，參數更新0；非全模型微調"],
        ["A純量化／B新聞方向", "A價格及方向，B只輸出漲／平／跌；只以同日收盤方向共同評比"],
        ["B1僅新聞解釋", "數字相同，價格MAE必相同；只能驗證數字忠實與來源"]])
        + '<p>MAE = 平均 |預測−實際|；RMSE = √平均(預測−實際)²。bp誤差以同標的／合約前收正規化；四價分開列出。</p>'
        '<p>100點內命中衡量價格容許誤差，目前無證據證明100點為經濟最佳門檻，因此只作描述性輔助，與25／50／200點等固定網格一起看。方向則以相對前收±5bp為中性帶，避免把極小變動當漲跌；5bp同樣不是已證明最佳值或交易成本模型。</p>'
        '<p>原夜盤Ridge已見評估收盤方向命中為72.63%／72.07%，但價格誤差≤100點僅16.76%／18.44%。兩者分母各179日、指標不同；不能把方向命中稱為價格準確率。門檻敏感度保持原政策，不重選贏家。</p>')
    for market in ("TAIEX", "TX_selected_monthly"):
        subset = [r for r in rows if r["market"] == market]
        add(f"{market}｜OHLC分項歷史誤差", '<p>各方法共同179日；以下為原凍結夜盤研究，價格單位為點，沒有新增擬合。</p>'
            + table(["方法", "價位", "MAE點", "RMSE點", "MAE bp", "n"], [
                [METHODS[r["method"]], LABELS[r["field"]], f'{float(r["mae_points"]):.2f}',
                 f'{float(r["rmse_points"]):.2f}', f'{float(r["mae_bps"]):.2f}', r["n"]] for r in subset])
            + '<p>本表三個方法皆為開盤MAE最小。原夜盤Ridge：TAIEX開盤154.83點、TX開盤183.73點；其他價位誤差更大。這不等於每個模型都最擅長開盤，也不能用四價平均掩蓋高低收誤差。</p>')
    for field, label in LABELS.items():
        add(f"{label}｜預測與實際值比較", ''.join(
            f'<figure><img src="figures/{m}-{field}.png" alt="{m} {label}歷史背景、預測及實際值曲線"><figcaption>{m}：179個已見評估日；左側20個2025/12背景點為驗證期，TX換月處不連線但不刪日期。</figcaption></figure>'
            for m in ("TAIEX", "TX_selected_monthly")))
    fm_rows = []
    for market, methods in timesfm["markets"].items():
        for method, entry in methods.items():
            metrics = entry["metrics"]
            assert metrics["scored"] == 179
            fm_rows.append([market, "歷史價格" if method.endswith("price_only") else "夜盤殘差",
                            *[f'{metrics[f+"_mae_bps"]:.2f}' for f in LABELS], "179"])
    add("TimesFM比較與負面結果", table(["市場", "TimesFM", "O MAE bp", "H", "L", "C", "n"], fm_rows)
        + '<p>夜盤殘差版收盤MAE仍比預列簡單夜盤基準差3.03%（TAIEX）／1.60%（TX），沒有達成原價格改善5%門檻；TAIEX以最高價MAE最小（74.92bp），開盤為86.25bp。因此「開盤最好」只在指定方法內成立，不能外推到所有模型。依既定規則保留最強基準。</p>'
        '<p>先前8訓練／8驗證視窗的輸出頭小試驗有參數更新；本次179日比較是zero-shot，兩者不混算。預訓練重疊未知，歷史發布時點未核實；不同資訊集合的版本也不能視為只改演算法的實驗。</p>'
        '<p>十個最大誤差案例的官方日盤、同合約前收、時間lag與原模型重現通過；9案主要誤差位於盤中body。此為代數歸因，不證明新聞或波動是原因。原Ridge查核20個head擬合，與TimesFM比較更新0次為不同範圍。</p>')
    group_rows = [[g["market"], g["paired_n"], f'{g["a_direction_accuracy"]:.0%}',
                   f'{g["b_direction_accuracy"]:.0%}', f'{g["b_minus_a_accuracy"]:+.0%}']
                  for g in cumulative["groups"]]
    add("量化與新聞：真實配對結果及缺漏", '<p>共同指標為同日收盤相對前收方向，±5bp中性帶；固定checkpoint、prompt與版本分組。B沒有新價格，所以不計B價格MAE。新聞與AI後端版本限制隨決策留存。</p>'
        + table(["市場", "配對n", "A方向命中", "B方向命中", "B−A百分點"], group_rows)
        + f'<p>最新完成收盤為 {last["target_date"]}：應執行{cumulative["expected_scheduled_days"]}日、應配對{cumulative["expected_scheduled_pairs"]}筆，有效{cumulative["paired_n"]}筆，流程覆蓋率{cumulative["overall_scheduled_pair_coverage"]:.0%}。僅11/18有完整配對：現貨實漲、A/B皆判跌；TX202611實跌、A/B皆判跌。</p>'
        '<p>每市場僅1筆，且同日兩市場相關，不能當兩個獨立交易日。尚未達每市場30筆，不給穩健改善區間，不能宣稱新聞有效或無效。</p>'
        '<p>11/17逾時；11/19、11/20因未選遠月合約的H/L缺值使固定解析器失敗，沒有數字快照。這些日子保留於分母；缺值失敗不當成預測錯誤或零誤差。</p>'
        f'<p>今天 {late["target_date"]} 首次查核已13:19:57，盤前逾時已留存；本報告沒有提前執行今日收盤，也沒有以舊日期代替。今日狀態不混入上列最新已完成收盤統計。</p>'
        f'<p><a href="{last_meta["html_url"]}">已完成收盤累計證據</a> · <a href="{late_meta["html_url"]}">今日逾時證據</a></p>')
    add("產品價值、使用情境與SWOT", table(["使用者", "使用情境", "提供的幫助與邊界"], [
        ["研究學生／教授", "專題方法展示與可重現比較", "分清模型、資料時序、指標及證據；不保證精準"],
        ["自主研究者", "盤前記錄與盤後覆盤", "保存當時數字與消息，觀察偏差；不提供已驗證交易策略"],
        ["資料／模型開發者", "追查來源失敗與版本效果", "按日期／合約／版本核對缺漏，保留回復路徑"]])
        + table(["SWOT", "實際定位"], [
        ["S 優勢", "來源、截止與版本可查；數字先凍結；四價分項、共同日期與失敗均可檢查"],
        ["W 弱點", "價格尚未勝過強基準；新聞樣本少；固定解析器缺陷阻斷試跑；Sheet未整合"],
        ["O 機會", "作為學術與學習工具，把量化、資訊證據與覆盤集中；後續可做預宣告特徵消融"],
        ["T 威脅", "資料格式變動、延遲、模型漂移、預訓練重疊、授權限制及事後挑結果"]])
        + '<p>展示方式：開啟隨包 demo/report.html，切換市場、查看曲線與合成晨報／覆盤。合成示範只驗證操作，不計入真實命中；舊16頁簡報為11/16版本，最新結論以本報告為準。</p>')
    add("結論、驗收邊界與後續順序", '<p>主要研究交付已具備：整體流程與時間軸、兩市場分項MAE／RMSE、八張曲線、基準與TimesFM比較、研究報告、介面展示及SWOT。量化與新聞已完成共同指標和首次實際配對，但目前增益仍無法定論。</p>'
        '<p>先處理前向資料可靠性，再累積固定版本的新樣本。解析修正應有針對缺欄位／低成交遠月的測試、完整同合約資料仍須拒絕缺失；若啟用新版本，事前記錄新程式及協議指紋，分組統計，不倒填失敗日。這份交付沒有更動固定試跑、原四任務、Sheet或PR。</p>'
        '<p>其後才做ARIMA同日期同資訊集合對照，再視訓練／驗證結果考慮情境切換；不反覆調已見評估區間。正式整合還需Sheet實際寫入／讀回與遷移證據，完成展示不能代替。</p>'
        '<p>限制：歷史已見、發布時點未知、TimesFM預訓練重疊未知、樣本與制度變動、來源解析與排程延遲。結果只支持當前研究設計，沒有獲利、可靠性或普遍LLM增益保證。</p>'
        '<p>證據：price-comparison.csv、evidence/timesfm3-summary.json、direction-sensitivity.csv、forward-status.json。完整逐日行情、原始新聞HTML、逐字稿與模型權重不放入本包。</p>')
    css = '''@page{size:A4;margin:14mm}*{box-sizing:border-box}body{font:14px/1.65 "Noto Sans CJK TC",sans-serif;color:#17384b;margin:0;background:#edf3f5}main{max-width:980px;margin:auto}section{background:white;padding:28px;margin:18px 0;break-before:page}section:first-child{break-before:auto}h1{font-size:25px;line-height:1.4;margin:0 0 18px;border-bottom:3px solid #137878;padding-bottom:12px}p{margin:13px 0}table{border-collapse:collapse;width:100%;font-size:12px;margin:15px 0}td,th{padding:7px;border-bottom:1px solid #d9e3e8;text-align:left;vertical-align:top}th{background:#e6f1f3}tr{break-inside:avoid}img{max-width:100%;display:block;margin:12px auto}figure{margin:14px 0;break-inside:avoid}figcaption,.caption{font-size:11px;color:#59707d}.note{background:#fff2d7;border-left:4px solid #d8a033;padding:14px}.eyebrow{font-size:12px;color:#137878}.lead{font-size:19px}a{color:#116867}@media print{body{font-size:12px;background:white}main{max-width:none}section{padding:0;margin:0}h1{font-size:22px}table{font-size:10px}td,th{padding:5px}figure img{max-height:108mm;width:auto}section>img{max-height:120mm;width:auto}}@media(max-width:600px){section{padding:14px}h1{font-size:22px}table{font-size:10px}td,th{padding:4px;overflow-wrap:anywhere}}'''
    (output / "report.html").write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>教授要求驗收與整合研究報告</title><style>'+css+'</style><main>'+"".join(pages)+'</main></html>', encoding="utf-8")
    md = "# 教授要求驗收表｜"+REPORT_DATE+"\n\n"
    md += "狀態依原凍結研究與最新GitHub伺服器回讀核對；不把資料不足寫成已證明改善。\n\n| 項目 | 狀態 | 交付證據／邊界 |\n|---|---|---|\n"
    md += "\n".join("| "+" | ".join(row)+" |" for row in checklist)+"\n\n"
    md += f'最後完成收盤：{last["target_date"]}，{cumulative["paired_n"]}/{cumulative["expected_scheduled_pairs"]}對；每市場1筆。今日僅盤前逾時，不提前算收盤。\n\n'
    md += "閱讀 report.pdf／report.html；demo/report.html保留原離線互動示範，demo/presentation.pptx保留原16頁簡報。原檔案日期與限制不改寫。\n\n"
    md += "優先未完成：足夠真實新聞配對及來源可靠性；正式Sheet整合未完成。ARIMA與策略切換仍後排。本次無模型擬合、無程式試跑版本切換、無PR更動。\n"
    (output / "CHECKLIST.md").write_text(md, encoding="utf-8")
    shutil.copytree(assets, output / "demo")
    inputs = [root/"reports/academic/2026-11-16/price-comparison.csv",
              root/"reports/followup/2026-11-16/timesfm3-summary.json", protocol]
    manifest = {"recorded_at": now().isoformat(), "trial_source_commit": COMMIT,
                "report_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "private_comments_sha256": hashlib.sha256(comments_path.read_bytes()).hexdigest(),
                "private_failure_readback_sha256": hashlib.sha256(latest_failure.read_bytes()).hexdigest(),
                "sections": len(pages), "comparison_replay_verified": True,
                "model_fits": 0, "real_sheet_write": False, "PR_modified": False}
    (output / "source-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(output), "sections": len(pages), "last_completed_close": last["target_date"],
                      "paired_n": cumulative["paired_n"], "expected_pairs": cumulative["expected_scheduled_pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comments", type=Path, required=True)
    parser.add_argument("--latest-failure", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.comments, args.latest_failure, args.assets, args.output)

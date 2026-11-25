# SRT 斷句修正工具

修正 AI 語音辨識字幕中「把詞切斷」的斷句，例如：

```
原：「我們明天要去台」|「北看展覽」
新：「我們明天要去台北」|「看展覽」
```

## 安裝

```bash
pip install -r requirements.txt
```

## 使用

```bash
python srt_resegment.py input.srt                       # 輸出 input.fixed.srt
python srt_resegment.py input.srt --dry-run             # 只列出修改，不寫檔
python srt_resegment.py input.srt -o out.srt --report changes.txt
python srt_resegment.py input.srt --dict mywords.txt    # 加入人名/專有名詞
python srt_resegment.py input.srt --en-words examples/en.txt  # 修正切斷的英文單字
python srt_resegment.py input.srt --encoding big5       # Big5 編碼的檔案
```

## 運作原理

1. 讀入 SRT，同一條字幕的多行文字先合併成一行。
2. 檢查每一對相鄰字幕：把兩段文字接起來，用 jieba 分詞的語言模型計算
   「在原斷點強制切開」比「最佳分詞」差多少。差很多就代表原斷點切到詞的中間
   （例如「台|北」「設|定」「20|24年」）。
3. 在原斷點前後 `--max-shift` 個字內找新斷點，評分考量：
   - 不切斷詞（語言模型分數）
   - 離原斷點越近越好
   - 斷在標點（，。！？）後面優先
   - 下一句不以語助詞（了、嗎、呢、吧、們…）開頭
   - 上一句不以「的、在、把、和…」結尾
   - 每條長度不超過 `--max-chars`
4. 新斷點分數比原本好 `--min-gain` 以上才修改；找不到好斷點、且合併後不會太長時，
   會把兩條合成一條（`--no-merge` 可關閉）。
5. 依字數比例重新估算交界的時間點。
6. 字幕間隔超過 `--max-gap` 秒的交界視為真的停頓，不會處理。

繁體字會逐字轉成簡體再分詞（jieba 內建詞典為簡體），輸出仍然是原本的繁體文字。

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--max-chars` | 18 | 每條字幕最大長度（中文字算 1，英數字算 0.5） |
| `--max-line` | 0 | 超過這個長度就在詞界折成兩行（0 表示不折行） |
| `--max-shift` | 8 | 新斷點離原斷點最多幾個字 |
| `--min-gain` | 3.0 | 修改門檻：越大越保守，越小改得越多 |
| `--max-gap` | 1.5 | 間隔超過幾秒的交界不處理 |
| `--no-merge` | | 不把兩條字幕合併成一條 |
| `--dict` | | 自訂詞典，每行「詞 [詞頻]」 |
| `--en-words` | | 英文單字表（每行一個字），提供時才會修正切斷的英文單字 |

## 建議

- 先用 `--dry-run` 看修改清單，再決定要不要調整 `--min-gain`。
- 影片裡常出現的人名、品牌、專業術語請加進 `--dict`，判斷會準很多。
- 時間軸是依字數比例估算的，不是從音訊重新對齊；若要精準到字，需要辨識結果附帶逐字時間戳。

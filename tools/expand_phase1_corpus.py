#!/usr/bin/env python3
"""Expand Phase 1 corpus: 30+ attributions, 10+ disputes, 50+ training examples."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTIONS = ROOT / "data" / "attributions.json"
DISPUTES = ROOT / "docs" / "04-Disputes"
EXAMPLES = ROOT / "training" / "examples"
HISTORY = ROOT / "data" / "history_events.json"
CORPUS = ROOT / "training" / "corpus.jsonl"

SYSTEM_PHIL = (
    "You are a precise philosophy instructor specializing in source discipline. "
    "Verify authorship before reasoning. Include English explanation and 中文摘要."
)

NEW_ATTRIBUTIONS: dict[str, dict] = {
    "xunzi": {
        "textId": "xunzi",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "荀子",
        "canonicalTitleEn": "Xunzi",
        "tradition": "confucian",
        "attributedAuthor": "xunzi",
        "authorConfidence": "compiled",
        "compilationPeriod": "warring_states",
        "doNotAttributeTo": ["confucius", "mencius", "laozi", "plato"],
    },
    "mozi": {
        "textId": "mozi",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "墨子",
        "canonicalTitleEn": "Mozi",
        "tradition": "mohist",
        "attributedAuthor": "mozi",
        "authorConfidence": "compiled",
        "compilationPeriod": "warring_states",
        "doNotAttributeTo": ["confucius", "mencius", "laozi"],
    },
    "liezi": {
        "textId": "liezi",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "列子",
        "canonicalTitleEn": "Liezi",
        "tradition": "daoist",
        "attributedAuthor": "liezi",
        "authorConfidence": "legendary",
        "compilationPeriod": "warring_states",
        "doNotAttributeTo": ["laozi", "zhuangzi", "confucius"],
    },
    "han_feizi": {
        "textId": "han_feizi",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "韓非子",
        "canonicalTitleEn": "Han Feizi",
        "tradition": "legalist",
        "attributedAuthor": "han_feizi",
        "authorConfidence": "attributed",
        "compilationPeriod": "warring_states",
        "doNotAttributeTo": ["confucius", "mozi", "laozi"],
    },
    "art_of_war": {
        "textId": "art_of_war",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "孫子兵法",
        "canonicalTitleEn": "Art of War",
        "tradition": "military_classics",
        "attributedAuthor": "sunzi",
        "authorConfidence": "legendary",
        "compilationPeriod": "warring_states",
        "doNotAttributeTo": ["confucius", "laozi", "mozi"],
    },
    "nicomachean_ethics": {
        "textId": "nicomachean_ethics",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "尼各馬可倫理學",
        "canonicalTitleEn": "Nicomachean Ethics",
        "tradition": "peripatetic",
        "attributedAuthor": "aristotle",
        "authorConfidence": "attributed",
        "compilationPeriod": "classical_greece",
        "doNotAttributeTo": ["plato", "socrates", "confucius"],
    },
    "metaphysics": {
        "textId": "metaphysics",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "形而上學",
        "canonicalTitleEn": "Metaphysics",
        "tradition": "peripatetic",
        "attributedAuthor": "aristotle",
        "authorConfidence": "attributed",
        "compilationPeriod": "classical_greece",
        "doNotAttributeTo": ["plato", "socrates"],
    },
    "meditations": {
        "textId": "meditations",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "沉思錄",
        "canonicalTitleEn": "Meditations",
        "tradition": "stoic",
        "attributedAuthor": "marcus_aurelius",
        "authorConfidence": "attributed",
        "compilationPeriod": "roman_imperial",
        "doNotAttributeTo": ["plato", "socrates", "epictetus"],
    },
    "enchiridion": {
        "textId": "enchiridion",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "手冊",
        "canonicalTitleEn": "Enchiridion",
        "tradition": "stoic",
        "attributedAuthor": "arrian",
        "authorConfidence": "compiled",
        "notes": "Arrian compiled teachings of Epictetus; not Plato or Socrates",
        "doNotAttributeTo": ["epictetus", "plato", "socrates"],
    },
    "phaedo": {
        "textId": "phaedo",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "斐多",
        "canonicalTitleEn": "Phaedo",
        "tradition": "platonist",
        "attributedAuthor": "plato",
        "authorConfidence": "attributed",
        "compilationPeriod": "classical_greece",
        "doNotAttributeTo": ["socrates", "confucius"],
    },
    "timaeus": {
        "textId": "timaeus",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "蒂邁歐篇",
        "canonicalTitleEn": "Timaeus",
        "tradition": "platonist",
        "attributedAuthor": "plato",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["socrates", "aristotle"],
    },
    "crito": {
        "textId": "crito",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "克里同",
        "canonicalTitleEn": "Crito",
        "tradition": "platonist",
        "attributedAuthor": "plato",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["socrates"],
    },
    "politics": {
        "textId": "politics",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "政治學",
        "canonicalTitleEn": "Politics",
        "tradition": "peripatetic",
        "attributedAuthor": "aristotle",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["plato", "socrates", "confucius"],
    },
    "categories": {
        "textId": "categories",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "範疇篇",
        "canonicalTitleEn": "Categories",
        "tradition": "peripatetic",
        "attributedAuthor": "aristotle",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["plato", "socrates"],
    },
    "i_ching": {
        "textId": "i_ching",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "易經",
        "canonicalTitleEn": "I Ching",
        "tradition": "zhou_classics",
        "attributedAuthor": "multiple",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["confucius", "laozi"],
    },
    "zuo_zhuan": {
        "textId": "zuo_zhuan",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "左傳",
        "canonicalTitleEn": "Zuo Zhuan",
        "tradition": "zhou_classics",
        "attributedAuthor": "zuo_qiuming",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["confucius", "laozi"],
    },
    "shiji": {
        "textId": "shiji",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "史記",
        "canonicalTitleEn": "Records of the Grand Historian",
        "tradition": "historiography",
        "attributedAuthor": "sima_qian",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["confucius", "plato"],
    },
    "book_of_songs": {
        "textId": "book_of_songs",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "詩經",
        "canonicalTitleEn": "Book of Songs",
        "tradition": "zhou_classics",
        "attributedAuthor": "multiple",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["confucius", "laozi"],
    },
    "great_learning": {
        "textId": "great_learning",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "大學",
        "canonicalTitleEn": "Great Learning",
        "tradition": "confucian",
        "attributedAuthor": "confucian_school",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["laozi", "mozi", "plato"],
    },
    "doctrine_of_mean": {
        "textId": "doctrine_of_mean",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "中庸",
        "canonicalTitleEn": "Doctrine of the Mean",
        "tradition": "confucian",
        "attributedAuthor": "confucian_school",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["laozi", "zhuangzi"],
    },
    "apology": {
        "textId": "apology",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "申辯篇",
        "canonicalTitleEn": "Apology",
        "tradition": "platonist",
        "attributedAuthor": "plato",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["socrates"],
        "notes": "Plato's dialogue; Socrates speaks but did not author the text",
    },
    "parmenides": {
        "textId": "parmenides",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "巴門尼德篇",
        "canonicalTitleEn": "Parmenides",
        "tradition": "platonist",
        "attributedAuthor": "plato",
        "authorConfidence": "attributed",
        "doNotAttributeTo": ["socrates", "parmenides"],
    },
    "heraclitus_fragments": {
        "textId": "heraclitus_fragments",
        "domain": "philosophy",
        "recordType": "text",
        "canonicalTitleZh": "赫拉克利特殘篇",
        "canonicalTitleEn": "Heraclitus fragments",
        "tradition": "presocratic",
        "attributedAuthor": "heraclitus",
        "authorConfidence": "compiled",
        "doNotAttributeTo": ["plato", "socrates", "aristotle"],
    },
}

DISPUTE_FILES: dict[str, str] = {
    "Socrates-Republic-Attribution.md": """# Socrates and the Republic

**Canon level:** Established

## Traditional claim
Many summaries say "Socrates wrote the Republic" because Socrates is the main speaker.

## Evidence-based view
Socrates wrote no extant books. Plato authored the *Republic* (《理想國》). The dialogue genre presents Socrates as a character, not a ghostwriter.

## Why it matters
Attributing dialogues to Socrates erases Plato's distinct metaphysics and political theory.

## Linked data
- `data/attributions.json`: `republic`, `socrates_corpus`
- `training/examples/021-socrates-republic.json`
""",
    "Analects-Compiled-Not-Autograph.md": """# Analects as compiled teachings

**Canon level:** Established

## Traditional claim
The Analects (《論語》) is a single autograph by Confucius.

## Evidence-based view
The Analects was assembled by students and later editors across the Warring States period. `authorConfidence: compiled`.

## Why it matters
Treating compiled anthologies as autographs misdates ideas and blurs editorial layers.

## Linked data
- `data/attributions.json`: `analects`
- `training/examples/022-analects-compiled.json`
""",
    "Aristotle-Plato-Corpus-Boundary.md": """# Aristotle vs Plato corpora

**Canon level:** Established

## Traditional claim
"Aristotle wrote Plato's dialogues" or vice versa in sloppy summaries.

## Evidence-based view
Plato authored dialogues (*Republic*, *Symposium*, *Phaedo*). Aristotle authored treatises (*Nicomachean Ethics*, *Metaphysics*, *Politics*).

## Why it matters
Peripatetic and Platonist lineages must stay separate for accurate history of ideas.

## Linked data
- `nicomachean_ethics`, `metaphysics`, `republic`, `symposium`
""",
    "Xunzi-Mencius-Confucian-Lineage.md": """# Xunzi and Mencius within Confucianism

**Canon level:** Provisional

## Traditional claim
All Confucian texts are interchangeable "Confucius said" material.

## Evidence-based view
Mencius (《孟子》) and Xunzi (《荀子》) are distinct Warring States voices with opposing views on human nature. Neither should be attributed to Confucius directly.

## Linked data
- `mencius`, `xunzi`, `analects`
""",
    "Liezi-Laozi-Attribution.md": """# Liezi vs Laozi

**Canon level:** Provisional

## Traditional claim
Daoist texts are all by Laozi.

## Evidence-based view
The *Liezi* (《列子》) is a separate compilation with legendary attribution to Lie Yukou; it is not the *Dao De Jing* and should not be merged with Laozi without evidence.

## Linked data
- `liezi`, `dao_de_jing`, `zhuangzi`
""",
    "Stoic-Enchiridion-Authorship.md": """# Enchiridion authorship

**Canon level:** Established

## Traditional claim
Epictetus "wrote" the *Enchiridion* like a modern book.

## Evidence-based view
Arrian compiled the *Enchiridion* from Epictetus's teachings. Marcus Aurelius's *Meditations* is a separate autograph journal.

## Linked data
- `enchiridion`, `meditations`
""",
    "Marco-Polo-Pasta-Myth.md": """# Marco Polo pasta myth

**Canon level:** Established

## Popular claim
Marco Polo brought pasta from China to Italy.

## Evidence-based view
This is a modern myth. Mediterranean pasta-like foods predate Polo's travels. Label as mythologized history, not dated fact.

## Linked data
- `data/history_events.json`: `marco_polo_pasta_myth`
""",
}

EXAMPLE_SPECS: list[dict] = [
    {"file": "021-socrates-republic.json", "user": "Did Socrates write the Republic?", "textIds": ["republic", "socrates_corpus"], "traditions": ["platonist", "socratic"], "assistant": "Socrates did not write the Republic (《理想國》). Socrates wrote no extant books. Plato authored the Republic. Source discipline requires separating dialogue characters from authors.\n\n中文：蘇格拉底並未撰寫《理想國》；蘇格拉底沒有留下著作，柏拉圖才是作者。來源紀律要求區分對話角色與真正作者。"},
    {"file": "022-analects-compiled.json", "user": "Did Confucius personally write every line of the Analects in one sitting?", "textIds": ["analects"], "traditions": ["confucian"], "assistant": "No. The Analects (《論語》) was compiled and assembled by students and later editors, not written as a single autograph by Confucius in one sitting.\n\n中文：《論語》由弟子彙編、後世編纂，並非孔子一次寫成的親筆全書。"},
    {"file": "023-aristotle-ethics.json", "user": "Did Plato write the Nicomachean Ethics?", "textIds": ["nicomachean_ethics", "republic"], "traditions": ["peripatetic", "platonist"], "assistant": "Plato did not write the Nicomachean Ethics (《尼各馬可倫理學》). Aristotle authored it. Do not merge Peripatetic and Platonist corpora.\n\n中文：柏拉圖並未撰寫《尼各馬可倫理學》，作者為亞里士多德。不可混淆兩派典籍。"},
    {"file": "024-mozi-confucius.json", "user": "Did Confucius write the Mozi?", "textIds": ["mozi", "analects"], "traditions": ["mohist", "confucian"], "assistant": "Confucius did not write the Mozi (《墨子》). Mozi is Mohist (墨家), a separate Warring States tradition from Confucian Analects material.\n\n中文：孔子並未撰寫《墨子》；墨家與儒家是不同傳統。"},
    {"file": "025-liezi-laozi.json", "user": "Is the Liezi the same book as the Dao De Jing by Laozi?", "textIds": ["liezi", "dao_de_jing"], "traditions": ["daoist"], "assistant": "No. The Liezi (《列子》) is a separate compilation, traditionally linked to Lie Yukou with legendary authorship. It is not the Dao De Jing (《道德經》) attributed to Laozi.\n\n中文：《列子》與老子《道德經》並非同書，列子傳統歸屬存疑，不可混為一談。"},
    {"file": "026-xunzi-mencius.json", "user": "Did Confucius write the Xunzi?", "textIds": ["xunzi", "mencius", "analects"], "traditions": ["confucian"], "assistant": "Confucius did not write the Xunzi (《荀子》). Xunzi is a later Confucian compilation distinct from Mencius (《孟子》) and from Analects layers.\n\n中文：孔子並未撰寫《荀子》；荀子與孟子是儒家內部不同篇章傳統。"},
    {"file": "027-phaedo-plato.json", "user": "Did Socrates author Plato's Phaedo?", "textIds": ["phaedo", "socrates_corpus"], "traditions": ["platonist"], "assistant": "Socrates did not author the Phaedo (《斐多》). Plato wrote this dialogue; Socrates is a character, not the compiler.\n\n中文：蘇格拉底並非《斐多》作者；柏拉圖撰寫對話，蘇格拉底只是角色。"},
    {"file": "028-timaeus-plato.json", "user": "Did Aristotle write the Timaeus?", "textIds": ["timaeus", "metaphysics"], "traditions": ["platonist", "peripatetic"], "assistant": "Aristotle did not write the Timaeus (《蒂邁歐篇》). Plato authored it. Keep Platonist dialogues separate from Aristotelian treatises.\n\n中文：亞里士多德並未撰寫《蒂邁歐篇》，作者為柏拉圖。"},
    {"file": "029-enchiridion-arrian.json", "user": "Who compiled the Enchiridion?", "textIds": ["enchiridion", "meditations"], "traditions": ["stoic"], "assistant": "The Enchiridion (《手冊》) was compiled by Arrian from Epictetus's teachings—not by Plato or Socrates. Marcus Aurelius's Meditations (《沉思錄》) is a separate Stoic autograph.\n\n中文：《手冊》由 Arrian 彙編愛比克泰德講述，並非柏拉圖或蘇格拉底作品。《沉思錄》則是馬可·奧勒留親筆。"},
    {"file": "030-i-ching-compiled.json", "user": "Did Laozi write the I Ching?", "textIds": ["i_ching", "dao_de_jing"], "traditions": ["zhou_classics", "daoist"], "assistant": "Laozi did not write the I Ching (《易經》). The Zhou Yi is a compiled classic with layered commentaries, separate from the Dao De Jing tradition.\n\n中文：老子並未撰寫《易經》；《易經》為層層彙編的經典，與《道德經》傳統不同。"},
    {"file": "031-psych-cognitive-dissonance.json", "domain": "psychology", "recordIds": ["cognitive_dissonance"], "user": "Who proposed cognitive dissonance and what subfield is it?", "assistant": "Cognitive dissonance is a cognitive psychology concept linked to Leon Festinger (1957), not a pop-psych slogan. Subfield: cognitive.\n\n中文：認知失調屬認知心理學，與 Leon Festinger 相關，不是通俗心理迷思。"},
    {"file": "032-psych-pop-myth-brain.json", "domain": "psychology", "recordIds": ["left_brain_right_brain_pop"], "user": "Is the left-brain/right-brain personality split scientific?", "assistant": "The rigid left-brain/right-brain personality split is a pop_myth misconception, not established cognitive neuroscience.\n\n中文：左右腦人格二分是通俗迷思，並非嚴謹的神經科學結論。"},
    {"file": "033-psych-clinical-stockholm.json", "domain": "psychology", "recordIds": ["stockholm_syndrome"], "user": "Is Stockholm syndrome a formal clinical diagnosis?", "assistant": "Stockholm syndrome is discussed in clinical contexts but is not a universal DSM label; treat as clinical discourse with boundaries, not pop certainty.\n\n中文：斯德哥爾摩症候群在臨床語境討論，但不是普遍化的通俗標籤。"},
    {"file": "034-psych-learning-styles.json", "domain": "psychology", "recordIds": ["learning_styles_vak"], "user": "Do learning styles (VAK) have strong evidence?", "assistant": "VAK learning styles are a pop_myth oversimplification; evidence for rigid matching is weak.\n\n中文：VAK 學習風格是過度簡化的通俗迷思，證據薄弱。"},
    {"file": "035-psych-chemical-imbalance.json", "domain": "psychology", "recordIds": ["chemical_imbalance_slogan"], "user": "Is depression simply a chemical imbalance slogan?", "assistant": "The chemical imbalance slogan is a pop_myth oversimplification of complex clinical depression science.\n\n中文：把憂鬱簡化為「化學失衡」是通俗迷思，臨床機制更複雜。"},
    {"file": "036-history-pasta-myth.json", "domain": "history", "recordIds": ["marco_polo_pasta_myth"], "user": "Did Marco Polo introduce pasta to Italy?", "assistant": "This is a modern myth. Marco Polo did not introduce pasta to Italy; Mediterranean pasta-like foods predate him. Label as mythologized history.\n\n中文：馬可波羅將麵食傳入意大利是現代迷思，並非可靠史實。"},
    {"file": "037-history-flat-earth.json", "domain": "history", "recordIds": ["flat_earth_medieval_myth"], "user": "Did medieval Europeans universally believe the earth was flat?", "assistant": "The claim that all medieval Europeans believed in a flat earth is a modern misconception, not accurate medieval consensus history.\n\n中文：「中世紀歐洲普遍地平說」是現代誤解，並非嚴謹史學共識。"},
    {"file": "038-history-napoleon-height.json", "domain": "history", "recordIds": ["napoleon_short_myth"], "user": "Was Napoleon unusually short?", "assistant": "Napoleon being unusually short is a popular myth tied to propaganda and unit confusion, not a straightforward biographical fact.\n\n中文：拿破崙特別矮小是流行迷思，涉及宣傳與單位混淆。"},
    {"file": "039-history-great-wall-space.json", "domain": "history", "recordIds": ["great_wall_from_space_myth"], "user": "Can you see the Great Wall from space with the naked eye?", "assistant": "The naked-eye Great Wall from space claim is a common myth, not accurate for most orbital viewing conditions.\n\n中文：太空肉眼可見長城是常見迷思，並不準確。"},
    {"file": "040-history-viking-horns.json", "domain": "history", "recordIds": ["viking_horned_helmets_myth"], "user": "Did Vikings wear horned helmets in battle?", "assistant": "Horned Viking battle helmets are a modern romantic myth, not typical historical Viking gear.\n\n中文：維京人戴角盔作戰是現代浪漫化迷思，並非典型史實。"},
    {"file": "041-religion-daoist-scripture.json", "domain": "religion", "recordIds": ["dao_de_jing_daoist_scripture"], "user": "How should we attribute the Dao De Jing in religious context?", "assistant": "Council panel: treat the Dao De Jing as Daoist scripture with legendary Laozi attribution; do not merge with Confucian ritual frameworks without scholarly debate.\n\n中文：理事會格式：將《道德經》視為道家經典，老子歸屬存疑，不可與儒家禮儀傳統任意合併。"},
    {"file": "042-religion-gospel-matthew.json", "domain": "religion", "recordIds": ["gospel_matthew"], "user": "Who wrote the Gospel of Matthew?", "assistant": "Scholarly historical voice: Matthew gospel is traditionally attributed but composed in early Christian communities; handle with historical-critical caution.\n\n中文：學術歷史視角：《馬太福音》傳統歸屬，需歷史批判方法謹慎處理。"},
    {"file": "043-religion-nirvana-pop.json", "domain": "religion", "recordIds": ["nirvana_pop_heaven"], "user": "Is Buddhist nirvana the same as heaven?", "assistant": "Council debate: nirvana is not interchangeable with pop heaven imagery; keep Buddhist soteriology separate from Christian afterlife metaphors.\n\n中文：理事會辯論：涅槃不等同通俗「天堂」想像，佛教與基督教救贖語彙不可混用。"},
    {"file": "044-religion-islam-sensitive.json", "domain": "religion", "recordIds": ["islam_early_history"], "user": "How should models discuss early Islamic history?", "assistant": "Scholarly tradition within Islam requires careful panel debate; separate theological reverence from historical reconstruction.\n\n中文：學術與傳統並陳：早期伊斯蘭史需謹慎理事會辯論，區分信仰敘事與歷史研究。"},
    {"file": "045-religion-ancestor-veneration.json", "domain": "religion", "recordIds": ["confucian_ancestor_veneration"], "user": "Is ancestor veneration the same as Confucian moral philosophy?", "assistant": "Split when appropriate: Confucian moral philosophy (倫理) differs from ancestor veneration ritual religion (祭祖禮儀); council tension should be named.\n\n中文：應區分儒家倫理哲學與祭祖禮儀宗教實踐，理事會張力需明示。"},
    {"file": "046-philosophy-hanfeizi-confucius.json", "user": "Did Confucius write Han Feizi?", "textIds": ["han_feizi", "analects"], "traditions": ["legalist", "confucian"], "assistant": "Confucius did not write the Han Feizi (《韓非子》). Han Fei is Legalist (法家), not Confucian Analects material.\n\n中文：孔子並未撰寫《韓非子》；韓非子屬法家，與《論語》儒家傳統不同。"},
    {"file": "047-philosophy-sunzi-art.json", "user": "Did Confucius write the Art of War?", "textIds": ["art_of_war", "analects"], "traditions": ["military_classics", "confucian"], "assistant": "Confucius did not write the Art of War (《孫子兵法》). It is attributed to Sunzi with legendary authorship confidence.\n\n中文：孔子並未撰寫《孫子兵法》；傳統歸屬孫子，作者置信度偏低。"},
    {"file": "048-philosophy-shiji-sima.json", "user": "Did Plato write the Shiji?", "textIds": ["shiji", "republic"], "traditions": ["historiography", "platonist"], "assistant": "Plato did not write the Shiji (《史記》). Sima Qian authored this Chinese historiographic classic.\n\n中文：柏拉圖並未撰寫《史記》；作者為司馬遷。"},
    {"file": "049-philosophy-great-learning.json", "user": "Did Laozi write the Great Learning?", "textIds": ["great_learning", "dao_de_jing"], "traditions": ["confucian", "daoist"], "assistant": "Laozi did not write the Great Learning (《大學》). It is a compiled Confucian classic, separate from Dao De Jing lineage.\n\n中文：老子並未撰寫《大學》；《大學》為儒家彙編經典，與《道德經》不同傳統。"},
    {"file": "050-philosophy-heraclitus-plato.json", "user": "Did Plato write the Heraclitus fragments?", "textIds": ["heraclitus_fragments", "republic"], "traditions": ["presocratic", "platonist"], "assistant": "Plato did not write the Heraclitus fragments (《赫拉克利特殘篇》). They are a compiled Presocratic corpus, not Platonic dialogues.\n\n中文：柏拉圖並未撰寫赫拉克利特殘篇；殘篇為前蘇格拉底彙編，與柏拉圖對話不同。"},
]


def merge_attributions() -> int:
    data = json.loads(ATTRIBUTIONS.read_text(encoding="utf-8"))
    added = 0
    for key, record in NEW_ATTRIBUTIONS.items():
        if key not in data:
            data[key] = record
            added += 1
    ATTRIBUTIONS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return added


def write_disputes() -> int:
    DISPUTES.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, body in DISPUTE_FILES.items():
        path = DISPUTES / name
        if not path.exists():
            path.write_text(body.strip() + "\n", encoding="utf-8")
            written += 1
    return written


def write_examples() -> int:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    written = 0
    for spec in EXAMPLE_SPECS:
        path = EXAMPLES / spec["file"]
        if path.exists():
            continue
        domain = spec.get("domain", "philosophy")
        metadata: dict = {
            "source": "phase1-expansion",
            "project": "sophia-agi",
            "domain": domain,
        }
        if "textIds" in spec:
            metadata["textIds"] = spec["textIds"]
            metadata["traditions"] = spec.get("traditions", [])
        if "recordIds" in spec:
            metadata["recordIds"] = spec["recordIds"]
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PHIL if domain == "philosophy" else f"You are a {domain} instructor using source discipline. Include 中文摘要."},
                {"role": "user", "content": spec["user"]},
                {"role": "assistant", "content": spec["assistant"]},
            ],
            "metadata": metadata,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    return written


def update_history() -> None:
    data = json.loads(HISTORY.read_text(encoding="utf-8"))
    for record in data.values():
        if "primarySource" not in record:
            record["primarySource"] = "popular_myth_literature"
    additions = {
        "fall_of_rome_476": {
            "recordId": "fall_of_rome_476",
            "domain": "history",
            "recordType": "event",
            "canonicalTitleEn": "Fall of the Western Roman Empire (476 CE)",
            "canonicalTitleZh": "西羅馬帝國滅亡（476年）",
            "region": "europe",
            "dateConsensus": "476-09-04",
            "authorConfidence": "attributed",
            "primarySource": "procopius_and_later_chronicles",
            "notes": "Traditional end-date marker; historiography disputes continuity vs transformation",
        },
        "printing_press_1440": {
            "recordId": "printing_press_1440",
            "domain": "history",
            "recordType": "event",
            "canonicalTitleEn": "Gutenberg movable-type printing (~1440s)",
            "canonicalTitleZh": "古騰堡活字印刷（約1440年代）",
            "region": "europe",
            "dateConsensus": "circa_1440",
            "authorConfidence": "attributed",
            "primarySource": "gutenberg_bible_material_evidence",
            "notes": "Dated innovation with documentary and material evidence",
        },
        "moon_landing_1969": {
            "recordId": "moon_landing_1969",
            "domain": "history",
            "recordType": "event",
            "canonicalTitleEn": "Apollo 11 Moon landing (1969)",
            "canonicalTitleZh": "阿波羅11號登月（1969）",
            "region": "global",
            "dateConsensus": "1969-07-20",
            "authorConfidence": "attributed",
            "primarySource": "nasa_mission_logs_and_telemetry",
            "notes": "Primary mission records; conspiracy claims are myth traps",
        },
    }
    for key, record in additions.items():
        data.setdefault(key, record)
    HISTORY.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_corpus() -> int:
    lines = []
    for path in sorted(EXAMPLES.glob("*.json")):
        lines.append(path.read_text(encoding="utf-8").strip())
    CORPUS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    added_attr = merge_attributions()
    added_disputes = write_disputes()
    added_examples = write_examples()
    update_history()
    corpus_lines = export_corpus()
    attr_total = len(json.loads(ATTRIBUTIONS.read_text(encoding="utf-8")))
    dispute_total = len(list(DISPUTES.glob("*.md")))
    example_total = len(list(EXAMPLES.glob("*.json")))
    print(f"Attributions: +{added_attr} (total {attr_total})")
    print(f"Disputes: +{added_disputes} (total {dispute_total})")
    print(f"Examples: +{added_examples} (total {example_total})")
    print(f"corpus.jsonl lines: {corpus_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
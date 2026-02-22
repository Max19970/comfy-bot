from __future__ import annotations

import re

QUALITY_PRESETS: dict[str, tuple[str, ...]] = {
    "pony": ("score_9", "score_8_up", "score_7_up", "source_anime"),
    "illustrious": ("masterpiece", "best quality", "amazing quality", "newest"),
    "default": ("masterpiece", "best quality", "highly detailed"),
}

CHECKPOINT_HINTS: list[tuple[str, str]] = [
    ("pony", "pony"),
    ("pdxl", "pony"),
    ("autismmix", "pony"),
    ("illustrious", "illustrious"),
    ("wai", "illustrious"),
    ("noob", "illustrious"),
    ("hassaku", "illustrious"),
]

CONTROL_MARKERS = (
    "<|",
    "|>",
    "<0x",
    "target:",
    "target :",
)

ANCHOR_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "this",
    "that",
    "these",
    "those",
    "with",
    "without",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "through",
    "for",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "become",
    "becoming",
    "becomes",
    "get",
    "gets",
    "getting",
    "got",
    "make",
    "makes",
    "making",
    "made",
    "as",
    "while",
    "during",
    "after",
    "before",
    "very",
    "more",
    "less",
    "some",
    "any",
    "each",
    "every",
    "all",
    "both",
    "few",
    "many",
    "much",
    "no",
    "not",
    "but",
    "so",
    "if",
    "then",
    "than",
    "too",
    "also",
    "it",
    "its",
    "he",
    "she",
    "his",
    "her",
    "they",
    "their",
    "we",
    "our",
    "my",
    "your",
    "who",
    "what",
    "where",
    "when",
    "how",
    "which",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "must",
    "и",
    "или",
    "на",
    "в",
    "во",
    "с",
    "со",
    "к",
    "по",
    "за",
    "для",
    "без",
    "под",
    "над",
    "из",
    "у",
    "при",
    "не",
    "ни",
    "это",
    "тот",
    "эта",
    "эти",
    "быть",
    "стал",
    "стала",
    "становится",
    "он",
    "она",
    "оно",
    "они",
    "его",
    "её",
    "их",
    "мой",
    "моя",
    "наш",
    "ваш",
    "который",
    "которая",
    "которые",
    "что",
    "как",
    "где",
    "когда",
    "очень",
    "более",
    "менее",
    "тоже",
    "также",
    "но",
    "а",
    "же",
}

RU_EN_VISUAL: dict[str, str] = {
    "девушка": "girl",
    "девочка": "girl",
    "женщина": "woman",
    "парень": "boy",
    "мальчик": "boy",
    "мужчина": "man",
    "кот": "cat",
    "кошка": "cat",
    "котенок": "kitten",
    "котёнок": "kitten",
    "собака": "dog",
    "щенок": "puppy",
    "пёс": "dog",
    "пес": "dog",
    "лошадь": "horse",
    "конь": "horse",
    "птица": "bird",
    "дракон": "dragon",
    "волк": "wolf",
    "лиса": "fox",
    "лис": "fox",
    "кролик": "rabbit",
    "заяц": "rabbit",
    "медведь": "bear",
    "рыцарь": "knight",
    "воин": "warrior",
    "маг": "mage",
    "ведьма": "witch",
    "принцесса": "princess",
    "принц": "prince",
    "король": "king",
    "королева": "queen",
    "робот": "robot",
    "ангел": "angel",
    "демон": "demon",
    "волосы": "hair",
    "глаза": "eyes",
    "лицо": "face",
    "улыбка": "smile",
    "длинные": "long",
    "короткие": "short",
    "большие": "large",
    "маленькие": "small",
    "красный": "red",
    "красная": "red",
    "красное": "red",
    "красные": "red",
    "синий": "blue",
    "синяя": "blue",
    "синее": "blue",
    "синие": "blue",
    "голубой": "light blue",
    "голубая": "light blue",
    "голубые": "light blue",
    "зелёный": "green",
    "зеленый": "green",
    "зелёная": "green",
    "зеленая": "green",
    "жёлтый": "yellow",
    "желтый": "yellow",
    "жёлтая": "yellow",
    "желтая": "yellow",
    "оранжевый": "orange",
    "оранжевая": "orange",
    "фиолетовый": "purple",
    "фиолетовая": "purple",
    "розовый": "pink",
    "розовая": "pink",
    "розовые": "pink",
    "белый": "white",
    "белая": "white",
    "белое": "white",
    "белые": "white",
    "чёрный": "black",
    "черный": "black",
    "чёрная": "black",
    "черная": "black",
    "серый": "grey",
    "серая": "grey",
    "золотой": "golden",
    "золотая": "golden",
    "серебряный": "silver",
    "серебряная": "silver",
    "платье": "dress",
    "юбка": "skirt",
    "шляпа": "hat",
    "шляпе": "hat",
    "корона": "crown",
    "плащ": "cape",
    "доспехи": "armor",
    "броня": "armor",
    "очки": "glasses",
    "маска": "mask",
    "перчатки": "gloves",
    "сапоги": "boots",
    "униформа": "uniform",
    "костюм": "suit",
    "рубашка": "shirt",
    "шарф": "scarf",
    "капюшон": "hood",
    "крылья": "wings",
    "хвост": "tail",
    "город": "city",
    "лес": "forest",
    "море": "sea",
    "океан": "ocean",
    "гора": "mountain",
    "горы": "mountains",
    "река": "river",
    "озеро": "lake",
    "небо": "sky",
    "облака": "clouds",
    "звёзды": "stars",
    "звезды": "stars",
    "луна": "moon",
    "солнце": "sun",
    "закат": "sunset",
    "рассвет": "sunrise",
    "ночь": "night",
    "ночью": "night",
    "день": "day",
    "днём": "day",
    "дождь": "rain",
    "снег": "snow",
    "туман": "fog",
    "гроза": "storm",
    "поле": "field",
    "сад": "garden",
    "цветы": "flowers",
    "цветок": "flower",
    "дерево": "tree",
    "деревья": "trees",
    "трава": "grass",
    "замок": "castle",
    "дворец": "palace",
    "храм": "temple",
    "башня": "tower",
    "улица": "street",
    "мост": "bridge",
    "дом": "house",
    "комната": "room",
    "окно": "window",
    "подоконник": "windowsill",
    "подоконнике": "windowsill",
    "пляж": "beach",
    "пустыня": "desert",
    "джунгли": "jungle",
    "космос": "space",
    "планета": "planet",
    "сидит": "sitting",
    "стоит": "standing",
    "лежит": "lying down",
    "бежит": "running",
    "идёт": "walking",
    "идет": "walking",
    "летит": "flying",
    "плывёт": "swimming",
    "плывет": "swimming",
    "держит": "holding",
    "смотрит": "looking",
    "читает": "reading",
    "играет": "playing",
    "танцует": "dancing",
    "спит": "sleeping",
    "красивый": "beautiful",
    "красивая": "beautiful",
    "милый": "cute",
    "милая": "cute",
    "грустный": "sad",
    "грустная": "sad",
    "счастливый": "happy",
    "счастливая": "happy",
    "тёмный": "dark",
    "темный": "dark",
    "тёмная": "dark",
    "темная": "dark",
    "светлый": "bright",
    "светлая": "bright",
    "магический": "magical",
    "магическая": "magical",
    "мрачный": "gloomy",
    "мрачная": "gloomy",
    "эпический": "epic",
    "эпичный": "epic",
    "фэнтези": "fantasy",
    "фентези": "fantasy",
    "киберпанк": "cyberpunk",
    "стимпанк": "steampunk",
    "реалистичный": "realistic",
    "реалистичная": "realistic",
    "аниме": "anime",
}


def detect_quality_preset(checkpoint: str) -> str:
    name = checkpoint.lower()
    for hint, preset in CHECKPOINT_HINTS:
        if hint in name:
            return preset
    return "default"


def quality_tags_for_checkpoint(checkpoint: str) -> list[str]:
    preset = detect_quality_preset(checkpoint)
    return list(QUALITY_PRESETS.get(preset, QUALITY_PRESETS["default"]))


def extract_anchor_tags(description: str) -> list[str]:
    words = re.split(r"[\s,;:!?.()\[\]{}\"/]+", description.lower())
    anchors: list[str] = []
    for word in words:
        word = word.strip().strip("'-")
        if not word or len(word) < 2:
            continue
        if word in ANCHOR_STOPWORDS:
            continue
        if word.isdigit():
            continue
        translated = RU_EN_VISUAL.get(word)
        if translated:
            anchors.append(translated)
        else:
            anchors.append(word.replace("_", " "))

    raw_words = re.split(r"[\s,;:!?.]+", description.lower())
    raw_words = [w.strip().strip("'-") for w in raw_words if w.strip()]
    for i in range(len(raw_words) - 1):
        w1, w2 = raw_words[i], raw_words[i + 1]
        if w1 in ANCHOR_STOPWORDS or w2 in ANCHOR_STOPWORDS:
            continue
        if len(w1) < 2 or len(w2) < 2:
            continue
        t1 = RU_EN_VISUAL.get(w1, w1)
        t2 = RU_EN_VISUAL.get(w2, w2)
        bigram = f"{t1} {t2}"
        anchors.append(bigram)

    seen: set[str] = set()
    result: list[str] = []
    for tag in anchors:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result[:32]


def split_tags(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,;\n]", raw)
    tags: list[str] = []
    for part in parts:
        cleaned = part.strip().strip("-").strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            tags.append(cleaned)
    return tags


def dedupe_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


def sanitize_final_prompt(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", "", text)
    lines = text.split("\n")
    clean_lines: list[str] = []
    for line in lines:
        lowered = line.casefold()
        if any(marker in lowered for marker in CONTROL_MARKERS):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)
    text = re.sub(r"\s*,\s*,+\s*", ", ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    text = text.strip().strip(",").strip()
    return text

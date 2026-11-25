"""
Hindi-English Transcript Quality Cleaner for LearnMate AI.
Transliterates Perso-Arabic (Urdu) vocabulary into natural Devanagari Hindi,
cleans foreign subword hallucinations (Korean, Cyrillic, French),
and preserves English/Hinglish programming syntax and technical terms.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent

# High-frequency dictionary mapping common Urdu words to Devanagari Hindi
URDU_TO_DEVANAGARI = {
    # Pronouns & Prepositions
    "آپ": "आप", "آپکو": "आपको", "آپکی": "आपकी", "آپکے": "आपके", "آپکا": "आपका", "آپلوگ": "आपलोग",
    "میں": "मैं", "ہم": "हम", "ہمیں": "हमें", "ہمارا": "हमारा", "ہماری": "हमारी", "ہمارے": "हमारे",
    "تو": "तो", "تم": "तुम", "تمہیں": "तुम्हें", "تمہارا": "तुम्हारा", "تمہاری": "तुम्हारी",
    "اس": "इस", "اسے": "इसे", "اسکو": "इसको", "اسکی": "इसकी", "اسکے": "इसके", "اسکا": "इसका", "اسمیں": "इसमें",
    "ان": "इन", "انہیں": "इन्हें", "انکو": "इनको", "انکی": "इनकी", "انکے": "इनके", "انکا": "इनका",
    "یہ": "यह", "یہاں": "यहाँ", "وہ": "वह", "وہاں": "वहाँ",
    "کو": "को", "کا": "का", "کی": "की", "کے": "के", "سے": "से", "پر": "पर", "تک": "तक", "میں": "में",

    # Conjunctions & Interrogatives
    "اور": "और", "یا": "या", "کہ": "कि", "کیونکہ": "क्योंकि", "لیکن": "लेकिन", "مگر": "मगर", "بلکہ": "बल्कि",
    "اگر": "अगर", "تو": "तो", "تاکہ": "ताकि", "تا": "ता", "چونکہ": "चूँकि", "حالانکہ": "हालाँकि",
    "کیا": "क्या", "کیوں": "क्यों", "کیسے": "कैसे", "کیسی": "कैसी", "کیسا": "कैसा",
    "کہاں": "कहाँ", "کب": "कब", "کون": "कौन", "کتنا": "कितना", "کتنی": "कितनी", "کتنے": "कितने",

    # Common Verbs & Auxiliaries
    "ہے": "है", "ہیں": "हैं", "ہو": "हो", "ہوں": "हूँ", "ہوتا": "होता", "ہوتی": "होती", "ہوتے": "होते",
    "تھا": "था", "تھی": "थी", "تھے": "थे", "تھیں": "थीं",
    "گا": "गा", "گی": "गी", "گے": "गे",
    "کر": "कर", "کرنا": "करना", "کرنے": "करने", "کرنی": "करनी", "کرتا": "करता", "کرتی": "करती", "کرتے": "करते",
    "کرو": "करो", "کریں": "करें", "کروں": "करूँ", "کررہا": "कर रहा", "کررہے": "कर रहे", "کررہی": "कर रही",
    "ہوگا": "होगा", "ہوگی": "होगी", "ہونگے": "होंगे", "ہوئے": "हुए", "ہوئی": "हुई", "ہوا": "हुआ",
    "سکتا": "सकता", "سکتی": "सकती", "سکتے": "सकते", "سکیں": "सकें",
    "چاہتا": "चाहता", "چاہتی": "चाहती", "چاہتے": "चाहते", "چاہیے": "चाहिए", "چاہے": "चाहे",
    "دیکھ": "देख", "دیکھنا": "देखना", "دیکھنے": "देखने", "دیکھتے": "देखते", "دیکھیں": "देखें", "دیکھو": "देखो",
    "دیکھا": "देखा", "دیکھائی": "दिखाई", "دکھاتا": "दिखाता", "دکھاتے": "दिखाते", "دکھائے": "दिखाए",
    "جان": "जान", "جاننا": "जानना", "جانیں": "जानें", "جانتے": "जानते",
    "بن": "बन", "بنانا": "बनाना", "بنانے": "बनाने", "بنائیں": "बनाएँ", "بنایا": "बनाया", "بناؤ": "बनाओ", "بناؤں": "बनाऊँ",
    "لے": "ले", "لینا": "लेना", "لینے": "लेने", "لیتے": "लेते", "لیں": "लें", "لیا": "लिया",
    "دے": "दे", "دینا": "देना", "دینے": "देने", "دیتے": "देते", "دیں": "दें", "دیا": "दिया", "دینہ": "देना",
    "آ": "आ", "آنا": "आना", "آتے": "आते", "آئیں": "आएँ", "آیا": "आया", "آئی": "आई", "آئے": "आए",
    "جا": "जा", "جانا": "जाना", "جاتے": "जाते", "جائیں": "जाएँ", "گیا": "गया", "گئی": "गई", "گئے": "गए",
    "بول": "बोल", "بولنا": "बोलना", "بولتے": "बोलते", "بولا": "बोला",
    "پڑھ": "पढ़", "پڑھنا": "पढ़ना", "پڑھا": "पढ़ा",
    "لکھ": "लिख", "لکھنا": "लिखना", "لکھتے": "लिखते", "لکھا": "लिखा", "لکھیں": "लिखें",
    "سمجھ": "समझ", "سمجھنا": "समझना", "سمجھتے": "समझते", "سمجھایا": "समझाया",
    "ڈال": "डाल", "ڈالنا": "डालना", "ڈالتے": "डालते", "ڈالا": "डाला",
    "کھول": "खोल", "کھولنا": "खोलना", "کھولیں": "खोलें", "کھولا": "खोला",

    # Common Nouns & Adjectives
    "لوگ": "लोग", "لوگوں": "लोगों", "بات": "बात", "باتیں": "बातें", "چیز": "चीज़", "چیزیں": "चीज़ें",
    "کام": "काम", "نام": "नाम", "طرح": "तरह", "طریقہ": "तरीक़ा", "طریقے": "तरीक़े",
    "ایک": "एक", "دو": "दो", "تین": "तीन", "چار": "चार", "پانچ": "पाँच", "چھ": "छह",
    "پہلا": "पहला", "پہلی": "पहली", "پہلے": "पहले", "دوسرا": "दूसरा", "دوسری": "दूसरी", "دوسرے": "दूसरे",
    "بہت": "बहुत", "زیادہ": "ज़्यादा", "کم": "कम", "اچھا": "अच्छा", "اچھی": "अच्छी", "اچھے": "अच्छे",
    "بڑا": "बड़ा", "بڑی": "बड़ी", "بڑے": "बड़े", "چھوٹا": "छोटा", "چھوٹی": "छोटी", "چھوٹے": "छोटे",
    "نیا": "नया", "نئی": "नई", "نئے": "नए", "پرانا": "पुराना",
    "سب": "सब", "سبھی": "सभी", "سارے": "सारे", "ساری": "सारी", "سارا": "सारा",
    "صرف": "सिर्फ़", "بھی": "भी", "نہیں": "नहीं", "مت": "मत", "نہ": "न",
    "اب": "अब", "ابھی": "अभी", "پھر": "फिर", "دوبارہ": "दोबारा", "ہمیشہ": "हमेशा",
    "نیچے": "नीचे", "اوپر": "ऊपर", "اندر": "अंदर", "باہر": "बाहर", "سامنے": "सामने", "پیچھے": "पीछे",

    # Tech & Course Specific Words in Urdu Script
    "ویڈیو": "वीडियो", "ویڈیوز": "वीडियोज़", "ویب": "वेब", "سائٹ": "साइट", "ویب سائٹ": "वेबसाइट",
    "پیج": "पेज", "پیجز": "पेजेस", "فائل": "फ़ाइल", "فائلیں": "फ़ाइलें",
    "کوڈ": "कोड", "ٹیگ": "टैग", "ٹیگز": "टैग्स", "لنک": "लिंक", "امیج": "इमेज",
    "آڈیو": "ऑडियो", "میڈیا": "मीडिया", "ایلیمنٹ": "एलिमेंट", "ایلیمنٹس": "एलिमेंट्स",
    "براؤزر": "ब्राउज़र", "فولڈر": "फ़ोल्डर", "اسکرین": "स्क्रीन", "کمپیوٹر": "कंप्यूटर",
    "انٹرنیٹ": "इंटरनेट", "سرور": "सर्वर", "گوگل": "गूगल", "رپورٹ": "रिपोर्ट",
    "سائز": "साइज़", "کلر": "कलर", "ٹیکسٹ": "टेक्स्ट", "ہیڈنگ": "हेडिंग",
    "پیراگراف": "पैराग्राफ", "بٹن": "बटन", "فارم": "फ़ॉर्म", "انپٹ": "इनपुट",
    "ویلیو": "वैल्यू", "پراپرٹی": "प्रॉपर्टी", "کلاس": "क्लास", "آئی ڈی": "आईडी",
    "سگما": "सिग्मा", "بیچ": "बैच", "کورس": "कोर्स", "دوستو": "दोस्तों", "بھائیو": "भाइयों",
    "کمنٹ": "कमेंट", "لائک": "लाइक", "شیئر": "शेयर", "سبسکرائب": "सब्सक्राइब", "چینل": "चैनल",
    "ضرور": "ज़रूर", "شکریہ": "शुक्रिया", "خوش": "खुश", "امید": "उम्मीद", "پتہ": "पता",
    "کس": "किस", "کیس": "किस", "کور": "कोर", "ویٹلز": "वाइटल्स", "وائٹلز": "वाइटल्स",
    "بانے": "बारे", "بارے": "बारे", "نیچے": "नीचे", "نیچھے": "नीचे", "سکچن": "सेक्शन",
    "سیکشن": "सेक्शन", "لک": "लिख", "لکھ": "लिख", "لیے": "लिए", "لئے": "लिए", "لیئے": "लिए",
    "اوپر": "ऊपर", "اپر": "ऊपर", "اون": "ऑन", "سرچ": "सर्च", "انجن": "इंजन",
    "ایسٹیامل": "HTML", "ایچھٹیامل": "HTML", "استعمال": "इस्तेमाल", "استمریکی": "इस्तेमाल",
    "امریک": "हम लोग", "بندہ": "बंदा", "بندا": "बंदा", "رکھنا": "रखना", "رکھیے": "रखिए",
    "پڑھیں": "पढ़ें", "پڑھا": "पढ़ा", "دکھایا": "दिखाया", "دیکھایا": "दिखाया",
}

# Phonetic character transliteration table for unmapped Urdu words
CHAR_MAP = {
    'ا': 'अ', 'آ': 'आ', 'ب': 'ब', 'پ': 'प', 'ت': 'त', 'ٹ': 'ट', 'ث': 'स',
    'ج': 'ज', 'چ': 'च', 'ح': 'ह', 'خ': 'ख़', 'د': 'द', 'ڈ': 'ड', 'ذ': 'ज़',
    'ر': 'र', 'ڑ': 'ड़', 'ز': 'ज़', 'ژ': 'झ़', 'س': 'स', 'ش': 'श', 'ص': 'स',
    'ض': 'ज़', 'ط': 'त', 'ظ': 'ज़', 'ع': 'अ', 'غ': 'ग़', 'ف': 'फ़', 'ق': 'क़',
    'ک': 'क', 'گ': 'ग', 'ل': 'ल', 'م': 'म', 'ن': 'न', 'ں': 'ं', 'و': 'व',
    'ہ': 'ह', 'ھ': 'ह', 'ء': '', 'ی': 'ी', 'ے': 'े', 'ئ': 'इ'
}

# Known foreign hallucination fixes
HALLUCINATION_FIXES = [
    # Korean artifacts
    (r"aujourdکیس\s*video", "आज के इस video"),
    (r"aujourd\b", "आज"),
    (r"이\s*video", "इस video"),
    (r"이\s*", "इस "),
    (r"MAS게", "Lighthouse"),
    (r"ч\s*", ""),
    (r"들은\s*nomination", ""),
    (r"ignonastrum\s*vs\s*qbop", ""),
    (r"Proporto", ""),
    (r"버\b|터\b", ""),

    # Cyrillic artifacts in tech speech
    (r"\bкопи\b", "copy"),
    (r"\bкопی\b", "copy"),
    (r"\bМیکسофт\b|\bمیکسоفٹ\b", "Microsoft"),
    (r"\bтак\s*ki\b|\bтак\b", "ताकि"),
    (r"\bбанда\b|\bبنда\b", "बंदा"),
    (r"\bди\b|\bда\b", "था"),
    (r"\bон\b", "on"),
    (r"\bс\b", "s"),
    (r"\bк\b", "k"),
    (r"\bо\b", "o"),
    (r"\bп\b", "p"),

    # Chinese/Greek random artifacts on noise
    (r"vindiah获å", "video"),
    (r"[获这是你们]", ""),
    (r"[ους]", ""),

    # Common Whisper Hindustani misspellings
    (r"\bchuko\s*here\s*upper\s*afnika\s*buddhas\s*screen\b", "chuka hoon yahan par computer screen"),
    (r"\bgambudress\s*screen\b", "computer screen"),
    (r"\blet's\s*roll\s*the\s*intro\b", "let's roll the intro"),
    (r"\bHTML\s*5\b", "HTML5"),
    (r"\bVS\s*Code\b", "VS Code"),
]


def transliterate_word(urdu_word: str) -> str:
    """Translate or transliterate a single Urdu word into Devanagari."""
    if urdu_word in URDU_TO_DEVANAGARI:
        return URDU_TO_DEVANAGARI[urdu_word]
    
    # Try stripping diacritics / trailing affixes
    base = re.sub(r"[\u064B-\u065F]", "", urdu_word)
    if base in URDU_TO_DEVANAGARI:
        return URDU_TO_DEVANAGARI[base]

    # Character-by-character transliteration fallback
    out = []
    for ch in base:
        out.append(CHAR_MAP.get(ch, ch))
    res = "".join(out)
    return res if res else urdu_word


def analyze_scripts(text: str) -> Tuple[Dict[str, int], int]:
    counts = {
        "devanagari": len(re.findall(r"[\u0900-\u097F]", text)),
        "perso_arabic_urdu": len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", text)),
        "korean": len(re.findall(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", text)),
        "cyrillic": len(re.findall(r"[\u0400-\u04FF]", text)),
        "latin": len(re.findall(r"[a-zA-Z]", text)),
        "digits_punct": len(re.findall(r"[0-9\s.,!?:;\"'()\[\]{}<>/\\-_+=*&^%$#@~`|]", text)),
    }
    total = len(text)
    return counts, total


def clean_text_hindi_english(text: str) -> Tuple[str, bool]:
    """
    Clean transcript text, replacing Urdu script with Devanagari Hindi,
    correcting foreign hallucinations, and preserving English/Hinglish.
    Returns (cleaned_text, had_corrections).
    """
    if not text:
        return "", False

    original = text
    cleaned = text

    # 1. Apply hallucination fixes
    for pattern, replacement in HALLUCINATION_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 2. Transliterate Urdu words
    def replace_urdu(match):
        w = match.group(0)
        return transliterate_word(w)

    cleaned = re.sub(r"[\u0600-\u06FF]+", replace_urdu, cleaned)

    # 3. Clean up residual isolated non-Hindi, non-English junk
    cleaned = re.sub(r"[^\x00-\x7F\u0900-\u097F\s.,!?:;\"'()\[\]{}<>/\\-_+=*&^%$#@~`|]", "", cleaned)

    # 4. Collapse spaces
    cleaned = re.sub(r" +", " ", cleaned).strip()

    had_corrections = (cleaned != original)
    return cleaned, had_corrections


def clean_transcript_file(
    input_path: Path,
    output_path: Path
) -> Dict[str, Any]:
    """
    Clean an entire transcript file, preserving all timings, segment IDs,
    and metadata while converting text to clean Hindi (Devanagari) + English.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    cleaned_segments = []
    corrections_count = 0

    for seg in segments:
        new_seg = dict(seg)
        txt = seg.get("text", "")
        c_txt, changed = clean_text_hindi_english(txt)
        new_seg["text"] = c_txt
        if changed:
            corrections_count += 1
            new_seg["was_script_corrected"] = True
        cleaned_segments.append(new_seg)

    data["segments"] = cleaned_segments
    data["full_transcript"] = " ".join(s["text"] for s in cleaned_segments)
    data["language_support"] = "Hindi (Devanagari) + English / Hinglish"
    data["total_segments_corrected"] = corrections_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "file": output_path.name,
        "total_segments": len(cleaned_segments),
        "corrected_segments": corrections_count,
        "corrected_pct": round((corrections_count / len(cleaned_segments)) * 100, 2) if cleaned_segments else 0
    }

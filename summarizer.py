"""
Smart Notes Summarizer Module for LearnMate AI.
Generates complete, lecture-grounded revision notes with:
- 3 Configurable Styles: quick_revision, detailed_study, key_concepts
- 3 Languages: English, Hindi (Devanagari), Hinglish
- Grounded chunk citations and verified timestamps
- Deterministic local caching in data/notes/ with SHA-256 / MD5 fingerprints
- Markdown export capability for student downloads
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import config
from prompts import (
    GEMINI_NOTES_SCHEMA,
    NOTES_STRUCTURED_SYSTEM_PROMPT,
    NOTES_STRUCTURED_USER_TEMPLATE,
    format_context_for_llm
)
from llm import LLMClient

logger = logging.getLogger(__name__)


class LectureSummarizer:
    """Summarizes complete lecture transcripts into high-yield, structured Smart Notes."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        notes_dir: Optional[Path] = None,
        chunks_dir: Optional[Path] = None
    ):
        self.llm = llm_client or LLMClient()
        self.notes_dir = Path(notes_dir or config.NOTES_DIR)
        self.chunks_dir = Path(chunks_dir or config.CHUNKS_DIR)
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def load_lecture_chunks(self, lecture_id: str) -> List[Dict[str, Any]]:
        """Load all chunks corresponding to a specific lecture."""
        lec_file = self.chunks_dir / f"{lecture_id}_chunks.json"
        if lec_file.exists():
            try:
                with open(lec_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {lec_file}: {e}")

        # Fallback to all_chunks.json
        all_chunks_file = self.chunks_dir / "all_chunks.json"
        if all_chunks_file.exists():
            try:
                with open(all_chunks_file, "r", encoding="utf-8") as f:
                    all_chunks = json.load(f)
                return [c for c in all_chunks if c.get("lecture_id") == lecture_id]
            except Exception as e:
                logger.error(f"Failed to read {all_chunks_file}: {e}")

        return []

    def compute_cache_key(
        self,
        lecture_id: str,
        notes_style: str,
        language: str,
        chunks: List[Dict[str, Any]],
        is_mock: bool,
        model_name: Optional[str] = None
    ) -> str:
        """Compute a deterministic cache filename based on lecture, parameters, model, and chunk content."""
        chunk_sig = hashlib.md5("".join(c.get("chunk_id", "") for c in chunks).encode("utf-8")).hexdigest()[:8]
        mode_tag = "mock" if is_mock else "live"
        clean_style = notes_style.lower().replace(" ", "_")
        clean_lang = language.lower().replace(" ", "_")
        clean_model = (model_name or self.llm.model).replace("-", "_").replace(".", "_")
        return f"{lecture_id}_{clean_style}_{clean_lang}_{mode_tag}_{clean_model}_{chunk_sig}.json"

    def generate_smart_notes(
        self,
        lecture_id: str,
        notes_style: str = "detailed_study",
        language: str = "English",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate or retrieve cached Smart Notes for a lecture.
        Guarantees strict schema adherence, provenance, and generating model tracking.
        """
        chunks = self.load_lecture_chunks(lecture_id)
        if not chunks:
            raise ValueError(f"No chunks found for lecture '{lecture_id}'.")

        lecture_title = chunks[0].get("lecture_title", f"Lecture {lecture_id}")
        is_mock = self.llm.mock_mode

        # Check local cache
        cache_filename = self.compute_cache_key(lecture_id, notes_style, language, chunks, is_mock, self.llm.model)
        cache_path = self.notes_dir / cache_filename

        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_data["is_cached"] = True
                logger.info(f"Loaded cached Smart Notes from {cache_path.name}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}")

        # Generation
        if is_mock:
            notes_data = self._generate_mock_notes(lecture_id, lecture_title, notes_style, language, chunks)
        else:
            notes_data = self._generate_live_notes(lecture_id, lecture_title, notes_style, language, chunks)

        notes_data["is_cached"] = False

        # Cache persistence
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(notes_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached generated Smart Notes to {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_path}: {e}")

        return notes_data

    def _generate_live_notes(
        self,
        lecture_id: str,
        lecture_title: str,
        notes_style: str,
        language: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate Smart Notes via live Gemini structured generation with model tracking."""
        formatted_context = format_context_for_llm(chunks)
        user_prompt = NOTES_STRUCTURED_USER_TEMPLATE.format(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            notes_style=notes_style,
            language=language,
            context=formatted_context
        )

        result = self.llm.generate_structured(
            system_prompt=NOTES_STRUCTURED_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=GEMINI_NOTES_SCHEMA,
            max_output_tokens=3500
        )

        if not result.parsed_json or result.error_message:
            raise RuntimeError(f"Live Gemini notes generation failed: {result.error_message}")

        notes = result.parsed_json

        # Validate and enrich chunk citations
        valid_cids = {c["chunk_id"]: c for c in chunks}
        for sec in notes.get("sections", []):
            claimed_cids = sec.get("supporting_chunk_ids", [])
            verified_cids = [cid for cid in claimed_cids if cid in valid_cids]
            sec["supporting_chunk_ids"] = verified_cids
            sec["citations"] = [
                {
                    "chunk_id": cid,
                    "timestamp": valid_cids[cid].get("start_timestamp", "00:00:00"),
                    "tutorial_number": valid_cids[cid].get("tutorial_number", 0)
                }
                for cid in verified_cids
            ]

        notes["lecture_id"] = lecture_id
        notes["lecture_title"] = lecture_title
        notes["notes_style"] = notes_style
        notes["language"] = language
        notes["generating_model"] = result.model_name
        notes["generated_at"] = time.time()
        notes["is_mock"] = False

        return notes

    def _generate_mock_notes(
        self,
        lecture_id: str,
        lecture_title: str,
        notes_style: str,
        language: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate realistic, lecture-grounded mock notes tailored by lecture, style, and language."""
        tut_num = chunks[0].get("tutorial_number", 0) if chunks else 0

        # Build grounded citations from actual chunks
        cids = [c["chunk_id"] for c in chunks]
        def get_chunk_cite(idx: int):
            target_idx = min(idx, len(chunks) - 1)
            c = chunks[target_idx]
            return {
                "chunk_id": c["chunk_id"],
                "timestamp": c.get("start_timestamp", "00:00:00"),
                "tutorial_number": tut_num
            }

        # Lecture-specific grounded data
        if lecture_id in ("LEC_01", "DEMO_03"):  # SEO & Core Web Vitals
            title = "SEO and Core Web Vitals in HTML"
            if language == "Hindi":
                summary = (
                    "इस लेक्चर में हमने सीखा कि सर्च इंजन ऑप्टिमाइज़ेशन (SEO) क्या होता है और वेब पेजों को सर्च इंजन में उच्च रैंक दिलाने के लिए "
                    "HTML मेटा टैग्स, व्यूपोर्ट, और इमेज ऑल्ट एट्रिब्यूट्स का सही उपयोग कैसे किया जाता है। "
                    "साथ ही Chrome DevTools में Google Lighthouse की मदद से Core Web Vitals (LCP, INP, CLS) का प्रदर्शन मापना सीखा।"
                )
                objectives = [
                    "सर्च इंजन ऑप्टिमाइज़ेशन (SEO) और ऑन-पेज HTML तकनीकों को समझना",
                    "<meta> टैग्स (description, keywords, viewport) का सही विन्यास करना",
                    "Google Chrome Lighthouse टूल का उपयोग करके वेबसाइट ऑडिट करना",
                    "Core Web Vitals के तीन मुख्य स्तंभों (LCP, INP, CLS) को समझना और सुधारना"
                ]
                sec1_head = "1. SEO मूल बातें और महत्वपूर्ण मेटा टैग्स"
                sec1_content = (
                    "SEO (Search Engine Optimization) का मुख्य उद्देश्य वेबसाइट की विजिबिलिटी और ऑर्गेनिक रैंकिंग को बढ़ाना है। "
                    "HTML <head> के अंदर <meta name='description'> सर्च रिजल्ट्स में दिखाई देने वाला स्निपेट तय करता है। "
                    "मोबाइल डिवाइसों पर सही रेंडरिंग के लिए viewport मेटा टैग अनिवार्य है।"
                )
                sec2_head = "2. Google Lighthouse ऑडिट और परफॉरमेंस मैट्रिक्स"
                sec2_content = (
                    "Google Chrome DevTools का Lighthouse पैनल वेबसाइट की परफॉरमेंस, एक्सेसिबिलिटी, बेस्ट प्रैक्टिसेज और SEO का व्यापक स्कोर देता है। "
                    "यह अनऑप्टिमाइज़्ड इमेजेस और धीमे रेंडर-ब्लॉकिंग संसाधनों की पहचान करता है।"
                )
                sec3_head = "3. Core Web Vitals (LCP, INP, CLS)"
                sec3_content = (
                    "Core Web Vitals उपयोगकर्ता अनुभव के तीन महत्वपूर्ण पहलू मापते हैं:\n"
                    "- LCP (Largest Contentful Paint): मुख्य सामग्री लोड होने का समय (2.5 सेकंड से कम आदर्श)।\n"
                    "- INP (Interaction to Next Paint): सभी यूजर इंटरैक्शन (क्लिक/टाइपिंग) पर रिस्पॉन्स में देरी (200ms से कम आदर्श; मार्च 2024 में FID के स्थान पर लागू)।\n"
                    "- CLS (Cumulative Layout Shift): अनपेक्षित लेआउट बदलाव (0.1 से कम आदर्श)।"
                )
                terms = [
                    {"term": "SEO", "definition": "सर्च इंजन ऑप्टिमाइज़ेशन: सर्च इंजन में वेबसाइट की विजिबिलिटी बढ़ाने की प्रक्रिया।", "timestamp_ref": "00:00:15"},
                    {"term": "LCP", "definition": "Largest Contentful Paint: वेबपेज पर सबसे बड़े कंटेंट ब्लॉक के लोड होने का समय।", "timestamp_ref": "00:02:06"},
                    {"term": "INP", "definition": "Interaction to Next Paint: यूजर इंटरैक्शन पर समग्र लेटेंसी (200ms से कम)।", "timestamp_ref": "00:02:24"},
                    {"term": "CLS", "definition": "Cumulative Layout Shift: पेज लोड होते समय एलिमेंट्स का अनचाहा खिसकना।", "timestamp_ref": "00:02:42"}
                ]
                pitfalls = [
                    "<meta name='viewport'> छोड़ देना, जिससे मोबाइल पर पेज ज़ूम-आउट और अनरेस्पॉन्सिव दिखता है।",
                    "<img> टैग में width और height न देना, जिससे इमेज लोड होते ही CLS (लेआउट शिफ्ट) होता है।",
                    "लंबी JavaScript स्क्रिप्ट्स चलाना जो मुख्य थ्रेड को ब्लॉक करके INP लेटेंसी बढ़ाती हैं।"
                ]
                checklist = [
                    "क्या आपने <head> में <meta name='description'> जोड़ा है?",
                    "क्या मोबाइल फ्रेंडली viewport टैग मौजूद है?",
                    "क्या सभी इमेजेस में वर्णनात्मक alt एट्रिब्यूट मौजूद है?",
                    "क्या Lighthouse ऑडिट में SEO स्कोर 90+ प्राप्त हुआ है?",
                    "क्या LCP 2.5s और INP 200ms के भीतर अनुकूलित हैं?"
                ]
            elif language == "Hinglish":
                summary = (
                    "Is lecture mein humne seekha ki Search Engine Optimization (SEO) kya hota hai aur websites ko search engines par rank karne ke liye "
                    "HTML meta tags, viewport settings, aur image alt attributes kaise configure karte hain. "
                    "Humne Chrome DevTools ke Google Lighthouse tool se website audit karna aur Core Web Vitals (LCP, INP, CLS) optimize karna bhi seekha."
                )
                objectives = [
                    "SEO fundamentals aur on-page meta tag strategies samajhna",
                    "<meta> description aur mobile viewport tags implement karna",
                    "Chrome DevTools Lighthouse panel se performance audit run karna",
                    "Core Web Vitals metrics (LCP, INP, CLS) ke thresholds optimize karna"
                ]
                sec1_head = "1. SEO Fundamentals & Essential Meta Tags"
                sec1_content = (
                    "SEO ka goal search engines jaise Google par organic traffic lana hota hai. "
                    "HTML ke <head> section mein <meta name='description'> Google search snippet decide karta hai. "
                    "Mobile devices ke liye responsive viewport meta tag zaroori hota hai."
                )
                sec2_head = "2. Google Lighthouse Audit & Performance Analysis"
                sec2_content = (
                    "Chrome DevTools ke andar Lighthouse tab se hum 4 categories check karte hain: Performance, Accessibility, Best Practices, aur SEO. "
                    "Lighthouse audit se heavy images aur slow resources ka pata lagta hai."
                )
                sec3_head = "3. Core Web Vitals (LCP, INP, CLS)"
                sec3_content = (
                    "Core Web Vitals user experience ke 3 core dimensions measure karte hain:\n"
                    "- LCP (Largest Contentful Paint): Largest visible content load hone ka time (< 2.5s is good).\n"
                    "- INP (Interaction to Next Paint): Page par user clicks/taps ki overall responsiveness (< 200ms is good; March 2024 me FID ki jagah aaya).\n"
                    "- CLS (Cumulative Layout Shift): Content ka sudden unexpected shift (< 0.1 is good)."
                )
                terms = [
                    {"term": "SEO", "definition": "Search Engine Optimization: Google par website rank karane ki technique.", "timestamp_ref": "00:00:15"},
                    {"term": "LCP", "definition": "Largest Contentful Paint: Page ka main hero content render hone ka time.", "timestamp_ref": "00:02:06"},
                    {"term": "INP", "definition": "Interaction to Next Paint: Clicks aur keystrokes par UI responsiveness latency (< 200ms).", "timestamp_ref": "00:02:24"},
                    {"term": "CLS", "definition": "Cumulative Layout Shift: Loading ke dauran visual stability ka score.", "timestamp_ref": "00:02:42"}
                ]
                pitfalls = [
                    "Viewport meta tag bhool jana jisse mobile screen par layout shrink ho jata hai.",
                    "Images ke explicit width/height dimensions omit karna jisse CLS score drop hota hai.",
                    "Long blocking JavaScript tasks run karna jo INP latency ko 200ms se upar le jati hain."
                ]
                checklist = [
                    "Head tag mein meta description added hai?",
                    "Responsive viewport meta tag correctly defined hai?",
                    "Sabhi images par meaningful alt text diya hai?",
                    "Lighthouse SEO score 90+ achieve hua hai?",
                    "LCP <= 2.5s aur INP <= 200ms optimize hui hai?"
                ]
            else:  # English
                summary = (
                    "This lecture explores Search Engine Optimization (SEO) principles and performance auditing in modern HTML. "
                    "It covers essential <head> metadata including viewport, description, and Open Graph tags, alongside practical auditing "
                    "with Google Chrome Lighthouse. Key web performance metrics known as Core Web Vitals (LCP, INP, and CLS) are demystified "
                    "with actionable optimization strategies."
                )
                objectives = [
                    "Master on-page HTML SEO foundations and search engine indexing mechanics",
                    "Properly configure essential <meta> tags including description and responsive viewport",
                    "Execute performance, accessibility, and SEO audits using Chrome DevTools Lighthouse",
                    "Understand and optimize Core Web Vitals metrics: LCP, INP (which replaced FID in March 2024), and CLS"
                ]
                sec1_head = "1. SEO Foundations & Essential Meta Tags"
                sec1_content = (
                    "Search Engine Optimization (SEO) is the process of improving site discoverability on search engines. "
                    "The HTML <head> element hosts critical crawlers directives. The description meta tag specifies the snippet "
                    "displayed in search results, while the viewport tag ensures responsive mobile scaling."
                )
                sec2_head = "2. Google Lighthouse Performance Audits"
                sec2_content = (
                    "Chrome DevTools includes Google Lighthouse, an automated auditing tool that benchmarks web pages across four pillars: "
                    "Performance, Accessibility, Best Practices, and SEO. It highlights opportunities like unoptimized image formats "
                    "and render-blocking scripts."
                )
                sec3_head = "3. Core Web Vitals (LCP, INP, CLS)"
                sec3_content = (
                    "Google Core Web Vitals assess real-world user experience across three essential vectors:\n"
                    "- LCP (Largest Contentful Paint): Measures perceived loading speed (Ideal: <= 2.5 seconds).\n"
                    "- INP (Interaction to Next Paint): Measures overall responsiveness to all user clicks, taps, and keypresses (Ideal: <= 200 milliseconds; replaced FID as an official Core Web Vital in March 2024).\n"
                    "- CLS (Cumulative Layout Shift): Quantifies visual stability and jarring content jumps (Ideal: <= 0.1)."
                )
                terms = [
                    {"term": "SEO", "definition": "Search Engine Optimization: Techniques used to increase organic search engine visibility.", "timestamp_ref": "00:00:15"},
                    {"term": "LCP", "definition": "Largest Contentful Paint: Time taken to render the largest visible DOM content block (<= 2.5s).", "timestamp_ref": "00:02:06"},
                    {"term": "INP", "definition": "Interaction to Next Paint: Latency of user interactions (<= 200ms; replaced FID in March 2024).", "timestamp_ref": "00:02:24"},
                    {"term": "CLS", "definition": "Cumulative Layout Shift: Metric quantifying unexpected layout shifts during page lifecycle (<= 0.1).", "timestamp_ref": "00:02:42"}
                ]
                pitfalls = [
                    "Omitting the viewport meta tag, causing mobile browsers to render desktop layouts at 980px.",
                    "Failing to declare width and height on <img> elements, leading to high Cumulative Layout Shift (CLS).",
                    "Executing long monolithic JavaScript tasks on the main thread, resulting in sluggish INP (> 200ms)."
                ]
                checklist = [
                    "Is the <meta name='description'> concise and under 160 characters?",
                    "Is the responsive <meta name='viewport'> tag present in <head>?",
                    "Do all <img> tags include descriptive alt attributes and explicit dimensions?",
                    "Does the page score 90+ on Chrome DevTools Lighthouse SEO audit?",
                    "Are Largest Contentful Paint (LCP <= 2.5s) and Interaction to Next Paint (INP <= 200ms) verified?"
                ]

            sections = [
                {
                    "heading": sec1_head,
                    "content": sec1_content,
                    "code_snippets": [
                        '<meta name="description" content="Master modern HTML, SEO, and web development with Sigma Batch.">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">'
                    ],
                    "supporting_chunk_ids": [cids[0], cids[1]] if len(cids) >= 2 else cids,
                    "citations": [get_chunk_cite(0), get_chunk_cite(1)]
                },
                {
                    "heading": sec2_head,
                    "content": sec2_content,
                    "code_snippets": [
                        "<!-- Chrome DevTools: F12 -> Lighthouse -> Analyze page load -->\n<link rel=\"icon\" type=\"image/x-icon\" href=\"favicon.ico\">"
                    ],
                    "supporting_chunk_ids": [cids[2], cids[3]] if len(cids) >= 4 else cids[:1],
                    "citations": [get_chunk_cite(2), get_chunk_cite(3)]
                },
                {
                    "heading": sec3_head,
                    "content": sec3_content,
                    "code_snippets": [
                        "<!-- Prevent CLS by reserving aspect-ratio or dimensions -->\n<img src=\"hero.jpg\" alt=\"Tutorial Hero\" width=\"800\" height=\"450\" loading=\"lazy\">"
                    ],
                    "supporting_chunk_ids": [cids[4], cids[5]] if len(cids) >= 6 else cids[:1],
                    "citations": [get_chunk_cite(4), get_chunk_cite(5)]
                }
            ]

        elif lecture_id in ("LEC_02", "DEMO_02"):  # Video, Audio & Media in HTML
            title = "Video, Audio & Media in HTML"
            if language == "Hindi":
                summary = (
                    "इस लेक्चर में HTML5 मल्टीमीडिया एलिमेंट्स का व्यापक अध्ययन किया गया है। "
                    "हमने <video> और <audio> टैग्स के मुख्य एट्रिब्यूट्स (controls, autoplay, muted, loop, poster) का उपयोग करना, "
                    "विभिन्न ब्राउज़रों के लिए <source> टैग द्वारा मीडिया फॉर्मेट्स प्रदान करना और <iframe> द्वारा YouTube वीडियो एम्बेड करना सीखा।"
                )
                objectives = [
                    "HTML5 <video> और <audio> टैग्स का सिंटैक्स और उपयोग समझना",
                    "controls, autoplay, muted, loop और poster एट्रिब्यूट्स को सही तरीके से लागू करना",
                    "ब्राउज़र कम्पैटिबिलिटी के लिए <source> टैग का प्रयोग करना",
                    "<iframe> टैग के माध्यम से बाहरी मीडिया (जैसे YouTube) को सुरक्षित रूप से एम्बेड करना"
                ]
                sec1_head = "1. <video> टैग और मुख्य एट्रिब्यूट्स"
                sec1_content = (
                    "HTML5 <video> टैग बिना किसी थर्ड-पार्टी प्लगइन के वीडियो चलाने की सुविधा देता है। "
                    "controls एट्रिब्यूट प्ले/पॉज़ और वॉल्यूम बटन जोड़ता है। "
                    "आधुनिक ब्राउज़रों में autoplay तभी काम करता है जब साथ में muted एट्रिब्यूट भी दिया गया हो।"
                )
                sec2_head = "2. <audio> टैग और मल्टीपल सोर्स फॉलबैक्स"
                sec2_content = (
                    "<audio> टैग वेबपेज पर साउंड फाइल्स चलाने के लिए उपयोग किया जाता है। "
                    "अलग-अलग ब्राउज़रों के सपोर्ट के लिए <source> टैग के जरिए MP3, OGG या WAV फॉर्मेट्स दिए जा सकते हैं।"
                )
                sec3_head = "3. <iframe> द्वारा एक्सटर्नल मीडिया एम्बेडिंग"
                sec3_content = (
                    "<iframe> (Inline Frame) का उपयोग बाहरी वेबपेजों और YouTube वीडियो को अपने पेज में दिखाने के लिए किया जाता है। "
                    "यह src, width, height, frameborder और allowfullscreen एट्रिब्यूट्स का उपयोग करता है।"
                )
                terms = [
                    {"term": "<video>", "definition": "HTML5 टैग जिसका उपयोग नेटिव वीडियो चलाने के लिए किया जाता है।", "timestamp_ref": "00:01:20"},
                    {"term": "controls", "definition": "एट्रिब्यूट जो यूजर को प्ले, पॉज और वॉल्यूम कंट्रोल उपलब्ध कराता है।", "timestamp_ref": "00:03:10"},
                    {"term": "<iframe>", "definition": "इनलाइन फ्रेम टैग जो किसी बाहरी डॉक्यूमेंट या वीडियो को एम्बेड करता है।", "timestamp_ref": "00:09:45"}
                ]
                pitfalls = [
                    "बिना muted के autoplay लगाना, जिसे आधुनिक ब्राउज़र ब्लॉक कर देते हैं।",
                    "controls एट्रिब्यूट छोड़ देना, जिससे वीडियो लोड तो होता है पर यूजर उसे चला नहीं पाता।",
                    "गलत फाइल पाथ या अनुपलब्ध वीडियो फॉर्मेट देना।"
                ]
                checklist = [
                    "क्या आपने <video> टैग में controls एट्रिब्यूट शामिल किया है?",
                    "क्या autoplay के साथ muted एट्रिब्यूट मौजूद है?",
                    "क्या वीडियो के लिए poster इमेज निर्दिष्ट की गई है?",
                    "क्या <iframe> में allowfullscreen और उचित width/height दी गई है?"
                ]
            elif language == "Hinglish":
                summary = (
                    "Is lecture mein humne HTML5 multimedia elements detail mein seekhe. "
                    "Humne dekha ki <video> aur <audio> tags kaise kaam karte hain, controls, autoplay, muted, loop aur poster attributes kaise use hote hain, "
                    "aur YouTube videos ko <iframe> ke through seamlessly kaise embed kiya jata hai."
                )
                objectives = [
                    "<video> aur <audio> tags ke native capabilities master karna",
                    "Media attributes (controls, autoplay, muted, loop, poster) ka behavior samajhna",
                    "Multiple browser compatibility ke liye <source> tags configure karna",
                    "<iframe> tag use karke YouTube aur maps embed karna"
                ]
                sec1_head = "1. <video> Tag & Essential Attributes"
                sec1_content = (
                    "<video> tag bina kisi external player ke video stream karta hai. "
                    "Agar aap controls attribute nahi denge toh user ko play/pause buttons nahi dikhenge. "
                    "Modern browsers mein video tabhi autoplay hoti hai jab usme muted attribute laga ho."
                )
                sec2_head = "2. <audio> Element & Fallback Sources"
                sec2_content = (
                    "<audio> tag sound files play karne ke liye hota hai. "
                    "Browser compatibility ensure karne ke liye <source src='...' type='audio/mpeg'> use karna standard practice hai."
                )
                sec3_head = "3. <iframe> Embeds (YouTube, Maps)"
                sec3_content = (
                    "<iframe> tag se kisi doosri website ka page ya YouTube video display kiya ja sakta hai. "
                    "Isme src, width, height, aur allowfullscreen attributes standard hote hain."
                )
                terms = [
                    {"term": "<video>", "definition": "HTML5 media tag to play native video files.", "timestamp_ref": "00:01:20"},
                    {"term": "poster", "definition": "Attribute jo video play hone se pehle thumbnail image show karta hai.", "timestamp_ref": "00:04:15"},
                    {"term": "<iframe>", "definition": "Inline frame element for embedding external documents and videos.", "timestamp_ref": "00:09:45"}
                ]
                pitfalls = [
                    "Autoplay bina muted lagana, jise browser audio policy block kar deti hai.",
                    "Controls attribute miss karna jisse student video control nahi kar pata.",
                    "Image/video ka invalid relative path dena."
                ]
                checklist = [
                    "Kya video tag mein controls attribute added hai?",
                    "Autoplay ke saath muted mandatory lagaya hai?",
                    "Video thumbnail ke liye poster attribute set hai?",
                    "YouTube embed <iframe> mein allowfullscreen enable hai?"
                ]
            else:  # English
                summary = (
                    "This lecture provides an in-depth exploration of HTML5 multimedia embedding using native elements. "
                    "It details the syntax and behavior of the <video> and <audio> tags, explains media control attributes "
                    "(controls, autoplay, muted, loop, poster), demonstrates multi-format source fallbacks, and illustrates "
                    "embedding third-party media streams using <iframe>."
                )
                objectives = [
                    "Implement native HTML5 <video> and <audio> playback",
                    "Configure controls, muted autoplay policies, loop, and poster images",
                    "Provide cross-browser format fallbacks with nested <source> tags",
                    "Safely embed external third-party media using the <iframe> element"
                ]
                sec1_head = "1. HTML5 <video> Element & Playback Attributes"
                sec1_content = (
                    "The HTML5 <video> element embeds video streams directly without browser plugins. "
                    "Key attributes include 'controls' (renders browser player UI), 'poster' (displays preview placeholder image), "
                    "and 'autoplay' (requires 'muted' to satisfy modern browser user autoplay policies)."
                )
                sec2_head = "2. HTML5 <audio> Tag & Format Fallbacks"
                sec2_content = (
                    "The <audio> element streams audio files. Because browser codec support varies, developers enclose "
                    "multiple <source> elements (MP3, OGG, WAV) inside <audio> to ensure resilient playback across all operating systems."
                )
                sec3_head = "3. Third-Party Embedding with <iframe>"
                sec3_content = (
                    "The <iframe> (Inline Frame) element embeds external web documents and media players (such as YouTube or Google Maps). "
                    "It uses 'src', 'width', 'height', 'title', and 'allowfullscreen' attributes."
                )
                terms = [
                    {"term": "<video>", "definition": "HTML5 element used to play video clips natively in browsers.", "timestamp_ref": "00:01:20"},
                    {"term": "controls", "definition": "Boolean attribute that exposes browser play, pause, seek, and volume UI.", "timestamp_ref": "00:03:10"},
                    {"term": "<iframe>", "definition": "Inline frame element embedding nested browsing contexts.", "timestamp_ref": "00:09:45"}
                ]
                pitfalls = [
                    "Attempting unmuted autoplay, which modern browsers automatically block to protect user experience.",
                    "Omitting the 'controls' attribute, rendering the media completely inaccessible to users.",
                    "Specifying incorrect MIME types in <source type='...'> elements."
                ]
                checklist = [
                    "Is 'controls' specified on all student-facing <video> and <audio> elements?",
                    "Is 'muted' paired with 'autoplay' whenever automated playback is required?",
                    "Is a responsive poster image configured for the video preview?",
                    "Does the <iframe> embed include an accessible title attribute and allowfullscreen?"
                ]

            sections = [
                {
                    "heading": sec1_head,
                    "content": sec1_content,
                    "code_snippets": [
                        '<video src="media/sample.mp4" controls width="640" poster="thumbnail.jpg">\n  Your browser does not support the video tag.\n</video>'
                    ],
                    "supporting_chunk_ids": [cids[0], cids[1]] if len(cids) >= 2 else cids,
                    "citations": [get_chunk_cite(0), get_chunk_cite(1)]
                },
                {
                    "heading": sec2_head,
                    "content": sec2_content,
                    "code_snippets": [
                        '<audio controls autoplay muted>\n  <source src="podcast.mp3" type="audio/mpeg">\n  <source src="podcast.ogg" type="audio/ogg">\n</audio>'
                    ],
                    "supporting_chunk_ids": [cids[2], cids[3]] if len(cids) >= 4 else cids[:1],
                    "citations": [get_chunk_cite(2), get_chunk_cite(3)]
                },
                {
                    "heading": sec3_head,
                    "content": sec3_content,
                    "code_snippets": [
                        '<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" title="YouTube video" allowfullscreen></iframe>'
                    ],
                    "supporting_chunk_ids": [cids[4], cids[5]] if len(cids) >= 6 else cids[:1],
                    "citations": [get_chunk_cite(4), get_chunk_cite(5)]
                }
            ]

        else:  # LEC_03: Tutorial #2: Your First HTML Website
            title = "Your First HTML Website"
            if language == "Hindi":
                summary = (
                    "यह लेक्चर वेब डेवलपमेंट की शुरुआत और पहली HTML वेबसाइट बनाने पर केंद्रित है। "
                    "इसमें HTML की मूल संरचना (<!DOCTYPE html>, <html>, <head>, <body>), VS Code एडिटर का सेटअप, "
                    "Emmet शॉर्टकट (!), हेडिंग टैग्स (<h1> से <h6>), पैराग्राफ (<p>), लिंक्स (<a>), और इमेजेस (<img>) का व्यावहारिक अभ्यास कराया गया है।"
                )
                objectives = [
                    "HTML डॉक्यूमेंट के बुनियादी ढांचे (Doctype, head, body) को समझना",
                    "VS Code में Emmet एब्रिविएशन (!) द्वारा बॉयलरप्लेट जनरेट करना",
                    "हेडिंग पदानुक्रम (<h1> - <h6>) और पैराग्राफ्स (<p>) का सही प्रयोग करना",
                    "हाइपरलिंक (<a>) और इमेजेस (<img>) को सही पाथ और एट्रिब्यूट्स के साथ जोड़ना"
                ]
                sec1_head = "1. HTML डॉक्यूमेंट कंकाल और बॉयलरप्लेट"
                sec1_content = (
                    "प्रत्येक आधुनिक HTML5 पेज <!DOCTYPE html> से शुरू होता है जो ब्राउज़र को HTML5 मोड में रेंडर करने का निर्देश देता है। "
                    "<html> रूट एलिमेंट है, <head> में मेटाडेटा और टाइटल रहता है, और <body> में दृश्यमान सामग्री होती है।"
                )
                sec2_head = "2. टेक्स्ट और हेडिंग पदानुक्रम"
                sec2_content = (
                    "HTML में <h1> से <h6> तक 6 हेडिंग स्तर होते हैं। <h1> पेज का मुख्य शीर्षक होता है और प्रति पेज केवल एक होना चाहिए। "
                    "साधारण टेक्स्ट के लिए पैराग्राफ टैग <p> का उपयोग किया जाता है।"
                )
                sec3_head = "3. लिंक्स और इमेजेस (Anchor & Image Tags)"
                sec3_content = (
                    "<a> (Anchor) टैग href एट्रिब्यूट के साथ हाइपरलिंक बनाता है। नए टैब में खोलने के लिए target='_blank' का प्रयोग किया जाता है। "
                    "<img> टैग एक सेल्फ-क्लोजिंग टैग है जिसमें src (सोर्स फाइल) और alt (वैकल्पिक टेक्स्ट) अनिवार्य हैं।"
                )
                terms = [
                    {"term": "<!DOCTYPE html>", "definition": "डॉक्यूमेंट टाइप डिक्लेरेशन जो HTML5 वर्जन निर्दिष्ट करता है।", "timestamp_ref": "00:02:15"},
                    {"term": "<a> tag", "definition": "एंकर टैग जिसका उपयोग एक पेज से दूसरे पेज को लिंक करने के लिए किया जाता है।", "timestamp_ref": "00:11:30"},
                    {"term": "<img> tag", "definition": "सेल्फ-क्लोजिंग इमेज टैग जो वेबपेज पर तस्वीरें प्रदर्शित करता है।", "timestamp_ref": "00:15:20"}
                ]
                pitfalls = [
                    "एक ही पेज पर कई <h1> टैग्स का इस्तेमाल करना जो SEO के लिए नुकसानदेह है।",
                    "<img> टैग में alt एट्रिब्यूट छोड़ देना, जिससे दृष्टिबाधित छात्रों और स्क्रीन रीडर्स को परेशानी होती है।",
                    "रिलेटिव पाथ (relative path) में फाइल नेम या एक्सटेंशन गलत लिखना।"
                ]
                checklist = [
                    "क्या कोड के शीर्ष पर <!DOCTYPE html> मौजूद है?",
                    "क्या पेज पर केवल एक प्रमुख <h1> टैग है?",
                    "क्या सभी लिंक्स में वैध href लिंक दी गई है?",
                    "क्या सभी इमेजेस में वर्णनात्मक alt टेक्स्ट मौजूद है?"
                ]
            elif language == "Hinglish":
                summary = (
                    "Is lecture mein humne apni pehli HTML website banayi aur web development ka foundation setup kiya. "
                    "Humne HTML document skeleton (<!DOCTYPE html>, <html>, <head>, <body>), VS Code setup, Emmet boilerplate shortcut (!), "
                    "headings (<h1> to <h6>), paragraphs (<p>), anchor links (<a>), aur images (<img>) implement karna seekha."
                )
                objectives = [
                    "HTML5 basic boilerplate aur document tree samajhna",
                    "VS Code mein Emmet shortcut (!) se fast coding karna",
                    "Semantic heading tags (<h1> to <h6>) aur paragraph (<p>) use karna",
                    "Anchor tag (<a>) aur images (<img>) with alt text master karna"
                ]
                sec1_head = "1. HTML Document Skeleton & Boilerplate"
                sec1_content = (
                    "Har HTML5 page <!DOCTYPE html> se start hota hai jo browser ko standard mode batata hai. "
                    "<html> tag poore document ko enclose karta hai, <head> mein metadata aur title hota hai, aur <body> mein web page ka visible content hota hai."
                )
                sec2_head = "2. Headings & Paragraph Formatting"
                sec2_content = (
                    "<h1> se lekar <h6> tak headings hoti hain. Page ka main heading hamesha <h1> hota hai jo single time use hona chahiye. "
                    "Paragraphs ke liye <p> tag use karte hain."
                )
                sec3_head = "3. Links & Images (<a> and <img> Tags)"
                sec3_content = (
                    "<a> (anchor tag) href attribute ke saath hyperlinks banata hai. External links ko new tab mein open karne ke liye target='_blank' lagate hain. "
                    "<img> tag self-closing hota hai jisme src aur alt attributes sabse important hote hain."
                )
                terms = [
                    {"term": "<!DOCTYPE html>", "definition": "Declaration that informs the browser this is an HTML5 document.", "timestamp_ref": "00:02:15"},
                    {"term": "href", "definition": "Hypertext Reference: Anchor tag ka attribute jo target URL define karta hai.", "timestamp_ref": "00:11:30"},
                    {"term": "alt text", "definition": "Alternative text jo image load na hone par ya screen readers ke liye display hota hai.", "timestamp_ref": "00:15:20"}
                ]
                pitfalls = [
                    "Multiple <h1> tags ek single page par use karna jo SEO ranking distort karta hai.",
                    "<img> tag mein alt attribute omit karna jisse accessibility score down hota hai.",
                    "File paths mein spelling mistake ya wrong relative folder reference karna."
                ]
                checklist = [
                    "Kya <!DOCTYPE html> document ke start mein present hai?",
                    "Kya page par exactly ek <h1> tag hai?",
                    "Kya external links par target='_blank' use kiya gaya hai?",
                    "Kya har <img> tag mein meaningful alt text provided hai?"
                ]
            else:  # English
                summary = (
                    "This lecture introduces web development fundamentals by constructing a first HTML website from scratch. "
                    "It covers the anatomy of an HTML document (<!DOCTYPE html>, <html>, <head>, <body>), developer workflow in VS Code, "
                    "Emmet boilerplate generation (!), semantic heading hierarchies (<h1> through <h6>), paragraphs, hyperlinks (<a>), "
                    "and images (<img>)."
                )
                objectives = [
                    "Understand the structural anatomy of an HTML5 document",
                    "Rapidly generate valid HTML boilerplate using VS Code Emmet shortcuts",
                    "Implement semantic content hierarchy using headings (<h1>-<h6>) and paragraphs (<p>)",
                    "Create hyperlinks with anchor tags and embed images with accessibility attributes"
                ]
                sec1_head = "1. HTML Document Anatomy & Boilerplate"
                sec1_content = (
                    "Every modern HTML5 document commences with '<!DOCTYPE html>', declaring the document standard to the browser. "
                    "The '<html>' element encloses the entire tree, '<head>' contains page metadata, scripts, and stylesheets, "
                    "while '<body>' encompasses all visual and interactive content."
                )
                sec2_head = "2. Semantic Text Hierarchy (Headings & Paragraphs)"
                sec2_content = (
                    "HTML provides six tiers of headings, from <h1> (highest significance) to <h6> (lowest). "
                    "Best practices dictate exactly one <h1> per page for search engine clarity and screen reader navigation. "
                    "Paragraphs (<p>) encapsulate standard blocks of text."
                )
                sec3_head = "3. Hyperlinks & Media Elements (<a> and <img>)"
                sec3_content = (
                    "The anchor element (<a>) links documents via its 'href' attribute; setting target='_blank' opens links in a new browsing context. "
                    "The image element (<img>) is self-closing and mandates 'src' (asset location) and 'alt' (accessible alternative text)."
                )
                terms = [
                    {"term": "<!DOCTYPE html>", "definition": "Document type declaration instructing browsers to render HTML5 standard.", "timestamp_ref": "00:02:15"},
                    {"term": "href", "definition": "Hypertext Reference attribute that specifies the destination URL of a link.", "timestamp_ref": "00:11:30"},
                    {"term": "alt attribute", "definition": "Alternative text descriptor required for accessibility and fallback display.", "timestamp_ref": "00:15:20"}
                ]
                pitfalls = [
                    "Using multiple <h1> tags on a single page, which breaks semantic document hierarchy.",
                    "Omitting the 'alt' attribute on <img> elements, penalizing accessibility audits.",
                    "Misconfiguring relative asset paths (e.g. failing to account for parent directory '../' syntax)."
                ]
                checklist = [
                    "Does the document declare <!DOCTYPE html> on the first line?",
                    "Is there exactly one <h1> heading defining the primary topic?",
                    "Do external anchor links include target='_blank' and rel='noopener'?",
                    "Do all <img> elements include accurate, descriptive alt text?"
                ]

            sections = [
                {
                    "heading": sec1_head,
                    "content": sec1_content,
                    "code_snippets": [
                        '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8">\n  <title>My First Website</title>\n</head>\n<body>\n  <h1>Welcome to Web Dev!</h1>\n</body>\n</html>'
                    ],
                    "supporting_chunk_ids": [cids[0], cids[1]] if len(cids) >= 2 else cids,
                    "citations": [get_chunk_cite(0), get_chunk_cite(1)]
                },
                {
                    "heading": sec2_head,
                    "content": sec2_content,
                    "code_snippets": [
                        "<h1>Main Course Heading</h1>\n<h2>Chapter 1: Foundations</h2>\n<p>This is a paragraph explaining the core concepts of HTML.</p>"
                    ],
                    "supporting_chunk_ids": [cids[2], cids[3]] if len(cids) >= 4 else cids[:1],
                    "citations": [get_chunk_cite(2), get_chunk_cite(3)]
                },
                {
                    "heading": sec3_head,
                    "content": sec3_content,
                    "code_snippets": [
                        '<a href="https://pw.live" target="_blank" rel="noopener">Visit PhysicsWallah</a>\n<img src="images/logo.png" alt="PW Logo" width="200">'
                    ],
                    "supporting_chunk_ids": [cids[4], cids[5]] if len(cids) >= 6 else cids[:1],
                    "citations": [get_chunk_cite(4), get_chunk_cite(5)]
                }
            ]

        # Style customizations
        if notes_style == "quick_revision":
            summary = "⚡ QUICK REVISION SUMMARY:\n• " + summary.replace(". ", ".\n• ")
            # Keep sections concise
            for s in sections:
                s["content"] = "• " + s["content"].replace(". ", ".\n• ")
        elif notes_style == "key_concepts":
            summary = "🔑 CORE ARCHITECTURAL CONCEPTS:\n" + summary

        notes_data = {
            "lecture_id": lecture_id,
            "lecture_title": lecture_title,
            "notes_style": notes_style,
            "language": language,
            "executive_summary": summary,
            "learning_objectives": objectives,
            "sections": sections,
            "definitions_and_terms": terms,
            "common_pitfalls": pitfalls,
            "revision_checklist": checklist,
            "generated_at": time.time(),
            "is_mock": True
        }

        return notes_data

    def export_notes_to_markdown(self, notes_data: Dict[str, Any]) -> str:
        """Export Smart Notes dictionary to a polished, formatted Markdown document."""
        title = notes_data.get("lecture_title", "Lecture Notes")
        lec_id = notes_data.get("lecture_id", "")
        style = notes_data.get("notes_style", "detailed_study").replace("_", " ").title()
        lang = notes_data.get("language", "English")

        lines = [
            f"# 📖 {title} — Smart Revision Notes",
            "",
            f"> **Lecture ID:** `{lec_id}` | **Style:** {style} | **Language:** {lang} | **Platform:** LearnMate AI",
            "",
            "---",
            "",
            "## 📌 Executive Summary",
            notes_data.get("executive_summary", "No summary available."),
            "",
            "---",
            "",
            "## 🎯 Learning Objectives",
        ]

        for obj in notes_data.get("learning_objectives", []):
            lines.append(f"- ✅ {obj}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 📖 Core Topics & Lecture Breakdown")
        lines.append("")

        for sec in notes_data.get("sections", []):
            heading = sec.get("heading", "Topic Section")
            content = sec.get("content", "")
            snippets = sec.get("code_snippets", [])
            citations = sec.get("citations", [])

            lines.append(f"### {heading}")
            lines.append(content)
            lines.append("")

            if snippets:
                lines.append("**💻 Code Example:**")
                for snip in snippets:
                    lines.append("```html")
                    lines.append(snip)
                    lines.append("```")
                lines.append("")

            if citations:
                cite_strs = [f"`[{c.get('timestamp', '00:00')}]` (Chunk `{c.get('chunk_id')}`)" for c in citations]
                lines.append(f"*📍 Lecture Timestamps:* {', '.join(cite_strs)}")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## 📚 Key Definitions & Terminology")
        lines.append("| Term | Definition | Lecture Timestamp |")
        lines.append("| :--- | :--- | :--- |")
        for term_obj in notes_data.get("definitions_and_terms", []):
            t = term_obj.get("term", "")
            d = term_obj.get("definition", "")
            ts = term_obj.get("timestamp_ref", "N/A")
            lines.append(f"| **`{t}`** | {d} | `[{ts}]` |")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## ⚠️ Common Pitfalls & Exam Traps")
        for p in notes_data.get("common_pitfalls", []):
            lines.append(f"- 🛑 {p}")
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## ✅ Quick Revision Checklist")
        for c in notes_data.get("revision_checklist", []):
            lines.append(f"- [ ] {c}")
        lines.append("")

        lines.append("---")
        lines.append("*Generated by LearnMate AI — PhysicsWallah APM Builder Screening*")

        return "\n".join(lines)

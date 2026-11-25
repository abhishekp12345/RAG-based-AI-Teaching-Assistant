"""
Practice Quiz Generator Module for LearnMate AI.
Generates lecture-grounded interactive practice quizzes with:
- Configurable Question Counts: 5 or 10 questions
- 3 Difficulty Levels: Easy, Medium, Hard
- 3 Languages: English, Hindi, Hinglish
- Grounded citations and exact lecture timestamps
- Deterministic local caching in data/quizzes/
- Client-side evaluation, score analytics, and weak-topic revision recommendations
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
    GEMINI_QUIZ_SCHEMA,
    QUIZ_STRUCTURED_SYSTEM_PROMPT,
    QUIZ_STRUCTURED_USER_TEMPLATE,
    format_context_for_llm
)
from llm import LLMClient

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Generates interactive practice questions with verified explanations and topic analytics."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        quizzes_dir: Optional[Path] = None,
        chunks_dir: Optional[Path] = None
    ):
        self.llm = llm_client or LLMClient()
        self.quizzes_dir = Path(quizzes_dir or config.QUIZZES_DIR)
        self.chunks_dir = Path(chunks_dir or config.CHUNKS_DIR)
        self.quizzes_dir.mkdir(parents=True, exist_ok=True)

    def load_lecture_chunks(self, lecture_id: str) -> List[Dict[str, Any]]:
        """Load all chunks for a specific lecture."""
        lec_file = self.chunks_dir / f"{lecture_id}_chunks.json"
        if lec_file.exists():
            try:
                with open(lec_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {lec_file}: {e}")

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
        difficulty: str,
        num_questions: int,
        language: str,
        chunks: List[Dict[str, Any]],
        is_mock: bool,
        model_name: Optional[str] = None
    ) -> str:
        """Compute deterministic cache key for quiz parameters and model."""
        chunk_sig = hashlib.md5("".join(c.get("chunk_id", "") for c in chunks).encode("utf-8")).hexdigest()[:8]
        mode_tag = "mock" if is_mock else "live"
        diff_tag = difficulty.lower()
        lang_tag = language.lower().replace(" ", "_")
        clean_model = (model_name or self.llm.model).replace("-", "_").replace(".", "_")
        return f"{lecture_id}_{diff_tag}_{num_questions}q_{lang_tag}_{mode_tag}_{clean_model}_{chunk_sig}.json"

    def generate_quiz(
        self,
        lecture_id: str,
        difficulty: str = "Medium",
        num_questions: int = 5,
        language: str = "English",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate or retrieve cached interactive quiz for a lecture.
        Guarantees strictly 4 options per question and valid correct_index (0..3).
        """
        chunks = self.load_lecture_chunks(lecture_id)
        if not chunks:
            raise ValueError(f"No chunks found for lecture '{lecture_id}'.")

        lecture_title = chunks[0].get("lecture_title", f"Lecture {lecture_id}")
        is_mock = self.llm.mock_mode
        num_q = 10 if num_questions >= 10 else 5

        # Check local cache
        cache_filename = self.compute_cache_key(lecture_id, difficulty, num_q, language, chunks, is_mock, self.llm.model)
        cache_path = self.quizzes_dir / cache_filename

        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                cached_data["is_cached"] = True
                logger.info(f"Loaded cached quiz from {cache_path.name}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}")

        # Generation
        if is_mock:
            quiz_data = self._generate_mock_quiz(lecture_id, lecture_title, difficulty, num_q, language, chunks)
        else:
            quiz_data = self._generate_live_quiz(lecture_id, lecture_title, difficulty, num_q, language, chunks)

        quiz_data["is_cached"] = False

        # Cache persistence
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(quiz_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached quiz to {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to write cache file {cache_path}: {e}")

        return quiz_data

    def _generate_live_quiz(
        self,
        lecture_id: str,
        lecture_title: str,
        difficulty: str,
        num_questions: int,
        language: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate quiz via Google Gemini structured output with model tracking."""
        formatted_context = format_context_for_llm(chunks)
        user_prompt = QUIZ_STRUCTURED_USER_TEMPLATE.format(
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            num_questions=num_questions,
            difficulty=difficulty,
            language=language,
            context=formatted_context
        )

        result = self.llm.generate_structured(
            system_prompt=QUIZ_STRUCTURED_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=GEMINI_QUIZ_SCHEMA,
            max_output_tokens=3500
        )

        if not result.parsed_json or result.error_message:
            raise RuntimeError(f"Live Gemini quiz generation failed: {result.error_message}")

        quiz = result.parsed_json

        # Validate questions schema
        questions = quiz.get("questions", [])
        valid_cids = {c["chunk_id"] for c in chunks}
        validated_q = []

        for idx, q in enumerate(questions):
            opts = q.get("options", [])
            while len(opts) < 4:
                opts.append(f"Option {chr(65 + len(opts))}")
            opts = opts[:4]

            corr = int(q.get("correct_index", 0))
            if corr < 0 or corr > 3:
                corr = 0

            raw_cids = q.get("supporting_chunk_ids", [])
            clean_cids = [cid for cid in raw_cids if cid in valid_cids]

            validated_q.append({
                "id": idx + 1,
                "question": q.get("question", f"Question {idx + 1}"),
                "options": opts,
                "correct_index": corr,
                "explanation": q.get("explanation", "See lecture notes for explanation."),
                "topic": q.get("topic", "General HTML"),
                "timestamp_ref": q.get("timestamp_ref", "00:00"),
                "supporting_chunk_ids": clean_cids
            })

        quiz["quiz_id"] = f"QUIZ_{lecture_id}_{difficulty.upper()}_{int(time.time())}"
        quiz["lecture_id"] = lecture_id
        quiz["lecture_title"] = lecture_title
        quiz["difficulty"] = difficulty
        quiz["num_questions"] = len(validated_q)
        quiz["language"] = language
        quiz["questions"] = validated_q
        quiz["generating_model"] = result.model_name
        quiz["generated_at"] = time.time()
        quiz["is_mock"] = False

        return quiz

    def _generate_mock_quiz(
        self,
        lecture_id: str,
        lecture_title: str,
        difficulty: str,
        num_questions: int,
        language: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate realistic, lecture-grounded mock quiz bank."""
        cids = [c["chunk_id"] for c in chunks]

        # Bank of 10 grounded questions per lecture
        if lecture_id == "LEC_01":  # Tutorial #6: SEO & Core Web Vitals
            if language == "Hindi":
                bank = [
                    {
                        "question": "मोबाइल और विभिन्न स्क्रीन आकारों पर उत्तरदायी (responsive) रेंडरिंग के लिए कौन सा मेटा टैग अनिवार्य है?",
                        "options": [
                            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                            "<meta name='screen' content='responsive=true'>",
                            "<meta http-equiv='mobile-render' content='auto'>",
                            "<meta name='layout' content='fluid'>"
                        ],
                        "correct_index": 0,
                        "explanation": "Viewport मेटा टैग ब्राउज़र को डिवाइस की चौड़ाई के अनुसार पेज का दृश्य क्षेत्र स्केल करने का निर्देश देता है।",
                        "topic": "Responsive Viewport",
                        "timestamp_ref": "00:01:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "Google खोज परिणामों में वेबपेज के नीचे दिखने वाला संक्षिप्त विवरण किस टैग द्वारा परिभाषित होता है?",
                        "options": [
                            "<meta name='summary' content='...'>",
                            "<meta name='description' content='...'>",
                            "<meta name='snippet' content='...'>",
                            "<title description='...'>"
                        ],
                        "correct_index": 1,
                        "explanation": "<meta name='description'> सर्च रिजल्ट्स में दिखने वाला आधिकारिक स्निपेट तय करता है।",
                        "topic": "SEO Meta Tags",
                        "timestamp_ref": "00:02:40",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Google Core Web Vitals में LCP (Largest Contentful Paint) क्या मापता है?",
                        "options": [
                            "पेज पर पहला बाइट प्राप्त होने का समय",
                            "पेज के सबसे बड़े मुख्य कंटेंट ब्लॉक के रेंडर होने का समय",
                            "JavaScript फाइलों के निष्पादन की कुल गति",
                            "लेआउट में अनचाहे बदलाव की आवृत्ति"
                        ],
                        "correct_index": 1,
                        "explanation": "LCP पेज पर सबसे बड़े दृश्यमान कंटेंट (जैसे हीरो इमेज या बड़ा हेडिंग ब्लॉक) लोड होने का समय मापता है।",
                        "topic": "LCP Performance",
                        "timestamp_ref": "00:05:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "वेबसाइट पर Cumulative Layout Shift (CLS) का मुख्य कारण क्या होता है?",
                        "options": [
                            "इमेजेस और विज्ञापनों में स्पष्ट width और height आयाम न होना",
                            "Google Fonts का उपयोग करना",
                            "वीडियो पर autoplay एट्रिब्यूट लगाना",
                            "मेटा टैग्स में कीवर्ड्स न लिखना"
                        ],
                        "correct_index": 0,
                        "explanation": "जब इमेजेस में width और height नहीं होते, तो वे लोड होते समय नीचे के कंटेंट को अचानक धकेल देती हैं जिससे CLS बढ़ता है।",
                        "topic": "CLS Visual Stability",
                        "timestamp_ref": "00:08:12",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Google Chrome DevTools में वेबसाइट की परफॉरमेंस, एक्सेसिबिलिटी और SEO ऑडिट करने के लिए कौन सा टूल इन-बिल्ट है?",
                        "options": ["Network Tab", "Console", "Lighthouse", "Memory Profiler"],
                        "correct_index": 2,
                        "explanation": "Chrome DevTools का Lighthouse पैनल संपूर्ण वेबसाइट की परफॉरमेंस और SEO का स्कोर 100 में से देता है।",
                        "topic": "Lighthouse Audits",
                        "timestamp_ref": "00:03:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "एक उत्कृष्ट उपयोगकर्ता अनुभव के लिए Google द्वारा अनुशंसित LCP की अधिकतम सीमा क्या है?",
                        "options": ["1.0 सेकंड", "2.5 सेकंड या उससे कम", "5.0 सेकंड", "100 मिलीसेकंड"],
                        "correct_index": 1,
                        "explanation": "Google Core Web Vitals के अनुसार 2.5 सेकंड या उससे कम का LCP 'Good' माना जाता है।",
                        "topic": "LCP Thresholds",
                        "timestamp_ref": "00:06:10",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "सर्च इंजन क्रॉलर्स के लिए <img> टैग का कौन सा एट्रिब्यूट सबसे महत्वपूर्ण माना जाता है?",
                        "options": ["title", "alt", "class", "id"],
                        "correct_index": 1,
                        "explanation": "alt (alternative text) इमेज की सामग्री का वर्णन करता है जो SEO और स्क्रीन रीडर्स दोनों के लिए महत्वपूर्ण है।",
                        "topic": "Image SEO & Alt Text",
                        "timestamp_ref": "00:10:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "Core Web Vitals में FID (First Input Delay) क्या दर्शाता है?",
                        "options": [
                            "यूजर के पहले क्लिक/टैप पर ब्राउज़र के प्रतिक्रिया देने में लगा समय",
                            "सर्वर का पहला रिस्पॉन्स टाइम",
                            "CSS फाइल के डाउनलोड होने का समय",
                            "डोमेन नाम के DNS रिज़ॉल्यूशन का समय"
                        ],
                        "correct_index": 0,
                        "explanation": "FID यूजर के पहले इंटरैक्शन (क्लिक, टैप, की-प्रेस) और मुख्य थ्रेड द्वारा उसे प्रोसेस करने के बीच की देरी को मापता है।",
                        "topic": "FID Interactivity",
                        "timestamp_ref": "00:07:25",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "सोशल मीडिया प्लेटफॉर्म्स पर लिंक शेयर करते समय कार्ड प्रीव्यू को नियंत्रित करने के लिए कौन से टैग्स का उपयोग किया जाता है?",
                        "options": ["Twitter Bootstrap", "Open Graph (og:) meta tags", "Robots.txt", "Sitemap.xml"],
                        "correct_index": 1,
                        "explanation": "Open Graph टैग्स (og:title, og:image, og:description) सोशल मीडिया पर शेयरिंग कार्ड्स को फॉर्मेट करते हैं।",
                        "topic": "Social Media Meta Tags",
                        "timestamp_ref": "00:04:30",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "ब्राउज़र टैब पर प्रदर्शित होने वाले फेविकॉन (Favicon) को जोड़ने का सही सिंटैक्स क्या है?",
                        "options": [
                            "<link rel='icon' type='image/x-icon' href='favicon.ico'>",
                            "<meta icon='favicon.ico'>",
                            "<favicon src='favicon.ico'></favicon>",
                            "<img type='favicon' src='favicon.ico'>"
                        ],
                        "correct_index": 0,
                        "explanation": "<link rel='icon'> टैग को <head> में फेविकॉन लिंक करने के लिए निर्दिष्ट किया जाता है।",
                        "topic": "Favicon Setup",
                        "timestamp_ref": "00:09:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    }
                ]
            elif language == "Hinglish":
                bank = [
                    {
                        "question": "Mobile devices par responsive design ensure karne ke liye <head> mein kaunsa meta tag add karna zaroori hai?",
                        "options": [
                            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                            "<meta name='mobile' content='responsive=true'>",
                            "<meta http-equiv='screen' content='device-width'>",
                            "<meta name='layout' content='fluid'>"
                        ],
                        "correct_index": 0,
                        "explanation": "Viewport tag browser ko screen width ke hisab se layout adjust karne ka command deta hai.",
                        "topic": "Responsive Viewport",
                        "timestamp_ref": "00:01:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "Google search results page par website title ke niche jo text snippet dikhta hai wo kis tag se define hota hai?",
                        "options": [
                            "<meta name='summary' content='...'>",
                            "<meta name='description' content='...'>",
                            "<meta name='snippet' content='...'>",
                            "<meta name='info' content='...'>"
                        ],
                        "correct_index": 1,
                        "explanation": "<meta name='description'> search results mein preview description provide karta hai.",
                        "topic": "SEO Meta Tags",
                        "timestamp_ref": "00:02:40",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Core Web Vitals metric LCP (Largest Contentful Paint) exactly kya measure karta hai?",
                        "options": [
                            "Server se pehla byte aane ka time (TTFB)",
                            "Page ka largest hero content render hone ka time",
                            "JavaScript execution ka total time",
                            "Unexpected layout shifts ka score"
                        ],
                        "correct_index": 1,
                        "explanation": "LCP screen ke sabse bade visual content block ke render hone ka duration calculate karta hai.",
                        "topic": "LCP Performance",
                        "timestamp_ref": "00:05:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Website par unexpected CLS (Cumulative Layout Shift) aane ka sabse common reason kya hota hai?",
                        "options": [
                            "Images par explicit width aur height na dena",
                            "External CSS stylesheet link karna",
                            "Favicon ka size 16x16 hona",
                            "Emmet shortcut se code likhna"
                        ],
                        "correct_index": 0,
                        "explanation": "Bina dimension wali images load hote waqt content ko neeche push karti hain jisse layout shift hota hai.",
                        "topic": "CLS Visual Stability",
                        "timestamp_ref": "00:08:12",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Chrome DevTools mein website ki Performance, SEO aur Accessibility ka comprehensive score kaunsa tool deta hai?",
                        "options": ["Console", "Network Tab", "Lighthouse", "Application Tab"],
                        "correct_index": 2,
                        "explanation": "Lighthouse tool automated audits run karke detailed scores aur improvements provide karta hai.",
                        "topic": "Lighthouse Audits",
                        "timestamp_ref": "00:03:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Google Core Web Vitals ke hisab se ideal LCP time kitna hona chahiye?",
                        "options": ["1.0 second", "2.5 seconds ya usse kam", "4.0 seconds", "100 ms"],
                        "correct_index": 1,
                        "explanation": "2.5 seconds ke under LCP score ko Google 'Good' status classify karta hai.",
                        "topic": "LCP Thresholds",
                        "timestamp_ref": "00:06:10",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Image SEO aur screen readers ke liye <img> element par kaunsa attribute mandatory hota hai?",
                        "options": ["title", "alt", "name", "role"],
                        "correct_index": 1,
                        "explanation": "alt attribute alternative text describe karta hai jo SEO crawlers inspect karte hain.",
                        "topic": "Image SEO & Alt Text",
                        "timestamp_ref": "00:10:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "Core Web Vitals mein FID (First Input Delay) kya evaluate karta hai?",
                        "options": [
                            "User ke first click/tap par browser ki responsiveness delay",
                            "Server connection establish hone ka time",
                            "Video file buffering speed",
                            "Total page download weight"
                        ],
                        "correct_index": 0,
                        "explanation": "FID user ke first action aur main thread response ke beech ka delay track karta hai.",
                        "topic": "FID Interactivity",
                        "timestamp_ref": "00:07:25",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Social media sharing cards (WhatsApp, Twitter, LinkedIn) customize karne ke liye konse meta tags use hote hain?",
                        "options": ["Bootstrap metadata", "Open Graph (og:) tags", "Favicon tags", "Google Ads tags"],
                        "correct_index": 1,
                        "explanation": "Open Graph protocol tags (og:image, og:title) link previews generate karte hain.",
                        "topic": "Social Media Meta Tags",
                        "timestamp_ref": "00:04:30",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Browser tab par website icon lagane ke liye standard syntax kya hai?",
                        "options": [
                            "<link rel='icon' type='image/x-icon' href='favicon.ico'>",
                            "<meta src='favicon.ico'>",
                            "<icon file='favicon.ico'>",
                            "<img rel='icon' src='favicon.ico'>"
                        ],
                        "correct_index": 0,
                        "explanation": "<link rel='icon'> se browser tab par favicon render hota hai.",
                        "topic": "Favicon Setup",
                        "timestamp_ref": "00:09:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    }
                ]
            else:  # English
                bank = [
                    {
                        "question": "Which <meta> tag configuration is essential to ensure responsive rendering across mobile screens?",
                        "options": [
                            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
                            "<meta name='screen' content='device-width, initial-scale=1.0'>",
                            "<meta http-equiv='mobile-render' content='auto'>",
                            "<meta name='responsive' content='true'>"
                        ],
                        "correct_index": 0,
                        "explanation": "The viewport meta tag matches the virtual viewport to physical device width at 1:1 initial scale.",
                        "topic": "Responsive Viewport",
                        "timestamp_ref": "00:01:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "Which HTML tag supplies the summary snippet displayed beneath your page title on Google search results?",
                        "options": [
                            "<meta name='summary' content='...'>",
                            "<meta name='description' content='...'>",
                            "<meta name='snippet' content='...'>",
                            "<meta name='details' content='...'>"
                        ],
                        "correct_index": 1,
                        "explanation": "Search engines inspect <meta name='description'> to populate the organic search results snippet.",
                        "topic": "SEO Meta Tags",
                        "timestamp_ref": "00:02:40",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "What core performance dimension does Largest Contentful Paint (LCP) measure in Google Core Web Vitals?",
                        "options": [
                            "Time to First Byte (TTFB) from the hosting server",
                            "Render time of the largest visible content element in the viewport",
                            "Duration taken to parse inline JavaScript tags",
                            "Frequency of jarring DOM repositioning during scrolling"
                        ],
                        "correct_index": 1,
                        "explanation": "LCP benchmarks perceived loading speed by recording when the primary hero content finishes rendering.",
                        "topic": "LCP Performance",
                        "timestamp_ref": "00:05:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "What is the primary cause of poor Cumulative Layout Shift (CLS) on educational web pages?",
                        "options": [
                            "Images and dynamic media embedded without explicit width and height dimensions",
                            "Loading external CSS from a Content Delivery Network (CDN)",
                            "Using CSS Grid and Flexbox simultaneously",
                            "Omitting the charset meta tag in <head>"
                        ],
                        "correct_index": 0,
                        "explanation": "Images without explicit dimensions force browser reflows as assets load, causing jarring layout jumps.",
                        "topic": "CLS Visual Stability",
                        "timestamp_ref": "00:08:12",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Which Chrome DevTools tool automates auditing for Performance, Accessibility, Best Practices, and SEO?",
                        "options": ["Console", "Network", "Lighthouse", "Application"],
                        "correct_index": 2,
                        "explanation": "Google Lighthouse delivers structured audits and performance scores directly within Chrome DevTools.",
                        "topic": "Lighthouse Audits",
                        "timestamp_ref": "00:03:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "What is the recommended threshold for Largest Contentful Paint (LCP) to achieve a 'Good' Core Web Vitals rating?",
                        "options": ["<= 1.0 second", "<= 2.5 seconds", "<= 4.0 seconds", "<= 100 milliseconds"],
                        "correct_index": 1,
                        "explanation": "Google defines an LCP of 2.5 seconds or lower as 'Good', 2.5-4.0s as 'Needs Improvement', and >4.0s as 'Poor'.",
                        "topic": "LCP Thresholds",
                        "timestamp_ref": "00:06:10",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Which attribute on <img> elements is critical for crawler comprehension and screen-reader accessibility?",
                        "options": ["title", "alt", "name", "role"],
                        "correct_index": 1,
                        "explanation": "The 'alt' attribute conveys image semantics to search engine indexers and assistive screen readers.",
                        "topic": "Image SEO & Alt Text",
                        "timestamp_ref": "00:10:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "In Google Core Web Vitals, what interaction does First Input Delay (FID) measure?",
                        "options": [
                            "The delay between a user's first input interaction and browser main-thread event handling",
                            "The time taken to establish an HTTPS handshake",
                            "The total frame rate rendering latency during scrolling",
                            "The duration required to load background web workers"
                        ],
                        "correct_index": 0,
                        "explanation": "FID quantifies responsiveness by timing the pause before the browser begins processing the user's initial tap or click.",
                        "topic": "FID Interactivity",
                        "timestamp_ref": "00:07:25",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Which protocol of meta tags formats preview cards when pages are shared across social channels?",
                        "options": ["JSON-LD Schema", "Open Graph (og:) protocol", "Robots Meta Directive", "Microformats 2"],
                        "correct_index": 1,
                        "explanation": "Open Graph (og:title, og:image, og:description) defines the multimedia card displayed on social feeds.",
                        "topic": "Social Media Meta Tags",
                        "timestamp_ref": "00:04:30",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "What is the standard HTML tag and relation used to declare a browser tab favicon?",
                        "options": [
                            "<link rel='icon' type='image/x-icon' href='favicon.ico'>",
                            "<meta name='icon' content='favicon.ico'>",
                            "<favicon src='favicon.ico'>",
                            "<img rel='icon' src='favicon.ico'>"
                        ],
                        "correct_index": 0,
                        "explanation": "<link rel='icon'> specifies the icon file that browsers render on tabs and bookmarks.",
                        "topic": "Favicon Setup",
                        "timestamp_ref": "00:09:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    }
                ]

        elif lecture_id == "LEC_02":  # Tutorial #10: Video, Audio & Media in HTML
            if language == "Hindi":
                bank = [
                    {
                        "question": "आधुनिक ब्राउज़रों में <video> टैग पर autoplay एट्रिब्यूट काम करने के लिए क्या आवश्यक है?",
                        "options": [
                            "साथ में muted एट्रिब्यूट भी मौजूद होना चाहिए",
                            "controls एट्रिब्यूट हटा देना चाहिए",
                            "वीडियो का फॉर्मेट केवल AVI होना चाहिए",
                            "loop एट्रिब्यूट लगाना आवश्यक है"
                        ],
                        "correct_index": 0,
                        "explanation": "ब्राउज़र सुरक्षा नीतियों के तहत बिना muted के ऑटोप्ले ब्लॉक कर दिया जाता है ताकि यूजर को अचानक तेज आवाज से परेशानी न हो।",
                        "topic": "Autoplay & Audio Policies",
                        "timestamp_ref": "00:03:45",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "<video> टैग में controls एट्रिब्यूट का क्या कार्य होता है?",
                        "options": [
                            "वीडियो प्लेयर में प्ले, पॉज, वॉल्यूम और सीक बार प्रदर्शित करना",
                            "वीडियो की गुणवत्ता को स्वचालित रूप से बढ़ाना",
                            "वीडियो को डाउनलोड होने से रोकना",
                            "वीडियो को केवल फुलस्क्रीन में चलाना"
                        ],
                        "correct_index": 0,
                        "explanation": "controls एट्रिब्यूट ब्राउज़र का मूल मीडिया कंट्रोल इंटरफेस स्क्रीन पर प्रदर्शित करता है।",
                        "topic": "Video Controls",
                        "timestamp_ref": "00:02:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "वीडियो शुरू होने से पहले थंबनेल प्रीव्यू इमेज दिखाने के लिए किस एट्रिब्यूट का उपयोग किया जाता है?",
                        "options": ["thumbnail", "preview", "poster", "cover"],
                        "correct_index": 2,
                        "explanation": "poster एट्रिब्यूट इमेज का URL लेता है जो वीडियो प्ले होने से पहले प्रदर्शित होती है।",
                        "topic": "Video Poster Attribute",
                        "timestamp_ref": "00:04:50",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "<video> या <audio> के अंदर कई <source> टैग्स शामिल करने का क्या लाभ है?",
                        "options": [
                            "वीडियो की गति को दोगुना करना",
                            "अलग-अलग ब्राउज़रों के लिए फॉलबैक फॉर्मेट्स (जैसे mp4, webm) प्रदान करना",
                            "वीडियो पर विज्ञापन दिखाना",
                            "बैकग्राउंड म्यूजिक जोड़ना"
                        ],
                        "correct_index": 1,
                        "explanation": "विभिन्न ब्राउज़र अलग-अलग कोडेक्स सपोर्ट करते हैं; <source> टैग्स ब्राउज़र को संगत फॉर्मेट चुनने की अनुमति देते हैं।",
                        "topic": "Media Source Fallbacks",
                        "timestamp_ref": "00:06:30",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "YouTube वीडियो या बाहरी वेबपेज को अपनी वेबसाइट में एम्बेड करने के लिए कौन सा टैग उपयोग किया जाता है?",
                        "options": ["<embed-video>", "<iframe>", "<external>", "<frame>"],
                        "correct_index": 1,
                        "explanation": "<iframe> (Inline Frame) बाहरी वेबपेज और मीडिया प्लेयर्स को एम्बेड करने के लिए मानक टैग है।",
                        "topic": "iframe Media Embeds",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "मीडिया फाइल के समाप्त होने पर उसे स्वचालित रूप से पुनः शुरू करने के लिए कौन सा एट्रिब्यूट प्रयोग होता है?",
                        "options": ["repeat", "replay", "loop", "restart"],
                        "correct_index": 2,
                        "explanation": "loop एट्रिब्यूट वीडियो या ऑडियो समाप्त होने पर उसे निरंतर रूप से पुनः चलाता है।",
                        "topic": "Loop Playback",
                        "timestamp_ref": "00:05:15",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "<iframe> पर allowfullscreen एट्रिब्यूट लगाने से क्या होता है?",
                        "options": [
                            "यूजर एम्बेडेड वीडियो को फुलस्क्रीन में देख सकता है",
                            "पेज हमेशा फुलस्क्रीन में लोड होता है",
                            "यह ब्राउज़र के सभी टैब्स को बंद कर देता है",
                            "यह विज्ञापन को फुलस्क्रीन कर देता है"
                        ],
                        "correct_index": 0,
                        "explanation": "allowfullscreen एट्रिब्यूट एम्बेडेड प्लेयर को फुलस्क्रीन मोड में जाने की अनुमति देता है।",
                        "topic": "iframe Permissions",
                        "timestamp_ref": "00:11:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "HTML5 में ऑडियो चलाने का सही सिंटैक्स कौन सा है?",
                        "options": [
                            "<audio controls><source src='music.mp3' type='audio/mpeg'></audio>",
                            "<music src='music.mp3' play='true'>",
                            "<sound play controls src='music.mp3'>",
                            "<media type='audio' src='music.mp3'>"
                        ],
                        "correct_index": 0,
                        "explanation": "HTML5 में <audio> टैग और <source> का संयोजन मानक सिंटैक्स है।",
                        "topic": "Audio Syntax",
                        "timestamp_ref": "00:07:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "यदि <video> टैग पर controls न दिया जाए और autoplay न हो, तो क्या होगा?",
                        "options": [
                            "वीडियो स्वचालित रूप से बैकग्राउंड में चलेगा",
                            "वीडियो दिखाई देगा लेकिन यूजर उसे प्ले या नियंत्रित नहीं कर सकेगा",
                            "ब्राउज़र क्रैश हो जाएगा",
                            "यह अपने आप ऑडियो में बदल जाएगा"
                        ],
                        "correct_index": 1,
                        "explanation": "बिना controls के यूजर को प्ले बटन नहीं मिलता, जिससे वीडियो स्थिर पोस्टर की तरह अचल दिखता है।",
                        "topic": "Media Error Handling",
                        "timestamp_ref": "00:08:20",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "<video> टैग पर width और height एट्रिब्यूट निर्दिष्ट करने का क्या लाभ है?",
                        "options": [
                            "ब्राउज़र वीडियो लोड होने से पहले ही जगह आरक्षित कर लेता है, जिससे लेआउट शिफ्ट नहीं होता",
                            "वीडियो का फाइल साइज छोटा हो जाता है",
                            "वीडियो की आवाज तेज हो जाती है",
                            "यह वीडियो को स्वचालित रूप से म्यूट कर देता है"
                        ],
                        "correct_index": 0,
                        "explanation": "स्पष्ट डाइमेंशन्स देने से पेज रेंडरिंग के समय अचानक लेआउट शिफ्ट (CLS) से बचाव होता है।",
                        "topic": "Player Dimensions",
                        "timestamp_ref": "00:03:00",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    }
                ]
            elif language == "Hinglish":
                bank = [
                    {
                        "question": "Modern browsers mein <video> tag par autoplay successfully trigger karne ke liye kaunsa attribute zaroori hota hai?",
                        "options": [
                            "muted attribute zaroori hota hai",
                            "controls attribute hatana padta hai",
                            "Sirf MP4 format allow hota hai",
                            "loop attribute mandatory hota hai"
                        ],
                        "correct_index": 0,
                        "explanation": "Browser security policies prevent noisy unexpected playback; autoplay requires muted attribute.",
                        "topic": "Autoplay & Audio Policies",
                        "timestamp_ref": "00:03:45",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "<video> tag mein controls attribute add karne se kya functionality milti hai?",
                        "options": [
                            "Play, pause, volume, aur seek bar controls display hote hain",
                            "Video speed automatically double ho jati hai",
                            "Video downloads strictly block ho jate hain",
                            "Video sirf fullscreen mein hi chalti hai"
                        ],
                        "correct_index": 0,
                        "explanation": "controls attribute browser ka default playback control bar enable karta hai.",
                        "topic": "Video Controls",
                        "timestamp_ref": "00:02:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Video play hone se pehle thumbnail preview image display karne ke liye kaunsa attribute use hota hai?",
                        "options": ["thumbnail", "preview", "poster", "cover"],
                        "correct_index": 2,
                        "explanation": "poster attribute thumbnail image ka path define karta hai jo initial screen par dikhta hai.",
                        "topic": "Video Poster Attribute",
                        "timestamp_ref": "00:04:50",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "<video> ya <audio> element ke andar multiple <source> tags specify karne ka main purpose kya hai?",
                        "options": [
                            "Ek saath 2 gaane bajana",
                            "Different browsers ke liye format fallback (mp4, webm) provide karna",
                            "Video resolution HD mein force karna",
                            "Background animation chalana"
                        ],
                        "correct_index": 1,
                        "explanation": "Cross-browser compatibility ke liye multiple media formats define kiye jate hain.",
                        "topic": "Media Source Fallbacks",
                        "timestamp_ref": "00:06:30",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "YouTube video ya external web page ko apni HTML website par embed karne ke liye standard tag kaunsa hai?",
                        "options": ["<youtube>", "<iframe>", "<embed-page>", "<media-box>"],
                        "correct_index": 1,
                        "explanation": "<iframe> (Inline Frame) standard element hai external content embed karne ke liye.",
                        "topic": "iframe Media Embeds",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "Video ya audio complete hote hi continuously repeat mode mein chalane ke liye kaunsa attribute lagate hain?",
                        "options": ["repeat", "replay", "loop", "continuous"],
                        "correct_index": 2,
                        "explanation": "loop attribute media finish hote hi auto restart initiate karta hai.",
                        "topic": "Loop Playback",
                        "timestamp_ref": "00:05:15",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "<iframe> par allowfullscreen attribute add karne ka kya purpose hota hai?",
                        "options": [
                            "User ko embedded video fullscreen mein expand karne ki permission dena",
                            "Page ko default fullscreen open karna",
                            "Browser address bar hide karna",
                            "Mobile screen ko landscape lock karna"
                        ],
                        "correct_index": 0,
                        "explanation": "allowfullscreen permission user ko full monitor view grant karti hai.",
                        "topic": "iframe Permissions",
                        "timestamp_ref": "00:11:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "HTML5 mein standard audio player create karne ka correct syntax kaunsa hai?",
                        "options": [
                            "<audio controls><source src='audio.mp3' type='audio/mpeg'></audio>",
                            "<music src='audio.mp3' controls='true'>",
                            "<sound play='yes' src='audio.mp3'>",
                            "<media type='audio' file='audio.mp3'>"
                        ],
                        "correct_index": 0,
                        "explanation": "<audio controls> with nested <source> standard HTML5 specification hai.",
                        "topic": "Audio Syntax",
                        "timestamp_ref": "00:07:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Agar <video> tag mein controls attribute omit kar diya jaye aur autoplay bhi na ho toh user par kya effect padega?",
                        "options": [
                            "Video audio mode mein background mein bajegi",
                            "Video display toh hogi par student play/pause nahi kar payega",
                            "Browser alert error dikhayega",
                            "Video automatic new window mein khul jayegi"
                        ],
                        "correct_index": 1,
                        "explanation": "Bina controls ke play/pause button accessible nahi hote.",
                        "topic": "Media Error Handling",
                        "timestamp_ref": "00:08:20",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "<video> element par width aur height specify karna best practice kyu maana jata hai?",
                        "options": [
                            "Layout shift (CLS) avoid hota hai kyunki browser space pehle reserve kar leta hai",
                            "Video compression high ho jati hai",
                            "Subtitles automatically generate hote hain",
                            "Video buffering time zero ho jata hai"
                        ],
                        "correct_index": 0,
                        "explanation": "Dimensions reserve karne se video load hote waqt layout jumps nahi aate.",
                        "topic": "Player Dimensions",
                        "timestamp_ref": "00:03:00",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    }
                ]
            else:  # English
                bank = [
                    {
                        "question": "What requirement must be met for modern web browsers to permit automated video playback using 'autoplay'?",
                        "options": [
                            "The 'muted' attribute must also be specified on the <video> tag",
                            "The 'controls' attribute must be omitted entirely",
                            "The video file must be encoded exclusively in WebM format",
                            "The 'loop' attribute must be enabled"
                        ],
                        "correct_index": 0,
                        "explanation": "Browser autonomy policies strictly enforce that unmuted video cannot autoplay to avoid startling users with unsolicited sound.",
                        "topic": "Autoplay & Audio Policies",
                        "timestamp_ref": "00:03:45",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "What is the function of the boolean 'controls' attribute on the HTML5 <video> element?",
                        "options": [
                            "It renders browser playback controls including play/pause, seek track, and volume slider",
                            "It enables hardware acceleration on user GPUs",
                            "It prevents users from right-clicking to download the video",
                            "It forces the media player into a full-viewport modal"
                        ],
                        "correct_index": 0,
                        "explanation": "The 'controls' attribute directs the browser to present native user playback controls.",
                        "topic": "Video Controls",
                        "timestamp_ref": "00:02:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Which attribute on the <video> tag designates a thumbnail preview image prior to playback?",
                        "options": ["thumbnail", "preview", "poster", "cover"],
                        "correct_index": 2,
                        "explanation": "The 'poster' attribute specifies an image URL displayed until the user initiates video playback.",
                        "topic": "Video Poster Attribute",
                        "timestamp_ref": "00:04:50",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Why is it best practice to enclose multiple <source> elements within an <audio> or <video> tag?",
                        "options": [
                            "To play multiple media channels simultaneously",
                            "To provide fallback formats (e.g. MP4, WebM) across different browser codec implementations",
                            "To enhance video compression speed by 50%",
                            "To inject subtitles automatically into the media pipeline"
                        ],
                        "correct_index": 1,
                        "explanation": "Different browsers support distinct audio/video codecs; multiple sources offer robust compatibility fallbacks.",
                        "topic": "Media Source Fallbacks",
                        "timestamp_ref": "00:06:30",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Which HTML element is the standard mechanism to embed third-party media like YouTube or interactive maps?",
                        "options": ["<embed-video>", "<iframe>", "<plugin>", "<portal>"],
                        "correct_index": 1,
                        "explanation": "The <iframe> (Inline Frame) encloses an external browsing context to embed videos, maps, and widgets.",
                        "topic": "iframe Media Embeds",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "Which attribute causes an HTML5 media element to automatically replay once it concludes?",
                        "options": ["repeat", "replay", "loop", "continuous"],
                        "correct_index": 2,
                        "explanation": "The 'loop' boolean attribute instructs the browser to seamlessly restart media playback upon reaching the end.",
                        "topic": "Loop Playback",
                        "timestamp_ref": "00:05:15",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "What capability does the 'allowfullscreen' attribute grant to an embedded <iframe> video player?",
                        "options": [
                            "It authorizes the embedded player to expand across the user's entire monitor display",
                            "It forces the outer website into kiosk mode",
                            "It hides browser navigation chrome permanently",
                            "It locks screen orientation to landscape"
                        ],
                        "correct_index": 0,
                        "explanation": "The 'allowfullscreen' attribute explicitly delegates fullscreen API capabilities to the nested frame context.",
                        "topic": "iframe Permissions",
                        "timestamp_ref": "00:11:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "What is the standard, valid HTML5 syntax for embedding native audio playback?",
                        "options": [
                            "<audio controls><source src='lecture.mp3' type='audio/mpeg'></audio>",
                            "<music src='lecture.mp3' controls='true'>",
                            "<sound src='lecture.mp3' play controls>",
                            "<audio file='lecture.mp3' auto>"
                        ],
                        "correct_index": 0,
                        "explanation": "<audio controls> containing nested <source> elements is the standard semantic implementation.",
                        "topic": "Audio Syntax",
                        "timestamp_ref": "00:07:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "What occurs if developer code specifies a <video> element without 'controls' and without 'autoplay'?",
                        "options": [
                            "The audio plays in the background while video is hidden",
                            "The video renders as an inert still image that users cannot trigger or navigate",
                            "The browser triggers a console fatal exception",
                            "The video defaults to launching in an external desktop application"
                        ],
                        "correct_index": 1,
                        "explanation": "Without controls or script triggers, the user has no interface to initiate playback, rendering the element inert.",
                        "topic": "Media Error Handling",
                        "timestamp_ref": "00:08:20",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Why is it strongly recommended to explicitly define 'width' and 'height' attributes on <video> tags?",
                        "options": [
                            "To reserve layout aspect ratio in advance, preventing Cumulative Layout Shift (CLS)",
                            "To reduce server streaming bandwidth by 40%",
                            "To auto-adjust playback volume based on screen area",
                            "To force high-definition 1080p stream resolution"
                        ],
                        "correct_index": 0,
                        "explanation": "Explicit dimensions allow the browser to reserve the required box model area before the video headers arrive.",
                        "topic": "Player Dimensions",
                        "timestamp_ref": "00:03:00",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    }
                ]

        else:  # LEC_03: Tutorial #2: Your First HTML Website
            if language == "Hindi":
                bank = [
                    {
                        "question": "प्रत्येक HTML5 डॉक्यूमेंट के सबसे पहली लाइन पर <!DOCTYPE html> लिखने का क्या कारण है?",
                        "options": [
                            "ब्राउज़र को यह निर्देश देना कि पेज HTML5 स्टैंडर्ड मोड में रेंडर होना चाहिए",
                            "वेबसाइट का पासवर्ड सेट करना",
                            "CSS फाइल को लोड करना",
                            "सर्वर का IP एड्रेस बताना"
                        ],
                        "correct_index": 0,
                        "explanation": "<!DOCTYPE html> डिक्लेरेशन ब्राउज़र को क्वार्क्स मोड (quirks mode) से बचाकर मानक HTML5 मोड में रेंडर करने का निर्देश देता है।",
                        "topic": "HTML5 Doctype",
                        "timestamp_ref": "00:02:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "HTML डॉक्यूमेंट में <head> और <body> एलिमेंट्स के बीच क्या अंतर है?",
                        "options": [
                            "<head> में मेटाडेटा और सेटिंग्स होती हैं; <body> में यूजर को दिखने वाली दृश्य सामग्री होती है",
                            "<head> केवल मोबाइल पर दिखता है और <body> डेस्कटॉप पर",
                            "<head> में इमेजेस होती हैं और <body> में केवल टेक्स्ट होता है",
                            "दोनों में कोई अंतर नहीं है"
                        ],
                        "correct_index": 0,
                        "explanation": "<head> पेज की तकनीकी जानकारी (मेटाडेटा, टाइटल, लिंक्स) रखता है, जबकि <body> स्क्रीन पर दिखने वाला दृश्य भाग है।",
                        "topic": "Document Structure",
                        "timestamp_ref": "00:04:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "HTML में हेडिंग पदानुक्रम (<h1> - <h6>) के बारे में कौन सा कथन सही है?",
                        "options": [
                            "<h1> सबसे महत्वपूर्ण हेडिंग है और पूरे पेज पर आदर्श रूप से केवल एक बार प्रयोग होना चाहिए",
                            "<h6> सबसे बड़ा हेडिंग फॉन्ट प्रदर्शित करता है",
                            "हेडिंग्स केवल स्टाइलिंग के लिए होती हैं, इनका SEO से कोई संबंध नहीं है",
                            "एक पेज पर कम से कम पांच <h1> टैग्स होने चाहिए"
                        ],
                        "correct_index": 0,
                        "explanation": "<h1> पेज का मुख्य विषय व्यक्त करता है और सर्च इंजनों व एक्सेसिबिलिटी के लिए प्रति पेज एक <h1> होना चाहिए।",
                        "topic": "Heading Hierarchy",
                        "timestamp_ref": "00:08:30",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "हाइपरलिंक (<a> टैग) पर क्लिक करने पर लिंक को नए ब्राउज़र टैब में खोलने के लिए कौन सा एट्रिब्यूट प्रयोग होता है?",
                        "options": ["target='_blank'", "open='new'", "window='tab'", "href='blank'"],
                        "correct_index": 0,
                        "explanation": "target='_blank' ब्राउज़र को लिंक को नए ब्राउज़िंग कॉन्टेक्स्ट (टैब/विंडो) में लोड करने का निर्देश देता है।",
                        "topic": "Anchor Target Attribute",
                        "timestamp_ref": "00:12:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "<img> टैग में alt एट्रिब्यूट का प्राथमिक उद्देश्य क्या है?",
                        "options": [
                            "इमेज लोड न होने पर वर्णनात्मक टेक्स्ट दिखाना और दृष्टिबाधित छात्रों (स्क्रीन रीडर) को समझाना",
                            "इमेज की चौड़ाई बढ़ाना",
                            "इमेज पर वॉटरमार्क लगाना",
                            "इमेज का बैकग्राउंड रंग बदलना"
                        ],
                        "correct_index": 0,
                        "explanation": "alt (alternative text) स्क्रीन रीडर्स द्वारा पढ़ा जाता है और टूटे हुए इमेज पाथ पर टेक्स्ट प्रदर्शित करता है।",
                        "topic": "Image Alt Accessibility",
                        "timestamp_ref": "00:15:20",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "VS Code में संपूर्ण HTML5 बॉयलरप्लेट कोड सेकंडों में जनरेट करने के लिए कौन सा Emmet शॉर्टकट उपयोग किया जाता है?",
                        "options": ["!", "html:boilerplate", "Ctrl + Alt + H", "#start"],
                        "correct_index": 0,
                        "explanation": "VS Code में विस्मयादिबोधक चिह्न (!) टाइप करके Tab या Enter दबाने पर पूरा HTML5 ढांचा तैयार हो जाता है।",
                        "topic": "Emmet Shortcuts",
                        "timestamp_ref": "00:03:30",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "निम्न में से कौन सा HTML एलिमेंट एक 'Void' (सेल्फ-क्लोजिंग) एलिमेंट है जिसका कोई क्लोजिंग टैग नहीं होता?",
                        "options": ["<img>", "<p>", "<div>", "<h1>"],
                        "correct_index": 0,
                        "explanation": "<img>, <meta>, <br>, और <input> जैसे टैग्स वॉयड एलिमेंट्स हैं जिन्हें अलग से बंद नहीं किया जाता।",
                        "topic": "Void Elements",
                        "timestamp_ref": "00:16:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "PhysicsWallah वेबसाइट पर नेविगेट करने के लिए सही एंकर टैग सिंटैक्स कौन सा है?",
                        "options": [
                            "<a href='https://pw.live'>PhysicsWallah</a>",
                            "<link to='https://pw.live'>PhysicsWallah</link>",
                            "<anchor url='https://pw.live'>PhysicsWallah</anchor>",
                            "<url href='https://pw.live'>PhysicsWallah</url>"
                        ],
                        "correct_index": 0,
                        "explanation": "<a> टैग href (Hypertext Reference) एट्रिब्यूट के साथ हाइपरलिंक बनाने का आधिकारिक सिंटैक्स है।",
                        "topic": "Hyperlink Syntax",
                        "timestamp_ref": "00:11:45",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "VS Code में 'Live Server' एक्सटेंशन का क्या कार्य है?",
                        "options": [
                            "HTML फाइल सेव करते ही ब्राउज़र को स्वचालित रूप से रीलोड करना",
                            "कोड में बग्स को स्वचालित ठीक करना",
                            "HTML को Python में बदलना",
                            "वेबसाइट को इंटरनेट पर पब्लिश करना"
                        ],
                        "correct_index": 0,
                        "explanation": "Live Server स्थानीय सर्वर चलाता है और फाइल सेव होते ही ब्राउज़र को ऑटो-रिफ्रेश करता है।",
                        "topic": "Developer Tooling",
                        "timestamp_ref": "00:05:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "SEO और वेब मानकों के अनुसार, प्रति वेबपेज कितने <h1> टैग्स होने चाहिए?",
                        "options": ["केवल 1", "जितने मन चाहे उतने", "कम से कम 10", "बिल्कुल शून्य"],
                        "correct_index": 0,
                        "explanation": "प्रत्येक पेज का एक ही मुख्य शीर्षक होना चाहिए, इसलिए प्रति पेज केवल एक <h1> अनुशंसित है।",
                        "topic": "SEO Semantic Heading",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    }
                ]
            elif language == "Hinglish":
                bank = [
                    {
                        "question": "HTML5 document ke top par <!DOCTYPE html> likhne ka exact purpose kya hota hai?",
                        "options": [
                            "Browser ko inform karna ki document modern HTML5 standard mein render karna hai",
                            "Website ka domain name register karna",
                            "Database connection establish karna",
                            "CSS file automatically generate karna"
                        ],
                        "correct_index": 0,
                        "explanation": "<!DOCTYPE html> browser ko quirks mode se bacha kar standard HTML5 rendering deta hai.",
                        "topic": "HTML5 Doctype",
                        "timestamp_ref": "00:02:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "HTML structure mein <head> aur <body> elements ke responsibilities mein kya difference hota hai?",
                        "options": [
                            "<head> contains metadata and page settings; <body> contains visible user content",
                            "<head> sirf mobile devices par dikhta hai",
                            "<head> optional hota hai par <body> nahi",
                            "Dono exactly same content render karte hain"
                        ],
                        "correct_index": 0,
                        "explanation": "<head> mein background metadata aur <body> mein actual webpage UI hota hai.",
                        "topic": "Document Structure",
                        "timestamp_ref": "00:04:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Semantic headings (<h1> se <h6>) ke usage ke baare mein kaunsa rule correct hai?",
                        "options": [
                            "<h1> main page topic represent karta hai aur single time use hona chahiye",
                            "<h6> sabse bada font display karta hai",
                            "Headings se SEO par koi effect nahi padta",
                            "Har paragraph ke baad ek <h1> lagana mandatory hai"
                        ],
                        "correct_index": 0,
                        "explanation": "Page par single <h1> best SEO and semantic hierarchy practice hai.",
                        "topic": "Heading Hierarchy",
                        "timestamp_ref": "00:08:30",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Anchor tag (<a>) par click karne par link ko new browser tab mein open karne ke liye kaunsa attribute use karte hain?",
                        "options": ["target='_blank'", "open='new'", "window='tab'", "href='blank'"],
                        "correct_index": 0,
                        "explanation": "target='_blank' destination URL ko separate tab mein open karta hai.",
                        "topic": "Anchor Target Attribute",
                        "timestamp_ref": "00:12:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "<img> tag mein alt attribute specify karna kyu compulsory maana jata hai?",
                        "options": [
                            "Accessibility screen readers aur image broken hone par text description display karne ke liye",
                            "Image size reduce karne ke liye",
                            "Image par border lagane ke liye",
                            "Animation speed control karne ke liye"
                        ],
                        "correct_index": 0,
                        "explanation": "alt text screen reader users aur SEO bots ke liye essential hota hai.",
                        "topic": "Image Alt Accessibility",
                        "timestamp_ref": "00:15:20",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "VS Code mein instant HTML boilerplate code create karne ka standard Emmet shortcut kya hai?",
                        "options": ["!", "html:5", "Ctrl + B", "#boilerplate"],
                        "correct_index": 0,
                        "explanation": "Exclamation mark (!) type karke Tab ya Enter press karne se instant boilerplate expand hota hai.",
                        "topic": "Emmet Shortcuts",
                        "timestamp_ref": "00:03:30",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "Neeche diye gaye tags mein se kaunsa tag ek self-closing (void) element hai?",
                        "options": ["<img>", "<p>", "<div>", "<h1>"],
                        "correct_index": 0,
                        "explanation": "<img> tag void element hota hai jiska separate closing </img > tag nahi hota.",
                        "topic": "Void Elements",
                        "timestamp_ref": "00:16:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "External website par hyperlink create karne ka valid HTML syntax kaunsa hai?",
                        "options": [
                            "<a href='https://pw.live'>PhysicsWallah</a>",
                            "<link to='https://pw.live'>PhysicsWallah</link>",
                            "<anchor url='https://pw.live'>PhysicsWallah</anchor>",
                            "<url link='https://pw.live'>PhysicsWallah</url>"
                        ],
                        "correct_index": 0,
                        "explanation": "<a> element with href attribute is standard HTML hyperlink syntax.",
                        "topic": "Hyperlink Syntax",
                        "timestamp_ref": "00:11:45",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "Web development mein VS Code ka 'Live Server' extension student developers ko kya benefit deta hai?",
                        "options": [
                            "File save karte hi browser automatically reload ho jata hai",
                            "HTML code automatically minify kar deta hai",
                            "Python code ko HTML mein translate karta hai",
                            "GitHub repository create karta hai"
                        ],
                        "correct_index": 0,
                        "explanation": "Live Server auto hot-reload provide karta hai file save karte hi.",
                        "topic": "Developer Tooling",
                        "timestamp_ref": "00:05:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "SEO aur document accessibility best practices ke mutabiq per web page kitne <h1> tags hone chahiye?",
                        "options": ["Exactly 1", "Unlimited", "Minimum 5", "Zero"],
                        "correct_index": 0,
                        "explanation": "Har document ka single main heading <h1> hota hai.",
                        "topic": "SEO Semantic Heading",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    }
                ]
            else:  # English
                bank = [
                    {
                        "question": "What is the critical purpose of placing '<!DOCTYPE html>' on the very first line of an HTML document?",
                        "options": [
                            "It informs the browser engine to render the page in standard HTML5 mode rather than legacy quirks mode",
                            "It assigns an administrative password to the web directory",
                            "It loads the default browser stylesheet",
                            "It registers the DNS server hostname"
                        ],
                        "correct_index": 0,
                        "explanation": "The '<!DOCTYPE html>' directive signals modern HTML5 document parsing to avoid legacy quirks mode rendering.",
                        "topic": "HTML5 Doctype",
                        "timestamp_ref": "00:02:15",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "What primary architectural difference distinguishes the <head> element from the <body> element in HTML?",
                        "options": [
                            "The <head> contains page metadata and settings; the <body> contains all visual document content",
                            "The <head> is rendered on mobile only; the <body> renders on desktop",
                            "The <head> contains graphical media; the <body> contains typography",
                            "There is no distinction; they are interchangeable"
                        ],
                        "correct_index": 0,
                        "explanation": "The <head> element hosts invisible metadata, character sets, and assets; <body> encloses visible layout elements.",
                        "topic": "Document Structure",
                        "timestamp_ref": "00:04:10",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "Which principle represents semantic best practice regarding heading hierarchy (<h1> through <h6>)?",
                        "options": [
                            "Use exactly one <h1> per page to define the primary subject for users and search engine crawlers",
                            "The <h6> tag displays the largest typography",
                            "Heading tags carry no semantic significance and serve purely as visual styling shortcuts",
                            "Every paragraph requires a preceding <h1>"
                        ],
                        "correct_index": 0,
                        "explanation": "An HTML document must maintain a logical outline; exactly one <h1> designates the primary topic of the page.",
                        "topic": "Heading Hierarchy",
                        "timestamp_ref": "00:08:30",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    },
                    {
                        "question": "Which attribute on an anchor tag (<a>) opens the destination URL in a new browser tab?",
                        "options": ["target='_blank'", "open='tab'", "window='new'", "href='new_window'"],
                        "correct_index": 0,
                        "explanation": "Setting target='_blank' instructs the browser to open the linked resource in a new browsing context.",
                        "topic": "Anchor Target Attribute",
                        "timestamp_ref": "00:12:15",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "What is the primary role of the 'alt' attribute on the <img> element?",
                        "options": [
                            "It provides alternative text for accessibility screen readers and fallback text if image loading fails",
                            "It sets the image border radius dynamically",
                            "It defines the high-DPI retina multiplier",
                            "It forces automatic image caching on proxy servers"
                        ],
                        "correct_index": 0,
                        "explanation": "The 'alt' attribute provides descriptive alternative text indispensable for screen reader users and SEO indexers.",
                        "topic": "Image Alt Accessibility",
                        "timestamp_ref": "00:15:20",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "Which Emmet abbreviation in VS Code expands into the complete HTML5 boilerplate structure?",
                        "options": ["!", "html:boilerplate", "Ctrl + Alt + H", "#start"],
                        "correct_index": 0,
                        "explanation": "Typing '!' (exclamation mark) and pressing Tab or Enter in VS Code immediately generates the full HTML5 scaffold.",
                        "topic": "Emmet Shortcuts",
                        "timestamp_ref": "00:03:30",
                        "supporting_chunk_ids": [cids[0]] if cids else []
                    },
                    {
                        "question": "Which of the following HTML elements is a 'void' (self-closing) element without an end tag?",
                        "options": ["<img>", "<p>", "<div>", "<h1>"],
                        "correct_index": 0,
                        "explanation": "Void elements like <img>, <br>, <meta>, and <hr> cannot have child nodes and omit closing tags.",
                        "topic": "Void Elements",
                        "timestamp_ref": "00:16:05",
                        "supporting_chunk_ids": [cids[4]] if len(cids) > 4 else cids
                    },
                    {
                        "question": "What is the correct and valid syntax to link to the PhysicsWallah website?",
                        "options": [
                            "<a href='https://pw.live'>PhysicsWallah</a>",
                            "<link to='https://pw.live'>PhysicsWallah</link>",
                            "<anchor url='https://pw.live'>PhysicsWallah</anchor>",
                            "<url link='https://pw.live'>PhysicsWallah</url>"
                        ],
                        "correct_index": 0,
                        "explanation": "The anchor tag (<a>) pairing with 'href' constitutes standard HTML hyperlink syntax.",
                        "topic": "Hyperlink Syntax",
                        "timestamp_ref": "00:11:45",
                        "supporting_chunk_ids": [cids[3]] if len(cids) > 3 else cids
                    },
                    {
                        "question": "How does the 'Live Server' extension in VS Code streamline web development workflow?",
                        "options": [
                            "It hosts files on a local development server and triggers automatic browser reload on file save",
                            "It minifies JavaScript and CSS bundles into production builds",
                            "It converts HTML tags into React components automatically",
                            "It deploys static HTML directly to a public cloud CDN"
                        ],
                        "correct_index": 0,
                        "explanation": "Live Server spawns a local development web server and injects live-reload web sockets upon file modifications.",
                        "topic": "Developer Tooling",
                        "timestamp_ref": "00:05:50",
                        "supporting_chunk_ids": [cids[1]] if len(cids) > 1 else cids
                    },
                    {
                        "question": "According to web standards and SEO guidelines, how many <h1> elements should appear per page?",
                        "options": ["Exactly 1", "As many as desired", "A minimum of 5", "Zero"],
                        "correct_index": 0,
                        "explanation": "Best practices dictate a single <h1> heading per document representing the overarching topic.",
                        "topic": "SEO Semantic Heading",
                        "timestamp_ref": "00:09:40",
                        "supporting_chunk_ids": [cids[2]] if len(cids) > 2 else cids
                    }
                ]

        # Select requested question count (5 or 10)
        selected_questions = bank[:num_questions]
        # Assign 1-based IDs
        for i, q in enumerate(selected_questions):
            q["id"] = i + 1

        quiz_data = {
            "quiz_id": f"QUIZ_{lecture_id}_{difficulty.upper()}_{num_questions}Q",
            "lecture_id": lecture_id,
            "lecture_title": lecture_title,
            "difficulty": difficulty,
            "num_questions": len(selected_questions),
            "language": language,
            "questions": selected_questions,
            "generated_at": time.time(),
            "is_mock": True
        }

        return quiz_data

    def evaluate_quiz_submission(
        self,
        quiz_data: Dict[str, Any],
        user_answers: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Evaluate student quiz submission entirely client-side with zero LLM API consumption.
        Computes score, percentage, grade, review cards, and topic revision recommendations.
        """
        questions = quiz_data.get("questions", [])
        total = len(questions)
        if total == 0:
            return {
                "score": 0,
                "total": 0,
                "percentage": 0.0,
                "passed": False,
                "grade": "N/A",
                "feedback_message": "No questions in quiz.",
                "question_results": [],
                "strong_topics": [],
                "weak_topics": []
            }

        correct_count = 0
        question_results = []
        topic_performance: Dict[str, Dict[str, Any]] = {}

        for q in questions:
            qid = q.get("id", 0)
            opts = q.get("options", [])
            corr_idx = q.get("correct_index", 0)
            topic = q.get("topic", "General Concept")
            ts_ref = q.get("timestamp_ref", "00:00")
            cids = q.get("supporting_chunk_ids", [])

            user_idx = user_answers.get(qid)
            is_correct = (user_idx is not None and int(user_idx) == corr_idx)

            if is_correct:
                correct_count += 1

            # Track topic stats
            if topic not in topic_performance:
                topic_performance[topic] = {
                    "total": 0,
                    "correct": 0,
                    "timestamp_ref": ts_ref,
                    "chunk_ids": cids
                }
            topic_performance[topic]["total"] += 1
            if is_correct:
                topic_performance[topic]["correct"] += 1

            question_results.append({
                "id": qid,
                "question": q.get("question", ""),
                "options": opts,
                "user_answer": user_idx,
                "user_answer_text": opts[user_idx] if user_idx is not None and 0 <= user_idx < len(opts) else "Not Answered",
                "correct_answer": corr_idx,
                "correct_answer_text": opts[corr_idx] if 0 <= corr_idx < len(opts) else "Unknown",
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
                "topic": topic,
                "timestamp_ref": ts_ref,
                "supporting_chunk_ids": cids
            })

        percentage = round((correct_count / total) * 100, 1)
        passed = percentage >= 60.0

        if percentage == 100.0:
            grade = "🌟 Outstanding (A+)"
            msg = "Flawless score! You have completely mastered this lecture's concepts."
        elif percentage >= 80.0:
            grade = "🎉 Proficient (A)"
            msg = "Great job! Strong conceptual understanding across topics."
        elif percentage >= 60.0:
            grade = "👍 Good Effort (B)"
            msg = "You passed the quiz! A few concepts require quick review."
        elif percentage >= 40.0:
            grade = "📚 Needs Revision (C)"
            msg = "Review the recommended timestamps below to reinforce core ideas."
        else:
            grade = "🛑 Needs Practice (D)"
            msg = "Significant gaps identified. We recommend re-watching key lecture sections."

        # Topic analytics
        strong_topics = []
        weak_topics = []

        for top, data in topic_performance.items():
            if data["correct"] == data["total"]:
                strong_topics.append(top)
            else:
                weak_topics.append({
                    "topic": top,
                    "accuracy": f"{data['correct']}/{data['total']}",
                    "recommended_timestamp": data["timestamp_ref"],
                    "supporting_chunk_ids": data["chunk_ids"],
                    "lecture_id": quiz_data.get("lecture_id", "")
                })

        return {
            "score": correct_count,
            "total": total,
            "percentage": percentage,
            "passed": passed,
            "grade": grade,
            "feedback_message": msg,
            "question_results": question_results,
            "strong_topics": strong_topics,
            "weak_topics": weak_topics
        }

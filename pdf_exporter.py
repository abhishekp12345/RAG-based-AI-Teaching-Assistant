"""
PDF Exporter Module for LearnMate AI.
Generates publication-quality, A4-formatted revision notes PDFs directly in memory.
Features:
- Full Unicode & multilingual support (English, Hindi Devanagari, Hinglish) via Noto Sans fonts.
- Complex script text shaping via uharfbuzz.
- Clean EdTech visual hierarchy (Blue-and-White theme, cards, code blocks, tables).
- Running headers, footers, page numbering (Page X of Y), and application URL metadata.
- Zero LLM requests & zero external API quota consumption.
"""

import io
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from fpdf import FPDF


# Color Palette Definitions (RGB tuples)
COLOR_PRIMARY_BLUE = (30, 58, 138)     # #1E3A8A - Dark Navy
COLOR_SECONDARY_BLUE = (37, 99, 235)   # #2563EB - Accent Royal Blue
COLOR_LIGHT_BLUE = (239, 246, 255)     # #EFF6FF - Soft Tint
COLOR_DARK_SLATE = (30, 41, 59)        # #1E293B - Body Text
COLOR_MUTED_SLATE = (100, 116, 139)    # #64748B - Captions & Subtitles
COLOR_BORDER_GRAY = (226, 232, 240)    # #E2E8F0 - Divider Borders
COLOR_CODE_BG = (248, 250, 252)        # #F8FAFC - Code Background
COLOR_CODE_BORDER = (203, 213, 225)    # #CBD5E1 - Code Border
COLOR_WARN_BG = (254, 243, 199)        # #FEF3C7 - Warning Amber Tint
COLOR_WARN_BORDER = (245, 158, 11)     # #F59E0B - Warning Amber Border
COLOR_WARN_TEXT = (146, 64, 14)        # #92400E - Warning Text
COLOR_SUCCESS_BG = (236, 253, 245)     # #ECFDF5 - Success Tint
COLOR_SUCCESS_BORDER = (16, 185, 129)  # #10B981 - Success Border
COLOR_SUCCESS_TEXT = (6, 95, 70)       # #065F46 - Success Text
COLOR_WHITE = (255, 255, 255)


class SmartNotesPDF(FPDF):
    """Custom FPDF2 subclass with professional header, footer, and Unicode typography."""

    def __init__(
        self,
        course_title: str,
        lecture_title: str,
        notes_style: str,
        language: str,
        font_dir: Optional[Path] = None,
        *args,
        **kwargs
    ):
        super().__init__(orientation="P", unit="mm", format="A4", *args, **kwargs)
        self.course_title = course_title
        self.lecture_title = lecture_title
        self.notes_style = notes_style
        self.language = language
        self.font_dir = font_dir or (Path(__file__).resolve().parent / "assets" / "fonts")
        
        # Configure layout geometry
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(left=15, top=14, right=15)
        
        # Enable text shaping for proper Devanagari ligature and matra rendering
        self.set_text_shaping(True)
        
        # Register Unicode fonts
        self._setup_fonts()
        
        # Default font family based on language
        self.is_devanagari = (
            language.lower() in ("hindi", "hinglish") or
            any('\u0900' <= char <= '\u097f' for char in f"{course_title} {lecture_title}")
        )
        self.primary_font = "NotoDev" if self.is_devanagari else "NotoSans"
        self.code_font = "NotoSans"

    def _setup_fonts(self):
        """Register Noto Sans and Noto Sans Devanagari fonts."""
        noto_reg = self.font_dir / "NotoSans-Regular.ttf"
        noto_bold = self.font_dir / "NotoSans-Bold.ttf"
        dev_reg = self.font_dir / "NotoSansDevanagari-Regular.ttf"
        dev_bold = self.font_dir / "NotoSansDevanagari-Bold.ttf"

        if dev_reg.exists() and dev_bold.exists():
            self.add_font("NotoDev", "", str(dev_reg))
            self.add_font("NotoDev", "B", str(dev_bold))
        
        if noto_reg.exists() and noto_bold.exists():
            self.add_font("NotoSans", "", str(noto_reg))
            self.add_font("NotoSans", "B", str(noto_bold))

    def header(self):
        """Running header on pages 2 and onward."""
        if self.page_no() > 1:
            self.set_font(self.primary_font, "", 8)
            self.set_text_color(*COLOR_MUTED_SLATE)
            self.cell(100, 6, text=f"LearnMate AI  |  {self.course_title}", align="L")
            self.set_font(self.primary_font, "B", 8)
            self.set_text_color(*COLOR_SECONDARY_BLUE)
            self.cell(0, 6, text=self.lecture_title[:50], align="R")
            self.ln(7)
            # Thin divider rule
            self.set_draw_color(*COLOR_BORDER_GRAY)
            self.set_line_width(0.2)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(4)

    def footer(self):
        """Standardized educational footer on every page."""
        self.set_y(-13)
        self.set_draw_color(*COLOR_BORDER_GRAY)
        self.set_line_width(0.2)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        
        # Left URL & Attribution
        self.set_font(self.primary_font, "", 7.5)
        self.set_text_color(*COLOR_MUTED_SLATE)
        self.cell(120, 5, text="https://learnmate-ai-rag.streamlit.app  •  Your Personal AI Teaching Assistant", align="L")
        
        # Right Page Numbers
        self.set_font(self.primary_font, "B", 8)
        self.set_text_color(*COLOR_PRIMARY_BLUE)
        self.cell(0, 5, text=f"Page {self.page_no()} of {{nb}}", align="R")


def render_banner(pdf: SmartNotesPDF, notes_data: Dict[str, Any]):
    """Render high-impact cover banner with branding and lecture metadata."""
    # Top Gradient-style banner box
    pdf.set_fill_color(*COLOR_PRIMARY_BLUE)
    pdf.rect(15, 14, 180, 28, "F")

    # Brand Title
    pdf.set_xy(18, 17)
    pdf.set_font(pdf.primary_font, "B", 15)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.cell(174, 8, text="🎓 LearnMate AI — Smart Revision Notes", align="L")

    # Course & Lecture Subtitle
    pdf.set_xy(18, 26)
    pdf.set_font(pdf.primary_font, "", 10)
    pdf.set_text_color(*COLOR_LIGHT_BLUE)
    c_title = notes_data.get("course_title", "Web Development Fundamentals")
    l_title = notes_data.get("lecture_title", f"Lecture {notes_data.get('lecture_id', '')}")
    pdf.cell(174, 6, text=f"Course: {c_title}  •  {l_title}", align="L")

    pdf.set_xy(18, 34)
    pdf.set_font(pdf.primary_font, "", 8)
    pdf.set_text_color(203, 213, 225)
    mode_text = "Verified Offline Curriculum" if notes_data.get("is_mock", True) else "Live AI Synthesis"
    pdf.cell(174, 5, text=f"Generated: {mode_text}  •  Zero-Quota Demonstration", align="L")

    pdf.set_y(46)

    # Metadata Pills Row
    style_label = notes_data.get("notes_style", "Detailed Study").replace("_", " ").title()
    lang_label = notes_data.get("language", "English")

    pdf.set_font(pdf.primary_font, "B", 8.5)
    
    # Pill 1: Style
    pdf.set_fill_color(*COLOR_LIGHT_BLUE)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(50, 7, text=f"  📌 Style: {style_label}", fill=True, border=1)
    pdf.cell(5, 7, text="")
    
    # Pill 2: Language
    pdf.set_fill_color(*COLOR_LIGHT_BLUE)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(45, 7, text=f"  🌐 Language: {lang_label}", fill=True, border=1)
    pdf.cell(5, 7, text="")
    
    # Pill 3: Model / Provenance
    pdf.set_fill_color(*COLOR_CODE_BG)
    pdf.set_text_color(*COLOR_DARK_SLATE)
    pdf.cell(75, 7, text=f"  ⚙️ Engine: {notes_data.get('generating_model', 'intfloat/multilingual-e5-small')}", fill=True, border=1)

    pdf.ln(11)


def render_executive_summary(pdf: SmartNotesPDF, summary: str):
    """Render executive summary card."""
    if not summary:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="1. Executive Summary", new_x="LMARGIN", new_y="NEXT")

    # Highlight box with left accent border
    pdf.set_fill_color(*COLOR_LIGHT_BLUE)
    pdf.set_draw_color(*COLOR_SECONDARY_BLUE)
    pdf.set_line_width(0.8)
    
    # Measure text height to size bounding box
    start_y = pdf.get_y()
    pdf.set_font(pdf.primary_font, "", 9.5)
    pdf.set_text_color(*COLOR_DARK_SLATE)
    
    # Compute lines required
    pdf.set_x(18)
    pdf.multi_cell(174, 5.5, text=summary, fill=True, border=0)
    end_y = pdf.get_y()
    
    # Draw left border bar
    pdf.set_draw_color(*COLOR_SECONDARY_BLUE)
    pdf.set_line_width(1.2)
    pdf.line(16, start_y, 16, end_y)
    pdf.ln(4)


def render_learning_objectives(pdf: SmartNotesPDF, objectives: List[str]):
    """Render bulleted learning objectives list."""
    if not objectives:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="2. Learning Objectives", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(pdf.primary_font, "", 9.5)
    for obj in objectives:
        pdf.set_x(16)
        pdf.set_text_color(*COLOR_SECONDARY_BLUE)
        pdf.cell(6, 5.5, text="✓", align="C")
        pdf.set_text_color(*COLOR_DARK_SLATE)
        pdf.multi_cell(168, 5.5, text=obj)
    pdf.ln(3)


def render_sections(pdf: SmartNotesPDF, sections: List[Dict[str, Any]]):
    """Render structured content breakdown, timestamps, and code snippets."""
    if not sections:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="3. Core Concepts & Lecture Breakdown", new_x="LMARGIN", new_y="NEXT")

    for idx, sec in enumerate(sections):
        # Section Heading
        heading = sec.get("heading", f"Topic {idx + 1}")
        pdf.set_font(pdf.primary_font, "B", 10.5)
        pdf.set_text_color(*COLOR_PRIMARY_BLUE)
        pdf.cell(0, 6.5, text=heading, new_x="LMARGIN", new_y="NEXT")

        # Timestamps and Citations Badge
        citations = sec.get("citations", [])
        if citations:
            ts_parts = [f"[{c.get('timestamp', '00:00')}]" for c in citations[:3]]
            ts_str = "Timestamp Ref: " + " & ".join(ts_parts)
            pdf.set_font(pdf.primary_font, "B", 8)
            pdf.set_text_color(*COLOR_MUTED_SLATE)
            pdf.cell(0, 4.5, text=ts_str, new_x="LMARGIN", new_y="NEXT")

        # Narrative Content
        content = sec.get("content", "")
        if content:
            pdf.set_font(pdf.primary_font, "", 9.2)
            pdf.set_text_color(*COLOR_DARK_SLATE)
            pdf.multi_cell(0, 5.2, text=content)
            pdf.ln(2)

        # Formatted Code Snippets
        snippets = sec.get("code_snippets", [])
        for snip in snippets:
            if snip and snip.strip():
                pdf.set_fill_color(*COLOR_CODE_BG)
                pdf.set_draw_color(*COLOR_CODE_BORDER)
                pdf.set_line_width(0.3)
                pdf.set_font(pdf.code_font, "", 8.2)
                pdf.set_text_color(*COLOR_DARK_SLATE)
                
                # Render code box with indentation
                pdf.set_x(18)
                pdf.multi_cell(174, 4.8, text=snip.strip(), fill=True, border=1)
                pdf.ln(2)
        pdf.ln(2)


def render_definitions(pdf: SmartNotesPDF, definitions: List[Dict[str, Any]]):
    """Render definitions and terminology table."""
    if not definitions:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="4. Key Definitions & Terminology", new_x="LMARGIN", new_y="NEXT")

    # Table Header
    pdf.set_fill_color(*COLOR_PRIMARY_BLUE)
    pdf.set_text_color(*COLOR_WHITE)
    pdf.set_font(pdf.primary_font, "B", 8.5)
    pdf.cell(42, 6.5, text=" Term", fill=True, border=1)
    pdf.cell(115, 6.5, text=" Verified Definition", fill=True, border=1)
    pdf.cell(23, 6.5, text=" Timestamp", fill=True, border=1, align="C")
    pdf.ln(6.5)

    # Table Rows
    pdf.set_font(pdf.primary_font, "", 8.5)
    pdf.set_draw_color(*COLOR_BORDER_GRAY)
    
    for row_idx, item in enumerate(definitions):
        term = item.get("term", "")
        defn = item.get("definition", "")
        ts = item.get("timestamp_ref", "00:00")

        # Alternate row fill
        fill_color = COLOR_WHITE if row_idx % 2 == 0 else COLOR_LIGHT_BLUE
        pdf.set_fill_color(*fill_color)
        
        # Calculate dynamic row height
        line_count = max(len(defn) // 65 + 1, 1)
        row_h = max(line_count * 4.8, 6.5)

        start_x = 15
        start_y = pdf.get_y()
        
        # Check if page break is needed
        if start_y + row_h > 275:
            pdf.add_page()
            start_y = pdf.get_y()

        # Column 1: Term
        pdf.set_xy(start_x, start_y)
        pdf.set_font(pdf.primary_font, "B", 8.5)
        pdf.set_text_color(*COLOR_PRIMARY_BLUE)
        pdf.multi_cell(42, row_h, text=term, fill=True, border=1)

        # Column 2: Definition
        pdf.set_xy(start_x + 42, start_y)
        pdf.set_font(pdf.primary_font, "", 8.2)
        pdf.set_text_color(*COLOR_DARK_SLATE)
        pdf.multi_cell(115, 4.8, text=defn, fill=True, border=1)

        # Column 3: Timestamp
        pdf.set_xy(start_x + 42 + 115, start_y)
        pdf.set_font(pdf.primary_font, "B", 8)
        pdf.set_text_color(*COLOR_SECONDARY_BLUE)
        pdf.cell(23, row_h, text=f"[{ts}]", fill=True, border=1, align="C")

        pdf.set_y(start_y + row_h)

    pdf.ln(4)


def render_pitfalls(pdf: SmartNotesPDF, pitfalls: List[str]):
    """Render common mistakes & pitfalls warning callout."""
    if not pitfalls:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="5. Common Pitfalls & Exam Traps", new_x="LMARGIN", new_y="NEXT")

    pdf.set_fill_color(*COLOR_WARN_BG)
    pdf.set_draw_color(*COLOR_WARN_BORDER)
    pdf.set_line_width(0.6)

    for pit in pitfalls:
        pdf.set_font(pdf.primary_font, "", 8.8)
        pdf.set_text_color(*COLOR_WARN_TEXT)
        pdf.set_x(16)
        pdf.multi_cell(178, 5.2, text=f"⚠️  {pit}", fill=True, border=1)
        pdf.ln(1.5)
    pdf.ln(3)


def render_checklist(pdf: SmartNotesPDF, checklist: List[str]):
    """Render revision checklist with checkbox squares."""
    if not checklist:
        return

    pdf.set_font(pdf.primary_font, "B", 11.5)
    pdf.set_text_color(*COLOR_PRIMARY_BLUE)
    pdf.cell(0, 7, text="6. Quick Revision Checklist", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(pdf.primary_font, "", 9)
    pdf.set_text_color(*COLOR_DARK_SLATE)

    for item in checklist:
        pdf.set_x(16)
        pdf.cell(6, 5.5, text="[  ]", align="C")
        pdf.multi_cell(172, 5.5, text=item)
    pdf.ln(3)


def generate_notes_pdf(notes_data: Dict[str, Any]) -> bytes:
    """
    Main entry point: Generates a complete, verified PDF in memory from notes_data.
    Returns bytes suitable for Streamlit st.download_button.
    """
    course_title = notes_data.get("course_title", "Web Development Fundamentals")
    lecture_title = notes_data.get("lecture_title", f"Lecture {notes_data.get('lecture_id', '')}")
    notes_style = notes_data.get("notes_style", "detailed_study")
    language = notes_data.get("language", "English")

    pdf = SmartNotesPDF(
        course_title=course_title,
        lecture_title=lecture_title,
        notes_style=notes_style,
        language=language
    )
    
    # Total pages alias for Page X of Y
    pdf.alias_nb_pages()
    pdf.add_page()

    # 1. Top Cover Banner & Metadata
    render_banner(pdf, notes_data)

    # 2. Executive Summary
    render_executive_summary(pdf, notes_data.get("executive_summary", ""))

    # 3. Learning Objectives
    render_learning_objectives(pdf, notes_data.get("learning_objectives", []))

    # 4. Core Topics Breakdown & Code Snippets
    render_sections(pdf, notes_data.get("sections", []))

    # 5. Definitions Table
    render_definitions(pdf, notes_data.get("definitions_and_terms", []))

    # 6. Common Pitfalls Warning Box
    render_pitfalls(pdf, notes_data.get("common_pitfalls", []))

    # 7. Revision Checklist
    render_checklist(pdf, notes_data.get("revision_checklist", []))

    # Return pure bytes
    return bytes(pdf.output())

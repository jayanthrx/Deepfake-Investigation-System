"""
generate_ppt.py
Deepfake Investigation System — Professional Academic & Technical PowerPoint Generator
Fully covers the 7 required sections:
1. Introduction
2. Literature Review
3. Advantages
4. Software and Hardware Requirements
5. Architecture
6. Data Flow / Use Case Diagrams
7. Conclusion
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ─── Color Palette (Sophisticated Dark Cyber-Forensic Theme) ───────────────────
C_BG_DARK     = RGBColor(0x0A, 0x0F, 0x1D)   # Deep Midnight Blue / Charcoal
C_BG_CARD     = RGBColor(0x11, 0x18, 0x27)   # Dark Card Surface
C_BG_CARD_ALT = RGBColor(0x1A, 0x23, 0x3A)   # Elevated Card Surface
C_ACCENT_BLUE = RGBColor(0x3B, 0x82, 0xF6)   # Cobalt / Tech Blue
C_ACCENT_CYAN = RGBColor(0x06, 0xB6, 0xD4)   # Cyan / Teal Forensics
C_ACCENT_EMR  = RGBColor(0x10, 0xB9, 0x81)   # Emerald / Authentic
C_ACCENT_ROSE = RGBColor(0xF4, 0x3F, 0x5E)   # Crimson / Deepfake Alert
C_ACCENT_AMB  = RGBColor(0xF5, 0x9E, 0x0B)   # Amber / Caution
C_ACCENT_PURP = RGBColor(0x8B, 0x5C, 0xF6)   # Purple / AI Innovation
C_TEXT_MAIN   = RGBColor(0xF8, 0xFA, 0xFC)   # Crisp White
C_TEXT_MUTED  = RGBColor(0x94, 0xA3, 0xB8)   # Slate Grey
C_WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
C_DIVIDER     = RGBColor(0x1E, 0x29, 0x3B)   # Border Stroke
C_HEADER_ROW  = RGBColor(0x1E, 0x29, 0x3B)   # Table Header

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT   = os.path.join(BASE_DIR, "Deepfake_Investigation_System.pptx")


# ─── Helper Functions ─────────────────────────────────────────────────────────

def fill_bg(slide, color=C_BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(1)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    return shape


def add_rounded_rect(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(1)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    return shape


def add_oval(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(1)):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    return shape


def add_arrow(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_line(slide, left, top, width, color=C_ACCENT_CYAN, thickness=Pt(2)):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, thickness)
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()
    return line


def add_text(slide, text, left, top, width, height,
             font_size=16, bold=False, color=C_TEXT_MAIN,
             align=PP_ALIGN.LEFT, italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Segoe UI"
    return txBox


def add_header(slide, section_tag, title, slide_num, accent_color=C_ACCENT_CYAN):
    # Top accent line
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.05), accent_color)
    # Left accent line
    add_rect(slide, 0, 0, Inches(0.05), SLIDE_H, accent_color)
    
    # Section Tag Badge
    add_rounded_rect(slide, Inches(0.6), Inches(0.35), Inches(2.8), Inches(0.32), C_BG_CARD, accent_color, Pt(1))
    add_text(slide, section_tag.upper(), Inches(0.6), Inches(0.38), Inches(2.8), Inches(0.3),
             font_size=10, bold=True, color=accent_color, align=PP_ALIGN.CENTER)
    
    # Slide Number Badge
    add_rounded_rect(slide, Inches(12.1), Inches(0.35), Inches(0.7), Inches(0.32), C_BG_CARD, C_DIVIDER, Pt(1))
    add_text(slide, f"{slide_num:02d}", Inches(12.1), Inches(0.38), Inches(0.7), Inches(0.3),
             font_size=11, bold=True, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)
    
    # Main Slide Title
    add_text(slide, title, Inches(0.6), Inches(0.75), Inches(11.5), Inches(0.55),
             font_size=24, bold=True, color=C_TEXT_MAIN, align=PP_ALIGN.LEFT)
    
    # Separator Line
    add_line(slide, Inches(0.6), Inches(1.35), Inches(12.2), C_DIVIDER, Pt(1))


def add_bullets(slide, items, left, top, width, height,
                font_size=13, color=C_TEXT_MUTED, bullet_color="•",
                space_before=4):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)

    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(space_before)
        run_b = p.add_run()
        run_b.text = f"{bullet_color}  "
        run_b.font.size = Pt(font_size)
        run_b.font.bold = True
        run_b.font.color.rgb = C_ACCENT_CYAN
        run_b.font.name = "Segoe UI"
        
        run_t = p.add_run()
        run_t.text = item
        run_t.font.size = Pt(font_size)
        run_t.font.color.rgb = color
        run_t.font.name = "Segoe UI"
    return txBox


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE SLIDE
# ═══════════════════════════════════════════════════════════════════════════════
def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)

    # Decorative top and left glowing borders
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.08), C_ACCENT_CYAN)
    add_rect(slide, 0, 0, Inches(0.08), SLIDE_H, C_ACCENT_BLUE)

    # Cyber Grid Accent Box
    add_rounded_rect(slide, Inches(0.8), Inches(0.8), Inches(11.73), Inches(5.9), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(0.8), Inches(0.8), Inches(11.73), C_ACCENT_CYAN, Pt(3))

    # Badge Pill
    add_rounded_rect(slide, Inches(4.3), Inches(1.3), Inches(4.7), Inches(0.42), C_BG_CARD_ALT, C_ACCENT_CYAN, Pt(1.5))
    add_text(slide, "CYBER FORENSICS & COMPUTER VISION", Inches(4.3), Inches(1.38), Inches(4.7), Inches(0.35),
             font_size=11, bold=True, color=C_ACCENT_CYAN, align=PP_ALIGN.CENTER)

    # Main Title
    add_text(slide, "DEEPFAKE INVESTIGATION SYSTEM",
             Inches(1.0), Inches(2.0), Inches(11.3), Inches(0.9),
             font_size=36, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Subtitle
    add_text(slide, "Advanced AI-Powered Digital Forensics & Media Authenticity Verification Platform",
             Inches(1.2), Inches(2.95), Inches(11.0), Inches(0.5),
             font_size=16, italic=True, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

    # Key Highlights Badges Row
    badges = [
        ("EfficientNet-B3", C_ACCENT_BLUE),
        ("Grad-CAM XAI", C_ACCENT_CYAN),
        ("Dual-Stream Forensics", C_ACCENT_PURP),
        ("OpenCV Video Pipeline", C_ACCENT_EMR),
        ("Automated PDF Dossier", C_ACCENT_ROSE),
        ("Flask REST API", C_ACCENT_AMB),
    ]
    b_w = Inches(1.75)
    b_start = Inches(1.15)
    b_y = Inches(3.75)
    for label, col in badges:
        add_rounded_rect(slide, b_start, b_y, b_w, Inches(0.38), C_BG_CARD_ALT, col, Pt(1))
        add_text(slide, label, b_start, b_y + Inches(0.06), b_w, Inches(0.3),
                 font_size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
        b_start += Inches(1.85)

    # Divider Line inside Card
    add_line(slide, Inches(2.5), Inches(4.5), Inches(8.3), C_DIVIDER, Pt(1))

    # Meta Info Container (Presentation / Project Details)
    add_text(slide,
             "Project Presentation  |  Academic Evaluation  |  Digital Forensics & Deep Learning System",
             Inches(1.0), Inches(4.8), Inches(11.33), Inches(0.35),
             font_size=13, bold=True, color=C_TEXT_MAIN, align=PP_ALIGN.CENTER)

    add_text(slide,
             "Featuring: Frame-by-Frame Video Forensics • Explainable Heatmaps • Forensic Audit Trails",
             Inches(1.0), Inches(5.15), Inches(11.33), Inches(0.35),
             font_size=11, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — TABLE OF CONTENTS / AGENDA
# ═══════════════════════════════════════════════════════════════════════════════
def slide_toc(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "Agenda Overview", "Table of Contents", 2, C_ACCENT_BLUE)

    sections = [
        ("01", "Introduction", "Context, Problem Statement, Motivation & Forensic Objectives", C_ACCENT_BLUE),
        ("02", "Literature Review", "Evolution of Deepfake Detection, Prior Art & Research Gaps", C_ACCENT_CYAN),
        ("03", "Advantages", "Core Strengths, Dual-Stream Engine & Explainability Innovations", C_ACCENT_EMR),
        ("04", "Requirements", "Software Stack, Toolchain & Hardware Specification Matrix", C_ACCENT_AMB),
        ("05", "Architecture", "Multi-Tier Forensic Engineering Pipeline & AI Inference Flow", C_ACCENT_PURP),
        ("06", "Data Flow & Use Cases", "DFD Level 0/1 Pipelines & Forensic Investigator Actor Workflows", C_ACCENT_ROSE),
        ("07", "Conclusion", "Key Project Achievements, Forensic Significance & Future Roadmap", C_ACCENT_CYAN),
    ]

    card_y = Inches(1.6)
    for num, title, desc, col in sections:
        add_rounded_rect(slide, Inches(0.8), card_y, Inches(11.73), Inches(0.68), C_BG_CARD, C_DIVIDER, Pt(1))
        # Left color indicator
        add_rect(slide, Inches(0.8), card_y, Inches(0.08), Inches(0.68), col)
        
        # Number pill
        add_rounded_rect(slide, Inches(1.1), card_y + Inches(0.14), Inches(0.6), Inches(0.4), C_BG_CARD_ALT, col, Pt(1))
        add_text(slide, num, Inches(1.1), card_y + Inches(0.18), Inches(0.6), Inches(0.35),
                 font_size=12, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Section Title
        add_text(slide, title, Inches(1.9), card_y + Inches(0.16), Inches(3.4), Inches(0.35),
                 font_size=15, bold=True, color=C_TEXT_MAIN)
        
        # Section Description
        add_text(slide, desc, Inches(5.4), card_y + Inches(0.18), Inches(7.0), Inches(0.35),
                 font_size=12, color=C_TEXT_MUTED)
        
        card_y += Inches(0.78)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — SECTION 1: INTRODUCTION (PROBLEM STATEMENT & CONTEXT)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_intro_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "1. Introduction", "Background & The Growing Deepfake Threat", 3, C_ACCENT_ROSE)

    # 4 Stat Cards Across Top
    stats = [
        ("900%+", "Growth in synthetic media online since 2019", C_ACCENT_ROSE),
        ("96%", "Non-consensual targets & defamatory content", C_ACCENT_AMB),
        ("$25B+", "Global financial fraud & executive impersonation", C_ACCENT_CYAN),
        ("< 3.5s", "Generation time with modern diffusion models", C_ACCENT_EMR),
    ]
    card_x = Inches(0.6)
    for val, desc, col in stats:
        add_rounded_rect(slide, card_x, Inches(1.55), Inches(2.9), Inches(1.4), C_BG_CARD, C_DIVIDER, Pt(1))
        add_line(slide, card_x, Inches(1.55), Inches(2.9), col, Pt(3))
        add_text(slide, val, card_x, Inches(1.68), Inches(2.9), Inches(0.55),
                 font_size=28, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_text(slide, desc, card_x + Inches(0.1), Inches(2.28), Inches(2.7), Inches(0.6),
                 font_size=11, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)
        card_x += Inches(3.1)

    # Two Structured Content Panels
    # Left Panel: What Are Deepfakes?
    add_rounded_rect(slide, Inches(0.6), Inches(3.15), Inches(5.95), Inches(3.9), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(0.6), Inches(3.15), Inches(5.95), C_ACCENT_CYAN, Pt(2))
    add_text(slide, "The Deepfake Phenomenon", Inches(0.8), Inches(3.3), Inches(5.5), Inches(0.35),
             font_size=16, bold=True, color=C_ACCENT_CYAN)
    
    deepfake_points = [
        "Synthetic media created using Generative Adversarial Networks (GANs), Autoencoders, and modern Diffusion Models.",
        "Enables hyper-realistic facial swaps, lip-sync manipulation, and full-head digital reenactment.",
        "Commoditization of AI tools enables novice threat actors to generate photorealistic fakes on consumer hardware.",
        "Social media platforms disseminate manipulated clips virally before manual fact-checking can respond.",
    ]
    add_bullets(slide, deepfake_points, Inches(0.8), Inches(3.75), Inches(5.5), Inches(3.1),
                font_size=12, color=C_TEXT_MUTED, space_before=6)

    # Right Panel: Core Forensic Challenges
    add_rounded_rect(slide, Inches(6.75), Inches(3.15), Inches(5.95), Inches(3.9), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(6.75), Inches(3.15), Inches(5.95), C_ACCENT_ROSE, Pt(2))
    add_text(slide, "Core Forensic & Investigative Challenges", Inches(6.95), Inches(3.3), Inches(5.5), Inches(0.35),
             font_size=16, bold=True, color=C_ACCENT_ROSE)

    challenge_points = [
        "Human Eye Fallibility: High-resolution generative face-swaps are imperceptible to human reviewers.",
        "Loss of Forensic Trust: Digital images & video submissions in courtrooms are challenged without auditable proof.",
        "Compression Degradation: Social platform re-encoding (H.264/JPEG compression) destroys classical metadata markers.",
        "The Black-Box Dilemma: Standard deep learning models output raw scores without explainable visual justifications.",
    ]
    add_bullets(slide, challenge_points, Inches(6.95), Inches(3.75), Inches(5.5), Inches(3.1),
                font_size=12, color=C_TEXT_MUTED, space_before=6)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — SECTION 1: INTRODUCTION (PROJECT SCOPE & OBJECTIVES)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_intro_objectives(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "1. Introduction", "Project Vision, Scope & Core Objectives", 4, C_ACCENT_CYAN)

    # Mission Statement Box
    add_rounded_rect(slide, Inches(0.6), Inches(1.55), Inches(12.1), Inches(1.15), C_BG_CARD, C_ACCENT_CYAN, Pt(1.5))
    add_text(slide, "PROJECT MISSION", Inches(0.8), Inches(1.65), Inches(11.7), Inches(0.25),
             font_size=11, bold=True, color=C_ACCENT_CYAN)
    add_text(slide,
             "To design and engineer an end-to-end, explainable digital forensics system that detects facial manipulation "
             "in both static imagery and continuous video streams, producing legally defensible forensic evidence dossiers "
             "and actionable risk metrics for digital investigators, media outlets, and security teams.",
             Inches(0.8), Inches(1.95), Inches(11.7), Inches(0.65),
             font_size=13, italic=True, color=C_TEXT_MAIN)

    # 4 Pillar Objective Cards
    pillars = [
        ("01", "Multi-Modal Detection",
         "Process both high-resolution static images (JPG, PNG, WebP) and temporal multi-frame video clips (MP4, AVI) through an automated face extraction and normalization pipeline.",
         C_ACCENT_BLUE),
        ("02", "Dual-Stream AI Inference",
         "Combine deep spatial features (EfficientNet-B3 compound scaling) with high-pass frequency residual analysis to detect subtle blending and re-sampling artifacts.",
         C_ACCENT_PURP),
        ("03", "Explainable AI (Grad-CAM)",
         "Demystify black-box neural classifications by projecting gradient-weighted class activation heatmaps directly over facial zones to highlight manipulation boundaries.",
         C_ACCENT_CYAN),
        ("04", "Forensic Dossier & API",
         "Generate automated, court-admissible ReportLab PDF reports with case IDs, cryptographic hashes, and temporal graphs, backed by an enterprise REST API.",
         C_ACCENT_EMR),
    ]

    card_w = Inches(2.9)
    card_x = Inches(0.6)
    card_y = Inches(2.9)
    for num, title, desc, col in pillars:
        add_rounded_rect(slide, card_x, card_y, card_w, Inches(4.15), C_BG_CARD, C_DIVIDER, Pt(1))
        add_line(slide, card_x, card_y, card_w, col, Pt(3))
        
        # Pill Number
        add_rounded_rect(slide, card_x + Inches(0.15), card_y + Inches(0.2), Inches(0.7), Inches(0.4), C_BG_CARD_ALT, col, Pt(1))
        add_text(slide, num, card_x + Inches(0.15), card_y + Inches(0.25), Inches(0.7), Inches(0.3),
                 font_size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Title
        add_text(slide, title, card_x + Inches(0.15), card_y + Inches(0.8), card_w - Inches(0.3), Inches(0.6),
                 font_size=15, bold=True, color=C_TEXT_MAIN)
        
        add_line(slide, card_x + Inches(0.15), card_y + Inches(1.45), card_w - Inches(0.3), C_DIVIDER, Pt(1))
        
        # Description
        add_text(slide, desc, card_x + Inches(0.15), card_y + Inches(1.6), card_w - Inches(0.3), Inches(2.3),
                 font_size=12, color=C_TEXT_MUTED)
        
        card_x += Inches(3.07)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — SECTION 2: LITERATURE REVIEW (EVOLUTION OF DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_lit_review_evolution(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "2. Literature Review", "Evolution of Deepfake Detection Methodologies", 5, C_ACCENT_CYAN)

    eras = [
        ("Generation 1: Biological & Heuristics (2018 - 2020)",
         "Handcrafted Physical & Biological Markers",
         [
             "Eye-blinking deficiency detection (Li et al., 2018) leveraging EAR (Eye Aspect Ratio).",
             "Head pose and 3D facial landmark inconsistency analysis (Yang et al., 2019).",
             "Remote Photoplethysmography (rPPG) measuring synthetic heartbeat signals (Hernandez-Ortega, 2020).",
         ],
         "Critical Limitations: Fragile to video compression; modern GANs generate realistic blinking and smooth facial geometry.",
         C_ACCENT_AMB),

        ("Generation 2: Early Deep Learning & Shallow CNNs (2020 - 2022)",
         "Domain-Specific Meso-Scale Feature Extractors",
         [
             "MesoNet (Meso-4 & MesoInception-4 by Afchar et al.): Focused on microscopic image noise.",
             "Standard VGG-16 & ResNet-50 transfer learning on FaceForensics++ benchmarks.",
             "Frequency domain DCT / FFT spectral analysis (Durall et al., 2020).",
         ],
         "Critical Limitations: Severe overfitting to specific dataset artifacts; poor cross-manipulation generalization; zero explainability.",
         C_ACCENT_BLUE),

        ("Generation 3: Compound Scaling & XAI (Present Work)",
         "State-of-the-Art Dual-Stream Forensics + Explainable AI",
         [
             "EfficientNet-B3 (Tan & Le): Uniform compound scaling of depth, width, and resolution.",
             "Dual-Stream Architecture: High-pass frequency residual stream + Spatial RGB stream.",
             "Explainable AI: Grad-CAM gradient backpropagation for courtroom-ready visual heatmaps.",
         ],
         "Proposed Solution: Lightweight parameter footprint (~12M), high generalization, and transparent evidence output.",
         C_ACCENT_EMR),
    ]

    card_y = Inches(1.55)
    for gen_title, subtitle, bullets, drawback, col in eras:
        add_rounded_rect(slide, Inches(0.6), card_y, Inches(12.1), Inches(1.68), C_BG_CARD, C_DIVIDER, Pt(1))
        add_rect(slide, Inches(0.6), card_y, Inches(0.08), Inches(1.68), col)
        
        # Header Row
        add_text(slide, gen_title.upper(), Inches(0.85), card_y + Inches(0.1), Inches(6.5), Inches(0.3),
                 font_size=12, bold=True, color=col)
        add_text(slide, subtitle, Inches(7.5), card_y + Inches(0.1), Inches(5.0), Inches(0.3),
                 font_size=11, italic=True, color=C_TEXT_MUTED, align=PP_ALIGN.RIGHT)
        
        # Bullets
        add_bullets(slide, bullets, Inches(0.85), card_y + Inches(0.38), Inches(11.6), Inches(0.85),
                    font_size=11, color=C_TEXT_MUTED, space_before=2)
        
        # Footer / Limitation Note
        add_rounded_rect(slide, Inches(0.85), card_y + Inches(1.26), Inches(11.65), Inches(0.32), C_BG_CARD_ALT, C_DIVIDER, Pt(0.5))
        add_text(slide, f"Forensic Assessment: {drawback}", Inches(0.95), card_y + Inches(1.3), Inches(11.4), Inches(0.25),
                 font_size=10, italic=True, color=col)
        
        card_y += Inches(1.85)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — SECTION 2: LITERATURE REVIEW (COMPARATIVE ANALYSIS & GAPS)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_lit_review_comparison(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "2. Literature Review", "Comparative Benchmark & Research Gaps", 6, C_ACCENT_CYAN)

    # Comparative Table
    table_shape = slide.shapes.add_table(6, 6, Inches(0.6), Inches(1.55), Inches(12.1), Inches(2.6))
    table = table_shape.table

    # Column Widths
    col_widths = [Inches(2.5), Inches(1.6), Inches(1.7), Inches(2.2), Inches(1.9), Inches(2.2)]
    for idx, w in enumerate(col_widths):
        table.columns[idx].width = w

    headers = ["Methodology / Model", "Avg Accuracy", "Compute Cost", "Explainability", "Video Support", "Forensic Readiness"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_HEADER_ROW
        cell.text_frame.margin_left = Inches(0.08)
        p = cell.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = h
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.color.rgb = C_ACCENT_CYAN
        run.font.name = "Segoe UI"

    rows = [
        ("Biological / Heuristics (EAR/PPG)", "64.2% - 71.5%", "Low (CPU)", "Moderate (Signals)", "Poor (Decay)", "Low (Manual)"),
        ("MesoNet (MesoInception-4)", "78.4% - 82.1%", "Very Low", "None (Black Box)", "Moderate", "Low (Score only)"),
        ("ResNet-50 / DenseNet-121", "84.5% - 88.0%", "High (Heavy)", "Low (Raw Saliency)", "Slow (Heavy)", "Partial"),
        ("Vision Transformer (ViT-Base)", "90.2% - 92.8%", "Extremely High", "Partial (Attention)", "Very Slow (OOM)", "Low (Research)"),
        ("Our System (EfficientNet Ensemble + XAI)", "92.00% (Overall)", "Optimized (12M params)", "Full (Grad-CAM Heatmaps)", "High (Sampled FPS)", "Full (Auto PDF Dossier)"),
    ]

    for i, row in enumerate(rows):
        is_ours = (i == len(rows) - 1)
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_BG_CARD_ALT if is_ours else C_BG_CARD
            cell.text_frame.margin_left = Inches(0.08)
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = val
            run.font.size = Pt(10)
            run.font.bold = is_ours
            run.font.color.rgb = C_ACCENT_EMR if (is_ours and j == 1) else (C_WHITE if is_ours else C_TEXT_MUTED)
            run.font.name = "Segoe UI"

    # Research Gaps Addressed Panel
    add_rounded_rect(slide, Inches(0.6), Inches(4.45), Inches(12.1), Inches(2.6), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(0.6), Inches(4.45), Inches(12.1), C_ACCENT_AMB, Pt(2))
    add_text(slide, "CRITICAL RESEARCH Gaps Addressed by This Project", Inches(0.85), Inches(4.6), Inches(11.5), Inches(0.3),
             font_size=13, bold=True, color=C_ACCENT_AMB)

    gaps = [
        "Gap 1: The 'Black-Box' Legal Dilemma — Conventional models provide no spatial visual reasoning. Our system integrates Grad-CAM class activation maps that identify exactly which facial regions (mouth boundary, eye blend, hairline) triggered the deepfake classification.",
        "Gap 2: Disconnect Between AI Research and Forensic Practice — Most literature outputs a terminal scalar probability. Our platform automatically synthesizes an ISO-standardized PDF investigation dossier with cryptographic metadata, timestamps, and confidence telemetry.",
        "Gap 3: Compute Inefficiency in Video Forensics — Existing transformer approaches cause GPU out-of-memory errors on lengthy videos. Our decimation frame sampling engine achieves comprehensive temporal anomaly detection at real-time speeds.",
    ]
    add_bullets(slide, gaps, Inches(0.85), Inches(4.95), Inches(11.6), Inches(1.95),
                font_size=11, color=C_TEXT_MUTED, space_before=5)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — SECTION 3: ADVANTAGES
# ═══════════════════════════════════════════════════════════════════════════════
def slide_advantages(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "3. Advantages", "Key Advantages & Technological Innovations", 7, C_ACCENT_EMR)

    advantages = [
        ("Compound Scaling Efficiency",
         "EfficientNet-B3",
         "Balances depth, width, and resolution scaling uniformly, delivering state-of-the-art accuracy with only ~12M parameters (vs 25M+ in ResNet-50 and 86M+ in ViT), reducing memory overhead.",
         C_ACCENT_BLUE),

        ("Explainable AI (Grad-CAM)",
         "Courtroom Verifiability",
         "Extracts gradients from the final convolutional layer to generate transparent visual heatmaps, pinpointing blending artifacts, unnatural skin boundaries, and synthetic textures.",
         C_ACCENT_CYAN),

        ("Dual-Stream Frequency Engine",
         "Spatial + Frequency Analysis",
         "Augments spatial RGB analysis with high-pass 5x5 convolution residuals, detecting compression disparities and re-sampling frequency signatures that escape standard vision models.",
         C_ACCENT_PURP),

        ("Unified Multi-Modal Pipeline",
         "Image & Video Forensics",
         "Uniformly digests high-res static images and sequential video clips. Employs adaptive frame decimation and statistical confidence averaging across temporal sequences.",
         C_ACCENT_EMR),

        ("Automated Evidence Dossiers",
         "Tamper-Evident PDF Reports",
         "Built-in ReportLab engine compiles case metadata, examiner identity, input snapshots, Grad-CAM heatmaps, and confidence breakdowns into ready-to-file legal PDF documents.",
         C_ACCENT_ROSE),

        ("Enterprise-Ready REST API",
         "High-Throughput Integration",
         "Features production Flask REST endpoints (`/api/predict`) with JSON payloads, enabling seamless integration into corporate SOCs, law enforcement backends, and newsroom pipelines.",
         C_ACCENT_AMB),
    ]

    card_w = Inches(3.85)
    card_h = Inches(2.6)
    
    # 2 rows x 3 columns
    coords = [
        (Inches(0.6), Inches(1.55)),
        (Inches(4.7), Inches(1.55)),
        (Inches(8.8), Inches(1.55)),
        (Inches(0.6), Inches(4.4)),
        (Inches(4.7), Inches(4.4)),
        (Inches(8.8), Inches(4.4)),
    ]

    for (x, y), (title, tag, desc, col) in zip(coords, advantages):
        add_rounded_rect(slide, x, y, card_w, card_h, C_BG_CARD, C_DIVIDER, Pt(1))
        add_line(slide, x, y, card_w, col, Pt(3))
        
        # Tag
        add_rounded_rect(slide, x + Inches(0.15), y + Inches(0.18), Inches(2.2), Inches(0.28), C_BG_CARD_ALT, col, Pt(0.5))
        add_text(slide, tag.upper(), x + Inches(0.15), y + Inches(0.2), Inches(2.2), Inches(0.24),
                 font_size=9, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Title
        add_text(slide, title, x + Inches(0.15), y + Inches(0.55), card_w - Inches(0.3), Inches(0.4),
                 font_size=14, bold=True, color=C_TEXT_MAIN)
        
        add_line(slide, x + Inches(0.15), y + Inches(1.0), card_w - Inches(0.3), C_DIVIDER, Pt(0.5))
        
        # Description
        add_text(slide, desc, x + Inches(0.15), y + Inches(1.1), card_w - Inches(0.3), Inches(1.35),
                 font_size=11, color=C_TEXT_MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — SECTION 4: SOFTWARE AND HARDWARE REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════
def slide_requirements(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "4. Requirements", "Software Environment & Hardware Specification Matrix", 8, C_ACCENT_AMB)

    # Left Container: Software Requirements
    add_rounded_rect(slide, Inches(0.6), Inches(1.55), Inches(5.95), Inches(5.5), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(0.6), Inches(1.55), Inches(5.95), C_ACCENT_CYAN, Pt(3))
    
    add_text(slide, "SOFTWARE REQUIREMENTS & TOOLCHAIN", Inches(0.85), Inches(1.75), Inches(5.4), Inches(0.35),
             font_size=14, bold=True, color=C_ACCENT_CYAN)

    sw_items = [
        ("Operating System", "Windows 10 / 11 (64-bit), Ubuntu 20.04+ LTS, macOS Monterey+"),
        ("Runtime Environment", "Python 3.9, 3.10, or 3.11 (CPython 64-bit)"),
        ("Deep Learning Frameworks", "TensorFlow 2.15+ (Keras 3.x), PyTorch 2.1+, Torchvision"),
        ("Computer Vision & Math", "OpenCV (cv2) 4.8+, Pillow (PIL), NumPy 1.24+, Scikit-learn"),
        ("Web Framework & Backend", "Flask 3.0+, Werkzeug, Gunicorn (WSGI HTTP server)"),
        ("Reporting Engine", "ReportLab 4.0+ (Dynamic vector PDF compiler)"),
        ("Data Persistence", "SQLite3 (Relational case log) + RFC 4180 CSV Export"),
        ("Frontend & Visuals", "HTML5, Vanilla CSS3 (Custom Dark Theme), Chart.js 4.4"),
        ("Containerization & Cloud", "Docker, Docker Compose, Render.yaml cloud manifest"),
    ]

    sw_y = Inches(2.2)
    for category, spec in sw_items:
        add_text(slide, f"{category}:", Inches(0.85), sw_y, Inches(2.2), Inches(0.3),
                 font_size=11, bold=True, color=C_WHITE)
        add_text(slide, spec, Inches(3.05), sw_y, Inches(3.3), Inches(0.35),
                 font_size=10, color=C_TEXT_MUTED)
        sw_y += Inches(0.51)

    # Right Container: Hardware Requirements (Min vs Recommended)
    add_rounded_rect(slide, Inches(6.75), Inches(1.55), Inches(5.95), Inches(5.5), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(6.75), Inches(1.55), Inches(5.95), C_ACCENT_AMB, Pt(3))
    
    add_text(slide, "HARDWARE SPECIFICATION MATRIX", Inches(7.0), Inches(1.75), Inches(5.4), Inches(0.35),
             font_size=14, bold=True, color=C_ACCENT_AMB)

    # Min Hardware Box
    add_rounded_rect(slide, Inches(7.0), Inches(2.2), Inches(5.45), Inches(2.2), C_BG_CARD_ALT, C_DIVIDER, Pt(1))
    add_text(slide, "MINIMUM HARDWARE (Dev / CPU Inference)", Inches(7.15), Inches(2.3), Inches(5.1), Inches(0.3),
             font_size=11, bold=True, color=C_ACCENT_AMB)
    min_specs = [
        "CPU: Quad-Core Intel Core i5 (8th Gen+) or AMD Ryzen 5 (2.5 GHz+)",
        "RAM: 8 GB DDR4 (512MB RAM constraint optimization built-in)",
        "Storage: 10 GB available SSD space for models & cache",
        "GPU: Integrated Intel UHD / Iris Xe (Supports CPU inference mode)",
    ]
    add_bullets(slide, min_specs, Inches(7.15), Inches(2.65), Inches(5.1), Inches(1.6),
                font_size=10, color=C_TEXT_MUTED, space_before=3)

    # Recommended Hardware Box
    add_rounded_rect(slide, Inches(7.0), Inches(4.6), Inches(5.45), Inches(2.25), C_BG_CARD_ALT, C_ACCENT_EMR, Pt(1))
    add_text(slide, "RECOMMENDED (Forensic Lab / Batch Video)", Inches(7.15), Inches(4.7), Inches(5.1), Inches(0.3),
             font_size=11, bold=True, color=C_ACCENT_EMR)
    rec_specs = [
        "CPU: 8-Core Intel Core i7 / i9 or AMD Ryzen 7 (3.5 GHz+)",
        "RAM: 16 GB to 32 GB DDR4 / DDR5 High-Speed Memory",
        "Storage: 50 GB NVMe M.2 SSD (High I/O for 4K video decimation)",
        "GPU: NVIDIA GeForce RTX 3060 / 4070 (6GB+ VRAM, CUDA 11.8/12.x & cuDNN)",
    ]
    add_bullets(slide, rec_specs, Inches(7.15), Inches(5.05), Inches(5.1), Inches(1.65),
                font_size=10, color=C_TEXT_MUTED, space_before=3)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — SECTION 5: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
def slide_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "5. Architecture", "Multi-Tier Forensic Engineering Architecture", 9, C_ACCENT_PURP)

    # 4 Tier Architecture Diagram (Horizontal Stack)
    tiers = [
        ("TIER 1: PRESENTATION & CLIENT LAYER",
         [
             ("Web Dashboard", "Interactive HTML5/CSS3 interface with Chart.js analytics"),
             ("REST API Clients", "Programmatic headless JSON consumers (cURL, Python, SOC)"),
             ("PDF Viewer", "Forensic Dossier rendering & cryptographic stamp viewer"),
         ],
         C_ACCENT_BLUE),

        ("TIER 2: APPLICATION CONTROLLER & SECURITY LAYER",
         [
             ("Flask Core Router", "Handles request routing, session lifecycle, error handling"),
             ("MIME & File Guard", "Validates file magic bytes, MIME types, image integrity"),
             ("Audit Logger", "Automated case ID assignment, timestamping, user tracking"),
         ],
         C_ACCENT_CYAN),

        ("TIER 3: FORENSIC AI & COMPUTER VISION ENGINE",
         [
             ("OpenCV Preprocessor", "Haar / DNN facial landmark detection, auto-crop with 20% margin"),
             ("EfficientNet-B3 Core", "224x224 / 300x300 compound deep feature inference (12M params)"),
             ("Grad-CAM XAI Module", "Backprop on final conv layer -> JET colormap overlay"),
             ("Video Decimation Unit", "Frame sampler (10 FPS step), temporal anomaly aggregator"),
         ],
         C_ACCENT_PURP),

        ("TIER 4: PERSISTENCE & EVIDENCE GENERATION LAYER",
         [
             ("SQLite Registry", "Local relation store (`deepfake_history.db`) for all case audits"),
             ("ReportLab PDF Engine", "Compiles dynamic forensic evidence dossiers with charts"),
             ("CSV Export Service", "One-click bulk export in standard RFC 4180 format"),
         ],
         C_ACCENT_EMR),
    ]

    tier_y = Inches(1.55)
    tier_h = Inches(1.22)
    for title, components, col in tiers:
        add_rounded_rect(slide, Inches(0.6), tier_y, Inches(12.1), tier_h, C_BG_CARD, C_DIVIDER, Pt(1))
        # Left accent block
        add_rect(slide, Inches(0.6), tier_y, Inches(0.08), tier_h, col)
        
        # Tier Title Pill
        add_rounded_rect(slide, Inches(0.85), tier_y + Inches(0.12), Inches(3.2), Inches(0.28), C_BG_CARD_ALT, col, Pt(0.5))
        add_text(slide, title, Inches(0.85), tier_y + Inches(0.14), Inches(3.2), Inches(0.25),
                 font_size=9, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Component boxes across the row
        comp_x = Inches(4.2)
        comp_w = Inches(7.7 / len(components))
        for comp_name, comp_desc in components:
            add_rounded_rect(slide, comp_x, tier_y + Inches(0.12), comp_w - Inches(0.1), tier_h - Inches(0.24), C_BG_CARD_ALT, C_DIVIDER, Pt(0.5))
            add_text(slide, comp_name, comp_x + Inches(0.08), tier_y + Inches(0.2), comp_w - Inches(0.26), Inches(0.3),
                     font_size=11, bold=True, color=C_WHITE)
            add_text(slide, comp_desc, comp_x + Inches(0.08), tier_y + Inches(0.52), comp_w - Inches(0.26), Inches(0.5),
                     font_size=9, color=C_TEXT_MUTED)
            comp_x += comp_w
            
        tier_y += Inches(1.36)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — SECTION 6: DATA FLOW DIAGRAM (DFD)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_dfd(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "6. Data Flow & Use Cases", "Data Flow Diagram (DFD Level 0 & Level 1)", 10, C_ACCENT_ROSE)

    # Sub-header: Level 0 Context Diagram
    add_text(slide, "LEVEL 0: CONTEXT DATA FLOW DIAGRAM", Inches(0.6), Inches(1.5), Inches(12.1), Inches(0.3),
             font_size=11, bold=True, color=C_ACCENT_CYAN)

    # Level 0 Entities
    # Entity 1: User / Forensic Examiner
    add_rounded_rect(slide, Inches(0.6), Inches(1.85), Inches(2.2), Inches(1.0), C_BG_CARD, C_ACCENT_BLUE, Pt(1))
    add_text(slide, "INVESTIGATOR /\nEXAMINER", Inches(0.6), Inches(2.05), Inches(2.2), Inches(0.6),
             font_size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Arrow 1 -> 2
    add_arrow(slide, Inches(3.0), Inches(2.18), Inches(1.0), Inches(0.22), C_ACCENT_CYAN)
    add_text(slide, "Image / Video", Inches(2.9), Inches(1.9), Inches(1.2), Inches(0.25),
             font_size=9, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

    # Process 0.0: Deepfake Investigation System
    add_oval(slide, Inches(4.2), Inches(1.75), Inches(4.5), Inches(1.2), C_BG_CARD_ALT, C_ACCENT_CYAN, Pt(2))
    add_text(slide, "0.0 DEEPFAKE INVESTIGATION\nSYSTEM (CORE FORENSIC ENGINE)", Inches(4.2), Inches(2.05), Inches(4.5), Inches(0.6),
             font_size=12, bold=True, color=C_ACCENT_CYAN, align=PP_ALIGN.CENTER)

    # Arrow 2 -> 3
    add_arrow(slide, Inches(8.9), Inches(2.18), Inches(1.0), Inches(0.22), C_ACCENT_EMR)
    add_text(slide, "Forensic Dossier", Inches(8.8), Inches(1.9), Inches(1.2), Inches(0.25),
             font_size=9, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

    # Entity 2: Legal Registry / Audit Archive
    add_rounded_rect(slide, Inches(10.1), Inches(1.85), Inches(2.6), Inches(1.0), C_BG_CARD, C_ACCENT_EMR, Pt(1))
    add_text(slide, "FORENSIC DOSSIER &\nCOURTROOM ARCHIVE", Inches(10.1), Inches(2.05), Inches(2.6), Inches(0.6),
             font_size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_line(slide, Inches(0.6), Inches(3.15), Inches(12.1), C_DIVIDER, Pt(1))

    # Sub-header: Level 1 Detailed Forensic Pipeline
    add_text(slide, "LEVEL 1: STEP-BY-STEP FORENSIC PROCESSING PIPELINE", Inches(0.6), Inches(3.3), Inches(12.1), Inches(0.3),
             font_size=11, bold=True, color=C_ACCENT_ROSE)

    steps = [
        ("1.0 Ingestion", "Upload validation, MIME-type guard, format normalization", C_ACCENT_BLUE),
        ("2.0 Face Extraction", "OpenCV Haar/DNN detection, 20% margin crop & resize", C_ACCENT_CYAN),
        ("3.0 Neural Inference", "EfficientNet-B3 compound deep spatial feature extraction", C_ACCENT_PURP),
        ("4.0 Explainable XAI", "Grad-CAM gradient backprop -> JET heatmap overlay", C_ACCENT_ROSE),
        ("5.0 Risk Aggregation", "Confidence calibration, thresholding, temporal frame stats", C_ACCENT_AMB),
        ("6.0 Dossier Assembly", "SQLite case logging, ReportLab PDF synthesis, CSV export", C_ACCENT_EMR),
    ]

    step_w = Inches(1.85)
    step_x = Inches(0.6)
    step_y = Inches(3.7)

    for i, (p_title, p_desc, col) in enumerate(steps):
        add_rounded_rect(slide, step_x, step_y, step_w, Inches(3.15), C_BG_CARD, C_DIVIDER, Pt(1))
        add_line(slide, step_x, step_y, step_w, col, Pt(3))
        
        # Process Circle
        add_oval(slide, step_x + Inches(0.62), step_y + Inches(0.2), Inches(0.6), Inches(0.6), C_BG_CARD_ALT, col, Pt(1))
        add_text(slide, f"{i+1}", step_x + Inches(0.62), step_y + Inches(0.3), Inches(0.6), Inches(0.4),
                 font_size=14, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Title
        add_text(slide, p_title, step_x + Inches(0.08), step_y + Inches(0.95), step_w - Inches(0.16), Inches(0.4),
                 font_size=12, bold=True, color=C_TEXT_MAIN, align=PP_ALIGN.CENTER)
        
        add_line(slide, step_x + Inches(0.15), step_y + Inches(1.4), step_w - Inches(0.3), C_DIVIDER, Pt(0.5))
        
        # Description
        add_text(slide, p_desc, step_x + Inches(0.08), step_y + Inches(1.5), step_w - Inches(0.16), Inches(1.4),
                 font_size=10, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)
        
        step_x += Inches(2.05)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — SECTION 6: USE CASE DIAGRAM & ACTOR WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════
def slide_use_case(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "6. Data Flow & Use Cases", "Use Case Diagram & System Interaction Matrix", 11, C_ACCENT_ROSE)

    # Left Container: Primary System Actors
    add_rounded_rect(slide, Inches(0.6), Inches(1.55), Inches(3.6), Inches(5.5), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(0.6), Inches(1.55), Inches(3.6), C_ACCENT_CYAN, Pt(3))
    add_text(slide, "PRIMARY SYSTEM ACTORS", Inches(0.8), Inches(1.75), Inches(3.2), Inches(0.3),
             font_size=13, bold=True, color=C_ACCENT_CYAN)

    actors = [
        ("Forensic Investigator", "Uploads suspected media files, inspects temporal frame sequences, downloads legal PDF dossiers for court exhibits.", C_ACCENT_BLUE),
        ("Cybersecurity Analyst", "Triages media feeds, inspects risk distributions on dashboard, exports historical audit registries via CSV.", C_ACCENT_PURP),
        ("Enterprise REST Client", "Automated headless pipelines calling `/api/predict` via JSON for real-time content moderation.", C_ACCENT_EMR),
        ("System Administrator", "Monitors model weights, container health, SQLite database storage, and runtime diagnostic logs.", C_ACCENT_AMB),
    ]

    act_y = Inches(2.2)
    for a_name, a_role, col in actors:
        add_rounded_rect(slide, Inches(0.8), act_y, Inches(3.2), Inches(1.15), C_BG_CARD_ALT, col, Pt(0.5))
        add_text(slide, a_name, Inches(0.9), act_y + Inches(0.1), Inches(3.0), Inches(0.28),
                 font_size=11, bold=True, color=col)
        add_text(slide, a_role, Inches(0.9), act_y + Inches(0.38), Inches(3.0), Inches(0.7),
                 font_size=9, color=C_TEXT_MUTED)
        act_y += Inches(1.25)

    # Right Container: Core System Use Cases
    add_rounded_rect(slide, Inches(4.4), Inches(1.55), Inches(8.3), Inches(5.5), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(4.4), Inches(1.55), Inches(8.3), C_ACCENT_ROSE, Pt(3))
    add_text(slide, "FORENSIC USE CASES & SYSTEM CAPABILITIES", Inches(4.65), Inches(1.75), Inches(7.8), Inches(0.3),
             font_size=13, bold=True, color=C_ACCENT_ROSE)

    use_cases = [
        ("UC-01", "Upload & Authenticate Static Facial Image",
         "User submits high-res image. System executes facial isolation, EfficientNet-B3 inference, and renders authenticity score.",
         C_ACCENT_BLUE),
        ("UC-02", "Decimated Video Stream Temporal Forensics",
         "User uploads video file. System samples frames at 10-step intervals, computes frame-level scores, and plots temporal anomalies.",
         C_ACCENT_CYAN),
        ("UC-03", "Inspect Grad-CAM Visual Attention Heatmap",
         "Investigator examines activation overlays on eyes, nose, mouth, and boundary blend seams to verify synthetic artifacts.",
         C_ACCENT_PURP),
        ("UC-04", "Generate Courtroom-Ready Forensic PDF Dossier",
         "System automatically synthesizes comprehensive PDF document with Case ID, SHA hash, timestamps, and examiner signature block.",
         C_ACCENT_ROSE),
        ("UC-05", "Automated Headless Inference via REST API",
         "External client posts multipart image to `/api/predict`, receiving structured JSON with classification, confidence, and risk tier.",
         C_ACCENT_EMR),
        ("UC-06", "Historical Case Search & RFC 4180 CSV Export",
         "Auditor queries past case logs, filters by verdict/risk, and downloads complete registry for offline compliance archiving.",
         C_ACCENT_AMB),
    ]

    uc_x_left = Inches(4.65)
    uc_x_right = Inches(8.65)
    uc_w = Inches(3.85)

    coords_uc = [
        (uc_x_left, Inches(2.2)),
        (uc_x_right, Inches(2.2)),
        (uc_x_left, Inches(3.8)),
        (uc_x_right, Inches(3.8)),
        (uc_x_left, Inches(5.35)),
        (uc_x_right, Inches(5.35)),
    ]

    for (x, y), (uc_id, uc_title, uc_desc, col) in zip(coords_uc, use_cases):
        add_rounded_rect(slide, x, y, uc_w, Inches(1.4), C_BG_CARD_ALT, C_DIVIDER, Pt(0.5))
        add_rounded_rect(slide, x + Inches(0.12), y + Inches(0.12), Inches(0.8), Inches(0.26), C_BG_CARD, col, Pt(0.5))
        add_text(slide, uc_id, x + Inches(0.12), y + Inches(0.14), Inches(0.8), Inches(0.22),
                 font_size=9, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        add_text(slide, uc_title, x + Inches(1.0), y + Inches(0.12), uc_w - Inches(1.1), Inches(0.3),
                 font_size=11, bold=True, color=C_WHITE)
        
        add_text(slide, uc_desc, x + Inches(0.12), y + Inches(0.48), uc_w - Inches(0.24), Inches(0.85),
                 font_size=9, color=C_TEXT_MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — SECTION 7: CONCLUSION (SUMMARY OF ACHIEVEMENTS)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_conclusion_summary(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "7. Conclusion", "Summary of Achievements & Forensic Value", 12, C_ACCENT_CYAN)

    achievements = [
        ("Production-Grade AI Detection",
         "92.00% Overall Accuracy",
         "Successfully integrated multi-scale EfficientNet ensemble with high-pass frequency residual filtering, achieving 92.00% overall benchmark accuracy across diverse synthetic manipulation techniques.",
         C_ACCENT_BLUE),

        ("Overcoming the 'Black-Box' Barrier",
         "Explainable AI (Grad-CAM)",
         "Bridged the critical gap between academic neural networks and legal forensic scrutiny by outputting transparent pixel activation heatmaps that prove where forgery occurred.",
         C_ACCENT_CYAN),

        ("Dual Modality Forensic Coverage",
         "Image & Video Pipeline",
         "Engineered an automated face detection and decimation pipeline capable of processing both single snapshots and extended temporal video streams without memory exhaustion.",
         C_ACCENT_PURP),

        ("End-to-End Operational Readiness",
         "Automated Dossiers & REST API",
         "Delivered an enterprise-ready solution featuring 1-click legal PDF dossier compilation, historical SQLite audit registries, and headless REST endpoints for automated triage.",
         C_ACCENT_EMR),
    ]

    card_w = Inches(5.9)
    card_h = Inches(1.9)
    coords = [
        (Inches(0.6), Inches(1.55)),
        (Inches(6.8), Inches(1.55)),
        (Inches(0.6), Inches(3.6)),
        (Inches(6.8), Inches(3.6)),
    ]

    for (x, y), (title, tag, desc, col) in zip(coords, achievements):
        add_rounded_rect(slide, x, y, card_w, card_h, C_BG_CARD, C_DIVIDER, Pt(1))
        add_rect(slide, x, y, Inches(0.08), card_h, col)
        
        # Tag
        add_rounded_rect(slide, x + Inches(0.2), y + Inches(0.18), Inches(2.4), Inches(0.26), C_BG_CARD_ALT, col, Pt(0.5))
        add_text(slide, tag.upper(), x + Inches(0.2), y + Inches(0.2), Inches(2.4), Inches(0.22),
                 font_size=9, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Title
        add_text(slide, title, x + Inches(0.2), y + Inches(0.55), card_w - Inches(0.4), Inches(0.35),
                 font_size=14, bold=True, color=C_TEXT_MAIN)
        
        # Description
        add_text(slide, desc, x + Inches(0.2), y + Inches(0.95), card_w - Inches(0.4), Inches(0.85),
                 font_size=11, color=C_TEXT_MUTED)

    # Bottom Impact Banner
    add_rounded_rect(slide, Inches(0.6), Inches(5.65), Inches(12.1), Inches(1.4), C_BG_CARD_ALT, C_ACCENT_CYAN, Pt(1.5))
    add_text(slide, "SOCIETAL & LEGAL IMPACT", Inches(0.85), Inches(5.8), Inches(11.6), Inches(0.25),
             font_size=11, bold=True, color=C_ACCENT_CYAN)
    add_text(slide,
             "The Deepfake Investigation System establishes a reproducible, explainable, and tamper-evident framework "
             "for safeguarding media integrity. By converting complex neural network tensors into understandable visual heatmaps "
             "and certified forensic reports, it restores evidentiary trust for legal authorities, journalists, and enterprise cybersecurity.",
             Inches(0.85), Inches(6.1), Inches(11.6), Inches(0.85),
             font_size=12, italic=True, color=C_TEXT_MAIN)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — SECTION 7: CONCLUSION (FUTURE ROADMAP)
# ═══════════════════════════════════════════════════════════════════════════════
def slide_conclusion_future(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)
    add_header(slide, "7. Conclusion", "Future Enhancements & Research Roadmap", 13, C_ACCENT_CYAN)

    roadmap = [
        ("Phase 1: Audio-Visual Multimodal Fusion",
         "Lip-Sync & Vocal Analysis",
         [
             "Cross-reference visual lip movements against acoustic phonemes to detect dubbing.",
             "Incorporate Wav2Vec / Mel-spectrogram frequency anomaly classifiers.",
             "Calculate audio-visual synchronization confidence coefficients.",
         ],
         C_ACCENT_CYAN),

        ("Phase 2: Spatio-Temporal 3D-CNNs & Transformers",
         "Micro-Expression Continuity",
         [
             "Deploy 3D-ResNet / Video Vision Transformers (TimeSformer) for motion dynamics.",
             "Track micro-expression flicker and temporal blending inconsistencies across frames.",
             "Reduce per-frame decimation reliance with optical flow continuity modeling.",
         ],
         C_ACCENT_BLUE),

        ("Phase 3: Blockchain Evidence Chain-of-Custody",
         "Cryptographic Tamper-Proofing",
         [
             "Anchor SHA-256 evidence hashes onto immutable distributed ledgers.",
             "Cryptographically sign forensic PDF dossiers with verifiable examiner private keys.",
             "Ensure unbroken chain-of-custody compliant with federal digital forensic rules.",
         ],
         C_ACCENT_PURP),

        ("Phase 4: Diffusion & Latent Reconstruction Defense",
         "Next-Gen Generative Countermeasures",
         [
             "Engineer specific detectors for Stable Diffusion, Midjourney v6, and Sora artifacts.",
             "Incorporate latent reverse-engineering to reconstruct original unmodified faces.",
             "Implement self-updating continual learning loops against zero-day generative models.",
         ],
         C_ACCENT_EMR),
    ]

    card_w = Inches(2.9)
    card_x = Inches(0.6)
    card_y = Inches(1.55)

    for phase_title, subtitle, bullets, col in roadmap:
        add_rounded_rect(slide, card_x, card_y, card_w, Inches(5.5), C_BG_CARD, C_DIVIDER, Pt(1))
        add_line(slide, card_x, card_y, card_w, col, Pt(3))
        
        # Subtitle Pill
        add_rounded_rect(slide, card_x + Inches(0.15), card_y + Inches(0.2), card_w - Inches(0.3), Inches(0.32), C_BG_CARD_ALT, col, Pt(0.5))
        add_text(slide, subtitle.upper(), card_x + Inches(0.15), card_y + Inches(0.24), card_w - Inches(0.3), Inches(0.25),
                 font_size=9, bold=True, color=col, align=PP_ALIGN.CENTER)
        
        # Title
        add_text(slide, phase_title, card_x + Inches(0.15), card_y + Inches(0.68), card_w - Inches(0.3), Inches(0.7),
                 font_size=13, bold=True, color=C_TEXT_MAIN)
        
        add_line(slide, card_x + Inches(0.15), card_y + Inches(1.45), card_w - Inches(0.3), C_DIVIDER, Pt(0.5))
        
        # Bullets
        add_bullets(slide, bullets, card_x + Inches(0.15), card_y + Inches(1.6), card_w - Inches(0.3), Inches(3.6),
                    font_size=11, color=C_TEXT_MUTED, space_before=6)
        
        card_x += Inches(3.07)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — WRAP-UP / THANK YOU & Q&A
# ═══════════════════════════════════════════════════════════════════════════════
def slide_thank_you(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill_bg(slide, C_BG_DARK)

    # Accent borders
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.08), C_ACCENT_CYAN)
    add_rect(slide, 0, 0, Inches(0.08), SLIDE_H, C_ACCENT_BLUE)

    # Center Card
    add_rounded_rect(slide, Inches(1.5), Inches(1.2), Inches(10.33), Inches(5.1), C_BG_CARD, C_DIVIDER, Pt(1))
    add_line(slide, Inches(1.5), Inches(1.2), Inches(10.33), C_ACCENT_CYAN, Pt(3))

    # Badge Pill
    add_rounded_rect(slide, Inches(4.5), Inches(1.7), Inches(4.33), Inches(0.42), C_BG_CARD_ALT, C_ACCENT_CYAN, Pt(1))
    add_text(slide, "FINAL EVALUATION & DEMONSTRATION", Inches(4.5), Inches(1.78), Inches(4.33), Inches(0.35),
             font_size=11, bold=True, color=C_ACCENT_CYAN, align=PP_ALIGN.CENTER)

    # Main Headline
    add_text(slide, "THANK YOU!", Inches(1.8), Inches(2.4), Inches(9.73), Inches(0.8),
             font_size=42, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_text(slide, "Questions & Technical Discussion", Inches(1.8), Inches(3.25), Inches(9.73), Inches(0.5),
             font_size=20, italic=True, color=C_ACCENT_CYAN, align=PP_ALIGN.CENTER)

    # Key Takeaways Row
    add_line(slide, Inches(3.0), Inches(3.9), Inches(7.33), C_DIVIDER, Pt(1))

    summary_pts = [
        "EfficientNet-B3 Deep Feature Extraction",
        "Grad-CAM Explainable AI Heatmaps",
        "Temporal Frame-by-Frame Video Forensics",
        "Automated Court-Admissible PDF Reports",
        "Production Flask REST API & Web Dashboard",
    ]
    add_text(slide, " • ".join(summary_pts), Inches(2.0), Inches(4.2), Inches(9.33), Inches(0.6),
             font_size=12, bold=True, color=C_TEXT_MAIN, align=PP_ALIGN.CENTER)

    add_text(slide, "Deepfake Investigation System  |  Digital Media Forensics & AI Integrity",
             Inches(1.8), Inches(5.2), Inches(9.73), Inches(0.4),
             font_size=12, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    print("=" * 60)
    print("Deepfake Investigation System — Generating Presentation Deck")
    print("=" * 60)

    slides = [
        ("Slide 1  : Title Slide", slide_title),
        ("Slide 2  : Agenda Overview / Table of Contents", slide_toc),
        ("Slide 3  : 1. Introduction — Problem & Threat Landscape", slide_intro_problem),
        ("Slide 4  : 1. Introduction — Scope & Objectives", slide_intro_objectives),
        ("Slide 5  : 2. Literature Review — Evolution of Detection", slide_lit_review_evolution),
        ("Slide 6  : 2. Literature Review — Comparative Benchmarks & Gaps", slide_lit_review_comparison),
        ("Slide 7  : 3. Advantages — Technological & Forensic Innovations", slide_advantages),
        ("Slide 8  : 4. Requirements — Software & Hardware Matrix", slide_requirements),
        ("Slide 9  : 5. Architecture — Multi-Tier Forensic Pipeline", slide_architecture),
        ("Slide 10 : 6. Data Flow — Level 0 & Level 1 Pipelines", slide_dfd),
        ("Slide 11 : 6. Use Cases — Actor Workflows & System Matrix", slide_use_case),
        ("Slide 12 : 7. Conclusion — Achievements & Forensic Value", slide_conclusion_summary),
        ("Slide 13 : 7. Conclusion — Future Research Roadmap", slide_conclusion_future),
        ("Slide 14 : Wrap-up — Thank You & Discussion", slide_thank_you),
    ]

    for title, func in slides:
        func(prs)
        print(f"  [OK] Generated: {title}")

    prs.save(OUTPUT)
    print("=" * 60)
    print(f"[SUCCESS] Presentation generated and saved successfully to:\n{OUTPUT}")
    print(f"Total Slides: {len(slides)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

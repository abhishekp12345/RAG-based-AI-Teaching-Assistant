"""
Phase 3B Execution:
1. Cleans all 3 transcripts using transcript_cleaner.py.
2. Saves corrected transcripts separately: *_hi_en_corrected.json.
3. Updates lecture_manifest.json.
4. Regenerates all chunks using run_phase3_chunking.py.
5. Verifies script distributions (before and after).
6. Gathers before-and-after segment samples for the report.
"""

import json
from pathlib import Path
from transcript_cleaner import clean_transcript_file, analyze_scripts

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def main():
    print("=" * 70)
    print("LEARNMATE AI - PHASE 3B: HINDI-ENGLISH QUALITY VALIDATION")
    print("=" * 70)

    manifest_path = DATA_DIR / "lecture_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cleaning_reports = []
    before_after_samples = []

    for lec in manifest["lectures"]:
        lec_id = lec["lecture_id"]
        in_path = BASE_DIR / lec["transcript_path"]
        
        # Determine output corrected path
        stem = in_path.stem.replace("_cleaned", "").replace("_hi_en_corrected", "")
        out_path = in_path.parent / f"{stem}_hi_en_corrected.json"

        print(f"\n--- Cleaning {lec_id}: {lec['lecture_title']} ---")
        print(f"  Input : {in_path.name}")
        print(f"  Output: {out_path.name}")

        # Read before state for sampling
        with open(in_path, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
        orig_text = orig_data.get("full_transcript", "")
        orig_scripts, _ = analyze_scripts(orig_text)

        # Clean
        report = clean_transcript_file(in_path, out_path)
        cleaning_reports.append((lec_id, report))
        print(f"  Corrected {report['corrected_segments']} / {report['total_segments']} segments ({report['corrected_pct']}%)")

        # Read after state
        with open(out_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
        new_text = new_data.get("full_transcript", "")
        new_scripts, _ = analyze_scripts(new_text)

        print(f"  Script distribution shift:")
        print(f"    Urdu/Perso-Arabic : {orig_scripts['perso_arabic_urdu']} -> {new_scripts['perso_arabic_urdu']} chars")
        print(f"    Devanagari (Hindi): {orig_scripts['devanagari']} -> {new_scripts['devanagari']} chars")
        print(f"    Korean            : {orig_scripts['korean']} -> {new_scripts['korean']} chars")
        print(f"    Cyrillic          : {orig_scripts['cyrillic']} -> {new_scripts['cyrillic']} chars")
        print(f"    English/Latin     : {orig_scripts['latin']} -> {new_scripts['latin']} chars")

        # Find 3 representative before/after segment examples
        samples_found = 0
        for orig_s, new_s in zip(orig_data["segments"], new_data["segments"]):
            if orig_s["text"] != new_s["text"] and samples_found < 3:
                before_after_samples.append({
                    "lecture_id": lec_id,
                    "segment_id": orig_s["id"],
                    "timestamps": f"[{orig_s['start']}s -> {orig_s['end']}s]",
                    "before": orig_s["text"],
                    "after": new_s["text"]
                })
                samples_found += 1

        # Update manifest pointer
        lec["transcript_path"] = f"data/transcripts/{out_path.name}"

    # Save updated manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(BASE_DIR / "lecture_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\nUpdated lecture_manifest.json with corrected transcript paths.")

    # Save before/after samples for reporting
    samples_path = DATA_DIR / "transcripts" / "phase3b_corrections_sample.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(before_after_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved before/after samples to {samples_path.name}")

    print("\n" + "=" * 70)
    print("PHASE 3B TRANSCRIPT CLEANING COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()


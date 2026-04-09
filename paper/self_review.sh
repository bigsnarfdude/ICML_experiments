#!/bin/bash
# Automated self-review using Gemini and Claude CLI
# Usage: ./self_review.sh

PAPER="$(cat /Users/vincent/ICML/paper/main.tex)"

PROMPT='You are an expert ICML reviewer (Area Chair level). Review the following paper for ICML 2026. Be rigorous and critical.

Provide:
1. Summary (3-4 sentences)
2. Strengths (numbered, with specific section/table references)
3. Weaknesses (labeled Major/Minor, with specific section/table references)
4. Questions for Authors (numbered)
5. Missing References
6. Scores table:
   - Soundness: 1-4
   - Presentation: 1-4
   - Contribution: 1-4
   - Overall Score: 1-10
   - Confidence: 1-5
7. Justification (2-3 sentences explaining the overall score)

Be specific. Reference sections, tables, and figures by number. Flag statistical issues, missing controls, overclaimed results, and scope mismatches between evidence and conclusions.

Here is the paper:

'"$PAPER"

echo "=========================================="
echo "GEMINI REVIEW"
echo "=========================================="
gemini -m gemini-2.5-pro -p "$PROMPT" 2>&1 | tee /Users/vincent/ICML/paper/review_gemini_$(date +%Y%m%d_%H%M%S).md

echo ""
echo "=========================================="
echo "CLAUDE REVIEW"
echo "=========================================="
echo "$PROMPT" | claude -p 2>&1 | tee /Users/vincent/ICML/paper/review_claude_$(date +%Y%m%d_%H%M%S).md

echo ""
echo "Done. Reviews saved to paper/review_*.md"

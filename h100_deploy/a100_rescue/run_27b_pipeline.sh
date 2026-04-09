#!/bin/bash
echo "Starting 27B IT-only BVP..."
python3 -u ~/behavioral_27b_it_only.py > ~/behavioral_27b_it_only.log 2>&1
echo "BVP done, starting 27B IT-only theorem..."
python3 -u ~/theorem_27b_it_only.py > ~/theorem_27b_it_only.log 2>&1
echo "All 27B done."

#!/bin/bash
set -e

# requirements.txt 파일이 없으면 종료
[ -f requirements.txt ] || exit 1

while IFS= read -r requirement || [ -n "$requirement" ]; do
    [[ "$requirement" =~ ^# ]] && continue
    [[ -z "$requirement" ]] && continue

    # pip freeze의 결과에 해당 requirement가 있으면 건너뛰고, 없으면 설치
    if pip freeze | grep -Fxq "$requirement"; then
        continue
    else
        pip install "$requirement" --quiet
    fi
done < requirements.txt

# Todo CLI

Python으로 만든 간단한 할일 목록 CLI 도구.

## 사용법

```bash
# 할일 추가
python todo.py add "장보기"

# 할일 목록 보기
python todo.py list

# 할일 완료 표시 (ID로 지정)
python todo.py done 1
```

## 저장

데이터는 현재 디렉토리의 `todos.json` 파일에 자동 저장됩니다.

## 요구사항

- Python 3.6+
- 외부 패키지 없음 (표준 라이브러리만 사용)

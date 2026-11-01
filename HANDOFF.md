# HANDOFF

## 목표
contilab: 비전·LLM 지속 학습 실증 프레임워크(벤치마크, 증류·병합, 30+ 지표). CPU 완주, 설정으로 GPU 확장. 공개 리포·위키·Pages.

## 완료
- v0.1: 전략 8종, TA/TIES/증류, vmap 앙상블, 러너·리포트·CLI
- v0.2.0 릴리스: 지표 수정, 견고성, CI, 검증 기반 sweep 선택, 연속 증류, 작은 학생 모델
- 이후 main: M10–M12(bench 정밀도, 반복 계산 제거, vmap 폴백·OOM 분할 기록), 릴리스 워크플로(PyPI job은 변수로 꺼둠)
- CPU 연구: `cpu_main` 324런(6 벤치마크, MNIST 6시드), `cpu_tuned` 228런, `cpu_student` 12런 → `docs/wiki/Results.md`, CSV는 `docs/assets/results/data/`
- 테스트 188 통과, 5 skip(네트워크)

## 남은 작업
- 사용자 설정: PyPI pending publisher + 변수 `PUBLISH_PYPI=true`, Zenodo 연동(`RELEASING.md`)
- 미검증: `configs/profiles/gpu.yaml`(GPU 없음), text5 전체 규모(CPU ~20시간+)
- main의 옛 작성자 커밋: 이력 재작성은 force push 필요(사용자)

## 실패한 시도와 원인
- HF Xet CDN 403 → 원본 미러(GCS MNIST, fast.ai S3)
- 위키 push·force push·릴리스 API가 프록시에서 차단 → 위키는 `wiki-sync.yml`, 릴리스는 `release.yml`
- CPU 과다구독 → text5 6배 저하, text 그룹 단독 실행
- EWC λ=100: Permuted MNIST·CIFAR-10/100 전 시드 발산(λ 10–30 안정)
- 증류 v0.1 기본값은 학습 부족 → ties·routed·10 epoch
- 컨테이너 재시작 시 분리 실행도 종료 → 재실행하면 끝난 런은 건너뜀

## 검증
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest tests -q  # --run-network: 다운로드 포함
.venv/bin/contilab run configs/experiments/smoke_all.yaml && .venv/bin/contilab report runs/smoke_all
.venv/bin/contilab run configs/experiments/cpu_main.yaml   # 4코어 ~22시간, 재개 가능
```

## 주의
- 커밋 작성자 `JonPark0 <yeochoony@gmail.com>`; 도구 서명 트레일러 금지
- main 직접 push 금지 → 브랜치+PR
- `runs/`(gitignore)는 보존 안 됨; 결과는 Results.md와 CSV
- `docs/wiki/` 변경이 main에 들어가면 위키 자동 동기화
- run id는 전체 설정 해시: 기본값을 바꾸면 기존 런과 id가 달라짐

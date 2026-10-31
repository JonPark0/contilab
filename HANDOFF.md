# HANDOFF

## 목표
contilab: 비전·LLM 지속 학습 실증 프레임워크(벤치마크, 증류·병합, 30+ 지표). CPU 완주, 설정으로 GPU 확장. 공개 리포·위키·Pages.

## 완료
- Task 1–27(`docs/design/` 스펙·계획): 전략 8종, TA/TIES/증류, vmap 앙상블, 러너·리포트·CLI
- v0.2(`release/v0.2`): 지표 수정(M4/M6/M8/M13), 견고성(M1/M2/M5/M14), CI, 검증 기반 sweep 선택, `method_sweep`, 연속 증류, 작은 학생 모델
- 테스트 178 통과, 5 skip(네트워크); CI는 Python 3.11·3.13
- CPU 연구: `cpu_main` 180런, `cpu_tuned` 228런, `cpu_student` 12런 → `docs/wiki/Results.md`
- README, CITATION.cff, MIT, 위키 동기화, Pages

## 남은 작업
- M10 `runner/bench.py`: bf16 처리량 측정을 기본 프로필 정밀도와 맞추기
- M11 성능 개선: 배치마다 class_table 계산, LwF forward 3회, distill의 TrainPool 재생성, probe 행렬 3중 계산
- M12 `runner/execute.py`: `vmappable=False` 순차 폴백 기록, 메모리 기반 E 분할
- 미검증: `configs/profiles/gpu.yaml`(GPU 없음), Split CIFAR-100
- 배포: v0.2.0 릴리스(Actions → Release → Run workflow, tag `v0.2.0`), Zenodo DOI

## 실패한 시도와 원인
- HF Xet CDN 403 → 원본 미러(GCS MNIST, fast.ai S3)
- 위키 git push·force push·리포 설정 API가 프록시에서 차단 → 위키는 `.github/workflows/wiki-sync.yml`, 설정은 수동
- 전문가 학습이 `max_E` 무시 → LM OOM, `plan_expert_groups`로 수정
- CPU 과다구독 → text5 6배 저하, text 그룹 단독 실행
- EWC λ=100: Permuted MNIST·CIFAR 전 시드 발산(λ 10–30 안정)
- 증류 v0.1 기본값(base·entropy·1 epoch)은 학습 부족 → 튜닝으로 ties·routed·10 epoch

## 검증
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest tests -q  # --run-network: 다운로드 포함
.venv/bin/contilab run configs/experiments/smoke_all.yaml && .venv/bin/contilab report runs/smoke_all
.venv/bin/contilab run configs/experiments/cpu_main.yaml   # 4코어 ~16시간, 재개 가능
```

## 주의
- 커밋 작성자 `JonPark0 <yeochoony@gmail.com>`; Co-Authored-By 등 도구 서명 트레일러 금지
- main 직접 push 금지 → 브랜치+PR
- `runs/`(3.2 GB, gitignore)는 보존 안 됨; 결과 표는 Results.md에만 있음
- `docs/wiki/` 변경이 main에 들어가면 위키 자동 동기화
- 데이터 캐시 `$CONTILAB_CACHE`(기본 `~/.cache/contilab`)

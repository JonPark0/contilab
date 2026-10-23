# HANDOFF

## 목표
contilab: 비전·LLM 지속 학습 실증 프레임워크(벤치마크, 증류·병합, 30+ 지표). CPU 완주, 설정으로 GPU 확장. 공개 리포·위키·Pages.

## 완료
- 구현 Task 1–27 (`docs/design/`의 스펙·계획): 전략 8종 + TA/TIES/다중 교사 증류, vmap 시드 앙상블, 러너·재개·리포트·CLI
- 테스트 146 통과, 5 skip(네트워크)
- CPU 연구 165런(159 done, EWC 6 발산), EWC λ 스윕 → `docs/wiki/Results.md`
- README, CITATION.cff, MIT, 위키 동기화 워크플로, Pages

## 남은 작업
- M1 `engine/trainer.py` `DivergenceGuard` 기준값을 `runner/execute.py` `_save_ckpt`/`_load_ckpt`에 저장
- M2 `engine/step.py` `maybe_compile`: 첫 호출 이후 재컴파일 실패도 eager 폴백
- M4 `metrics/representation.py` `cka_final`: j=T−1 자기항 제외
- M5 `_save_ckpt`, `consolidate/experts.py` `train_experts` 캐시, `data/cache.py` `cached_tensors`: tmp+`os.replace` 원자 쓰기
- M6 `metrics/collectors.py` buffer_bytes를 실제 샘플 바이트로
- M8 `metrics/recovery.py` `recovery`: 런의 옵티마이저 사용
- M12 `runner/execute.py` `VmapFallback` 기록, 메모리 기반 E 분할
- M13 `metrics/continual.py` `aaa` 등 Joint가 태스크 0만 추적
- M14 `train_experts`, `consolidate/distill.py` `DistillConsolidator`에 발산 감지
- 스윕 자동 선택: `runner/finalize.py` `finalize_experiment`
- 미검증: `configs/profiles/gpu.yaml`(GPU 없음), Split CIFAR-100

## 실패한 시도와 원인
- HF Xet CDN 403 → 원본 미러(GCS MNIST, fast.ai S3)
- 위키 git push·force push·리포 설정 API가 프록시에서 차단 → 위키는 `.github/workflows/wiki-sync.yml`, 설정은 수동
- 전문가 학습이 `max_E` 무시 → LM OOM, `plan_expert_groups`로 수정
- CPU 과다구독 → text5 6배 저하, text 그룹 단독 실행
- EWC λ=100: Permuted MNIST·CIFAR 전 시드 발산(λ 10–30 안정)
- 증류 기본값 학습 부족(LA 0.10–0.46)

## 검증
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest tests -q                # --run-network: 다운로드 테스트
.venv/bin/contilab run configs/experiments/smoke_all.yaml && .venv/bin/contilab report runs/smoke_all
.venv/bin/contilab run configs/experiments/cpu_main.yaml   # 4코어 약 16시간, 재개 가능
```

## 주의
- 커밋 작성자 `Yeochoon, Park <yeochoony@gmail.com>`; Co-Authored-By 등 도구 서명 트레일러 금지
- main 직접 push 금지 → 브랜치+PR
- `runs/`(3.2 GB, gitignore)는 보존 안 됨; 결과 표는 Results.md에만 있음
- `docs/wiki/` 변경이 main에 들어가면 위키 자동 동기화
- 데이터 캐시 `$CONTILAB_CACHE`(기본 `~/.cache/contilab`)

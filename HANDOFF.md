# HANDOFF

## 목표
contilab: 비전·LLM 지속 학습 실증 프레임워크(벤치마크, 증류·병합, 30+ 지표). CPU 완주, 설정으로 GPU 확장. 공개 리포·위키·Pages.

## 완료
- v0.1: 전략 8종, TA/TIES/증류, vmap 앙상블, 러너·리포트·CLI
- v0.2.0 릴리스: 지표 수정, 견고성, CI, 검증 기반 sweep 선택, 연속 증류, 작은 학생 모델
- v0.3.0: M10–M12(bench 정밀도, 반복 계산 제거, vmap 폴백·OOM 분할 기록), compile 재컴파일 수정(배치 축 dynamic, 그래프 한도 = 태스크 수 + 8), 릴리스 워크플로(PyPI job은 변수로 꺼둠), Split CIFAR-100
- CPU 연구: `cpu_main` 324런(6 벤치마크, MNIST 6시드), `cpu_tuned` 456런, `cpu_student` 12런, `cpu_text5_full` 36런(text5 설계 규모 2,000/500, 학습 19시간) → `docs/wiki/Results.md`, CSV는 `docs/assets/results/data/`
- EWC 비전 기본 λ=30(튜닝 선택값), CIFAR 프리셋은 compile 없이 bf16 → CIFAR 72런·MNIST EWC 18런 재실행
- EWC 강성 상한: SGD에서 Fisher를 max_stiffness/(lr·λ)로 clamp → 발산 0건(cpu_main EWC 24런·cpu_tuned EWC 72런 재실행)
- 문서 사이트(Just the Docs): 한국어 홈, FAQ, 구조화 데이터, llms.txt, IndexNow; 차트·변경 이력 페이지는 `scripts/sync_docs.py`가 생성
- 테스트 199 통과, 5 skip(네트워크)

## 남은 작업
- Zenodo: concept DOI `10.5281/zenodo.23906803`(전체 버전), v0.3.1 `10.5281/zenodo.23906804`; 이후 릴리스 DOI는 release 워크플로가 노트에 자동 추가
- 사용자 설정: PyPI pending publisher + 변수 `PUBLISH_PYPI=true`, Search Console 인증, 소셜 미리보기 업로드(`RELEASING.md`)
- 미검증: `configs/profiles/gpu.yaml`(GPU 없음)
- main의 옛 작성자 커밋: 이력 재작성은 force push 필요(사용자)

## 실패한 시도와 원인
- HF Xet CDN 403 → 원본 미러(GCS MNIST, fast.ai S3)
- 위키 push·force push·릴리스 API가 프록시에서 차단 → 위키는 `wiki-sync.yml`, 릴리스는 `release.yml`
- CPU 과다구독 → text5 6배 저하, text 그룹 단독 실행
- EWC λ=100: Permuted MNIST·CIFAR-10/100 전 시드 발산, λ=30도 CIFAR 후반 발산 → 원인은 SGD에서 lr·λ·F > 2(누적 Fisher가 계속 커짐), Fisher clamp로 해결
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

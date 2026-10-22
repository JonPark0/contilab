# contilab 설계 스펙

- 작성일: 2026-10-22
- 상태: 리뷰 대기
- 위치: `/home/user/contilab`

## 1. 목적과 범위

### 1.1 목적
Continual Learning(CL) 접근법을 **실증**하기 위한 신규 프레임워크. 하나의 코드베이스에서 다음을 수행한다.
- **벤치마크**: 비전·언어 CL 시나리오에서 메서드를 동일 조건으로 비교
- **학습**: 대표 CL 메서드(정규화·리플레이·파라미터 격리)로 순차 학습
- **증류**: (1) 망각 방지 기법으로서의 증류, (2) 태스크별 전문가를 단일 모델로 통합하는 사후 통합 파이프라인

### 1.2 설계 원칙
1. **비전·LLM 공통 추상화**: 동일한 Stream/Strategy/Evaluator 인터페이스
2. **성능 우선**: 학습 핫루프의 Python 오버헤드 제거, 다중 실행 병렬화(vmap), compile/bf16 자동 활용
3. **풍부한 평가**: 학계에서 사용되는 정량·정성 지표를 폭넓게 제공, 원자료 보존으로 사후 재계산 가능
4. **공정 비교**: 모든 메서드에 동일 루프·예산·평가 규약
5. **CPU→GPU 확장**: 설정 프로파일 전환만으로 규모 변경

### 1.3 범위에 포함 (1차)
- 메서드: Naive, Joint, EWC(Online), LwF, ER, DER++, A-GEM, LoRA-per-task
- 사후 통합: Multi-teacher 증류, Task Arithmetic, TIES (전체 파라미터 및 LoRA ΔW)
- 시나리오: Task-IL, Class-IL, Domain-IL
- 지표: 6장 카탈로그 전체 (티어별 수집)
- 실행 백엔드: CPU / CUDA / MPS (자동 선택)

### 1.4 범위에서 제외 (후속)
- Teacher→Student 지속 증류 (인터페이스 확장 지점만 1차에 포함, 5.6절)
- SI, iCaRL, PackNet, L2P/CODA-Prompt, O-LoRA/InfLoRA, 생성 리플레이
- 분산(멀티 GPU) 학습
- Rust/C++ 네이티브 확장 (프로파일링으로 병목이 입증된 경우에만 후속 검토)
- TRACE 벤치마크 및 GPU 프로파일의 zero-shot 일반 능력 세트(후속)

### 1.5 성공 기준
1. `smoke` 프로파일: 전 메서드·전 통합 기법이 60초 내 완주하고 전체 테스트 통과
2. `cpu` 프로파일: 이 클라우드 세션(4 CPU, 15GB RAM)에서 메서드×벤치마크×시드 매트릭스 완주, 리포트 생성
3. Split-MNIST Class-IL에서 알려진 정성 순서 재현: Naive < {EWC, LwF} < {ER, DER++} < Joint (ACC 기준, 시드 평균)
4. `contilab bench`로 vmap(E>1)·compile의 처리량 가속을 수치로 보고
5. 7장의 등가성·정합성 테스트 전부 통과

## 2. 실행 환경과 의존성

- Python 3.13, PyTorch(PyPI 최신 안정판), transformers, safetensors, datasets, numpy, scipy, scikit-learn(t-SNE/선형 probe 보조), matplotlib, pyyaml, psutil
- peft는 사용하지 않음 (자체 LoRA, 4.3절)
- 패키지 관리: `pyproject.toml`, `pip install -e .[dev]` (dev: pytest)
- 현재 세션 확인 사항: PyPI·HuggingFace 접근 가능, download.pytorch.org 차단 → PyPI 휠 사용, gcc/g++ 존재(inductor CPU 컴파일 가능)

## 3. 아키텍처

### 3.1 패키지 구조

```
contilab/
  core/          # 타입: Task, Stream, Batch, Scenario, RunContext
  data/          # 벤치마크 빌더, 디바이스 텐서 캐시, 인덱스 배처
  models/        # MLP, ReducedResNet18(GroupNorm), HF LM 분류기, LoRA
  engine/        # 앙상블(E축) 상태, 학습 스텝, 컴파일, 평가기
  strategies/    # CL 메서드 플러그인
  consolidate/   # 전문가 학습·캐시, TA, TIES, Multi-teacher 증류
  metrics/       # 지표 레지스트리, 수집기(probe), 지표 함수, 통계
  report/        # 표·그림·마크다운/HTML 리포트
  runner/        # 설정 로드, 매트릭스 전개, 스케줄러, 저장소, CLI
configs/         # 프로파일·실험 YAML
tests/
docs/
```

### 3.2 데이터 흐름

```
Config ─▶ Runner ─▶ Benchmark ─▶ Stream[Task]
                     │
                     ├─ 순차 CL:  Engine(Model, Strategy) ─ 태스크마다 ─▶ Collectors ─▶ RawLog
                     └─ 사후 통합: ExpertTrainer ─▶ Consolidator ─ 태스크마다 ─▶ Collectors ─▶ RawLog
RawLog ─▶ Metrics(순수 함수) ─▶ Results ─▶ Report
```

### 3.3 핵심 타입 (core)

- `Task`: `id`, `train: TensorSplit`, `test: TensorSplit`, `val: TensorSplit`, `classes: LongTensor`(전역 클래스 ID)
- `TensorSplit`: 디바이스 상주 텐서 `x`(이미지 또는 token ids), `y`, 텍스트일 경우 `attn_mask`
- `Stream`: `tasks: list[Task]`, `scenario: Literal["task", "class", "domain"]`, `num_classes`
- `Batch`: `x, y, task_id, mask` — 모든 필드는 선행 E축을 가짐 (`[E, B, ...]`)
- `RunContext`: 현재 태스크 인덱스, 스텝 수, 디바이스, E, RNG, 로거 핸들

## 4. 데이터·모델·성능 코어

### 4.1 벤치마크

| 이름 | 태스크 구성 | 원본 | `cpu` 규모 | `gpu` 규모 |
|---|---|---|---|---|
| `synthetic` | 가우시안 클러스터, 5태스크×2클래스 | 생성 | (smoke 전용) | — |
| `split_mnist` | 5태스크×2클래스 | `ylecun/mnist` | 전체 | 전체 |
| `permuted_mnist` | 10태스크, 고정 픽셀 순열, 10클래스 공유 | `ylecun/mnist` | 태스크당 학습 10k / 테스트 2k | 전체 |
| `split_cifar10` | 5태스크×2클래스 | `uoft-cs/cifar10` | 전체 | 전체 |
| `split_cifar100` | 10태스크×10클래스 | `uoft-cs/cifar100` | — | 전체 |
| `text5` | AG News(4) → Yelp(5) → Amazon(5) → DBpedia(14) → Yahoo(10) | `fancyzhx/ag_news`, `Yelp/yelp_review_full`, `SetFit/amazon_reviews_multi_en`, `fancyzhx/dbpedia_14`, `community-datasets/yahoo_answers_topics` | 태스크당 학습 2k / 테스트 500, 최대 길이 128 | 태스크당 학습 115k(원 설정) |

- `text5`의 Yelp·Amazon은 같은 5단계 평점 라벨을 공유(de Masson d'Autume 등 2019 설정), 전체 33 클래스
- 시나리오 지원: split 계열은 Task-IL/Class-IL, permuted는 Domain-IL, text5는 Class-IL(기본)과 Task-IL
- 검증 세트: 각 태스크 학습 데이터의 10%를 고정 시드로 분리 (TA/TIES λ 선택, 하이퍼파라미터 sweep용)
- 태스크 순서: 기본 고정, `task_order_seed` 지정 시 셔플

### 4.2 데이터 파이프라인 (속도)
- 최초 실행 시 다운로드 → 전처리(정규화·토크나이즈·패딩) → `~/.cache/contilab/<benchmark>/<hash>.safetensors`로 저장. 해시는 전처리 설정 + 토크나이저 이름 기준
- 실행 시 전체 분할을 디바이스 텐서로 적재. `torch.utils.data.DataLoader` 미사용
- 배칭: 에폭마다 멤버별 순열 `perm: [E, N]`을 생성하고 `x[perm[:, i:i+B]]`로 gather → `[E, B, ...]`
- 데이터 증강은 1차 범위에서 사용하지 않음 (teacher 로짓 캐시 전제와 일관)
- 텍스트: 태스크 내 길이 버킷 정렬은 사용하지 않고 고정 길이 패딩 (E축 텐서 형태 고정 → compile 재사용)

### 4.3 모델
- `MLP(in, hidden=400, depth=2, out)` — MNIST 계열 표준
- `ReducedResNet18(nf=20, norm=GroupNorm)` — CIFAR 계열. BatchNorm은 vmap과 running stats 문제로 사용하지 않음
- `HFClassifier(model_name, pooling="last_token")` — 텍스트. 기본 `EleutherAI/pythia-70m`(cpu), 대안 `HuggingFaceTB/SmolLM2-135M`, gpu 기본 `Qwen/Qwen2.5-1.5B`
- **분류 헤드**: 모든 모델은 전체 클래스 수의 단일 선형 헤드. 시나리오별 로짓 마스크(4.5절)
- **자체 LoRA**: `LoRALinear(base, r, alpha)`가 대상 `nn.Linear`를 감쌈. 어댑터 집합을 `dict[task_id, (A, B)]`로 보관, 활성 어댑터 선택, `delta_weight(task_id) = (alpha/r)·B@A` 제공. 대상 모듈은 설정(`lora.targets`). 기본: 텍스트는 어텐션 q/v 프로젝션, 비전은 헤드를 제외한 모든 `nn.Linear`/`nn.Conv2d`. `LoRAConv2d`는 A: [r, in·k·k], B: [out, r]로 ΔW = B@A를 커널 형태로 reshape해 `W + ΔW`로 합성곱
- 특징 추출 훅: 모든 모델은 `features(x) -> [B, D]`(헤드 직전 표현)를 제공 — probe·CKA·rank 지표용

### 4.4 앙상블 축 E (성능 코어)
- **정의**: E는 동시에 학습되는 독립 실행 수. 순차 CL에서는 시드, 전문가 학습에서는 (태스크 × 시드)
- **파라미터**: `torch.func.stack_module_state`로 `[E, ...]` 스택된 leaf 텐서 dict. 모델 계산은 `vmap(functional_call(base_model, ·))`
- **손실**: `loss_total = vmap(member_loss)(params, state, batch).sum()` → 멤버 간 그래디언트가 섞이지 않음
- **옵티마이저**: 원소별 연산 옵티마이저(SGD, AdamW)를 스택 텐서에 그대로 사용 → 멤버 독립성 유지. 그래디언트 노름 클리핑은 멤버별(`[E, -1]` 기준)로 직접 구현
- **멤버별 상태**: 데이터 순열, 리플레이 버퍼, Fisher, 구 파라미터 모두 `[E, ...]`
- **E=1 fast path**: vmap을 생략하고 일반 모듈 호출 (LLM 기본)
- **compile**: `engine.compile ∈ {auto, on, off}`. `auto`는 첫 스텝에서 컴파일을 시도하고 실패 시 eager 폴백 후 사유를 환경 정보에 기록
- **정밀도**: `engine.precision ∈ {auto, fp32, bf16}`. `auto`는 CUDA bf16 지원 또는 CPU AMX/AVX512-BF16 감지 시 bf16 autocast, 아니면 fp32
- **스레드**: `torch.set_num_threads`를 스케줄러가 프로세스별로 지정
- **처리량 계측**: 매 실행에서 samples/s, 스텝 시간, 피크 메모리를 기록

### 4.5 시나리오별 로짓 마스킹
- Task-IL: 학습·평가 모두 해당 태스크 클래스만 허용
- Class-IL: 학습 시 현재까지 본 클래스 허용(미래 클래스 마스크), 평가 시 지금까지 본 전체 클래스
- Domain-IL: 공유 클래스 전체
- 마스크는 `[num_classes]` 불리언 텐서를 로짓에 `masked_fill(-inf)`로 적용 (멀티헤드 대비 커널 호출 1회)

## 5. 학습 메서드와 통합 파이프라인

### 5.1 Strategy 인터페이스

```python
class Strategy:
    vmappable: bool = True
    # 호스트 측 (컴파일 대상 아님)
    def before_task(self, ctx: RunContext, task: Task) -> None: ...
    def replay_batch(self, ctx: RunContext) -> Batch | None: ...   # [E, B] 인덱스 기반
    def after_step(self, ctx: RunContext, batch: Batch, out: StepOutput) -> None: ...
    def after_task(self, ctx: RunContext, task: Task) -> None: ...
    # 핫패스 (멤버 단위로 작성, 엔진이 vmap·compile)
    def loss(self, params, state, batch, replay, logits_fn) -> Tensor: ...
    def transform_grad(self, grads, ref_grads) -> dict: return grads
    def state(self) -> dict[str, Tensor]: ...     # [E, ...] 스택 상태
```

### 5.2 엔진 학습 스텝
1. `replay = strategy.replay_batch(ctx)`
2. 핫패스 함수(컴파일 단위): 멤버별 `loss` 계산 → `torch.func.grad`로 그래디언트 → 필요 시 버퍼 배치의 참조 그래디언트 계산 → `transform_grad`
3. 옵티마이저 스텝
4. `strategy.after_step` (버퍼 갱신 등), NaN 검사, 수집기 호출(연속 평가 주기 도달 시)

### 5.3 메서드 구현 규약

| 메서드 | 구현 | 기본 하이퍼파라미터 |
|---|---|---|
| Naive | CE | — |
| Joint | 스트림 변환: 모든 태스크를 하나의 태스크로 합친 뒤 Naive. Intransigence 기준값 a*_j도 제공 | — |
| EWC (Online) | `after_task`에서 검증 제외 학습 샘플 N개에 대해 `vmap(grad)` per-sample 그래디언트로 대각 Fisher, F ← γF + F_new. 손실 `CE + λ/2·ΣF(θ−θ*)²` | λ=100(비전) / λ=10(텍스트), γ=1.0, N=1024 |
| LwF | `after_task`에서 파라미터 스냅샷 고정. 같은 스텝에서 no-grad forward, 이전 클래스 로짓에 KD | T=2, α=1 |
| ER | 멤버별 reservoir 버퍼(데이터셋 전역 인덱스만 저장). 현재 배치 + 같은 크기 리플레이 배치 결합 CE | M=500(cpu 비전)/200(text5) |
| DER++ | 버퍼에 인덱스 + 삽입 시점 로짓. `CE + α·MSE(z, z_buf) + β·CE(y_buf2)` | α=0.1, β=0.5, M 동일 |
| A-GEM | 스텝 내 참조 그래디언트(버퍼 배치). g·g_ref<0이면 `g − (g·g_ref/‖g_ref‖²)·g_ref` (멤버별) | M 동일, 참조 배치 256 |
| LoRA-per-task | 백본 고정, `before_task`에서 새 어댑터 추가·활성화. 분류 헤드는 모든 태스크에서 공유·학습(4.5절 마스킹으로 Task-IL에서는 현재 태스크 행만 갱신). Task-IL 평가는 태스크 ID로 어댑터 선택. Class-IL/Domain-IL 평가는 지금까지의 어댑터를 ΔW 단위 TA로 병합한 모델 사용(λ는 5.5절과 같은 방식으로 본 태스크 검증 세트에서 선택) | r=8, α=16 |

- 버퍼는 서로 다른 태스크의 데이터셋을 가리키므로 인덱스는 `(task_id, sample_idx)` 쌍으로 저장
- 공정성 규약: 동일 옵티마이저(기본 SGD lr=0.03 비전 / AdamW lr=1e-4 텍스트, LoRA는 1e-3), 동일 에폭(cpu: split_mnist 1, permuted_mnist 1, split_cifar10 3, text5 1), 동일 배치(비전 32, 텍스트 16), 동일 버퍼 예산
- 하이퍼파라미터는 `sweep` 블록으로 그리드 실행 가능, 선택은 검증 세트 기준

### 5.4 사후 통합: 전문가 학습
- 모든 전문가는 공통 초기값 θ₀(사전학습 가중치 또는 `init_seed` 고정 랜덤)에서 시작
- 전문가 i는 태스크 i만 학습 (전체 파라미터 또는 LoRA 어댑터)
- E축 = (태스크 × 시드)로 묶어 한 번의 vmap 실행으로 학습 (텍스트/LLM은 E=1 순차)
- 전문가 캐시: `~/.cache/contilab/experts/<config_hash>/`. 해시 = 벤치마크 + 모델 + 학습 설정 + 시드

### 5.5 통합 기법 (`Consolidator`)

```python
class Consolidator:
    def consolidate(self, base: Params, experts: list[Expert], ctx: RunContext) -> Params: ...
```
매 태스크 t 후 `experts[:t+1]`로 호출 → 정확도 행렬과 모든 지표가 순차 CL과 동일하게 정의됨.

| 기법 | 구현 |
|---|---|
| Task Arithmetic | τ_i = θ_i − θ₀, θ = θ₀ + λ·Σ τ_i. λ ∈ {0.1, 0.2, …, 1.0}에서 검증 ACC 최대값 선택 |
| TIES | τ_i의 상위 k% 크기만 유지(k=20 기본) → 원소별 부호 선출(합의 부호 = Σ τ_i의 부호) → 합의 부호와 일치하는 값만 평균 → θ₀ + λ·τ, λ는 TA와 같은 방식 선택 |
| Multi-teacher 증류 | 학생 초기값: θ₀ 또는 TA/TIES 병합 결과(`init: base|ta|ties`). 손실 KL(T=2), 라우팅 `routed`(샘플 소속 태스크 teacher) 또는 `entropy`(teacher별 예측 엔트로피의 softmax(−H) 가중 앙상블). 전이 데이터 3단계: `full`(전체 태스크 데이터), `buffer`(태스크당 M/t 샘플, 리플레이와 같은 총 예산 M), `proxy`(비전: 다른 데이터셋의 무라벨 이미지 — MNIST 계열은 CIFAR-10 그레이스케일 28×28 축소본, split_cifar10은 CIFAR-100, split_cifar100은 CIFAR-10; 텍스트: `Salesforce/wikitext` 문장) |
| LoRA 병합 | A·B 개별 병합 금지. ΔW_i = (α/r)·B_i A_i 단위로 TA/TIES 적용 후 백본에 가산 |

- **Class-IL 헤드**: 백본은 병합, 헤드는 각 전문가에서 자기 태스크 클래스 행만 가져와 결합. 로짓 스케일 불일치는 리포트에 한계로 명시하며, 증류가 이를 보정하는지를 비교 축으로 사용
- **teacher 로짓 캐시**: 증강이 없으므로 전이 데이터에 대한 teacher 로짓을 1회 계산해 저장, 증류 에폭 동안 teacher forward 없음
- **비용 산정**: 통합 기법의 학습 비용 = 전문가 학습 총합 + 통합 비용. 저장량 = 보관 전문가/어댑터 바이트 포함

### 5.6 확장 지점: Teacher→Student 지속 증류 (후속)
- `Consolidator.__init__(student_factory=None)`: teacher와 다른 아키텍처의 학생 생성 함수를 받을 수 있음
- `Consolidator.on_new_expert(expert, ctx)` 훅: 순차적으로 도착하는 teacher를 처리하는 구현을 위해 1차에서 기본 no-op으로 정의

## 6. 평가 지표 체계

### 6.1 원칙
- **Raw-log 우선**: 학습 중에는 원자료만 수집, 지표는 `RawLog → 값`의 순수 함수
- 지표 등록: `@metric(name, requires=[raw_keys], tier="core"|"extended"|"full")`
- 러너는 활성 티어의 지표들이 `requires`로 선언한 원자료만 수집
- `contilab metrics runs/<exp> --add <name>`: 원자료가 있으면 재실행 없이 계산, 없으면 필요한 원자료와 티어를 안내

### 6.2 원자료 (RawLog 키)

| 키 | 내용 | 수집 시점 | 티어 |
|---|---|---|---|
| `acc_matrix` | R[i, j]: 태스크 i 학습 후 태스크 j 테스트 정확도 | 태스크 종료 | core |
| `init_acc` | b_j: 학습 전(θ₀) 태스크 j 정확도 | 시작 시 | core |
| `cost` | 시간, samples/s, 피크 메모리, 학습 파라미터 수, 버퍼 바이트, 모델 바이트, FLOPs 추정 | 상시 | core |
| `pred_hist` | 태스크 종료 시 테스트 샘플별 예측 클래스·정답·confidence | 태스크 종료 | core |
| `stream_acc` | 매 ρ 스텝(기본 ρ=50)마다 본 태스크 전체의 서브셋(태스크당 200) 정확도 | 연속 | extended |
| `lca_curve` | 각 태스크 첫 β(기본 10) 배치에서의 현재 태스크 서브셋 정확도 | 태스크 시작부 | extended |
| `features` | 고정 probe 세트(태스크당 256 샘플)의 `features(x)` | 태스크 종료 | extended |
| `activations` | probe 세트의 은닉층별 활성화 통계(뉴런별 평균 |h|) | 태스크 종료 | extended |
| `param_snapshots` | 태스크 종료 시 파라미터(멤버 0만, safetensors) | 태스크 종료 | full |
| `task_grads` | 태스크 종료 시 각 태스크 검증 배치 손실의 그래디언트(평탄화, 멤버 0) | 태스크 종료 | full |
| `recovery` | 최종 모델 복사본을 태스크 j의 n=50 샘플로 K=20 스텝 재학습한 후 정확도 | 실행 종료 | full |
| `general_ppl` | (텍스트) `Salesforce/wikitext` 테스트 서브셋(cpu 200문장) perplexity, θ₀ 및 각 태스크 종료 | 태스크 종료 | extended |

### 6.3 정량 지표 정의
표기: T = 태스크 수, R[i, j] (0-index), b_j = init_acc.

**종합·안정성·가소성 (core)**
- ACC = (1/T) Σ_j R[T−1, j]
- AIA (Average Incremental Accuracy) = (1/T) Σ_i [(1/(i+1)) Σ_{j≤i} R[i, j]]
- BWT = (1/(T−1)) Σ_{j<T−1} (R[T−1, j] − R[j, j])
- FWT = (1/(T−1)) Σ_{j≥1} (R[j−1, j] − b_j)
- Forgetting: f_j = max_{l∈[j, T−2]} R[l, j] − R[T−1, j], F = mean_{j<T−1} f_j
- Relative Forgetting: f_j / max_{l∈[j, T−2]} R[l, j] 의 평균
- Learning Accuracy LA = (1/T) Σ_j R[j, j]
- Intransigence I = (1/T) Σ_j (a*_j − R[j, j]), a*_j = 같은 실험의 Joint 실행에서 태스크 j 정확도 (Joint 실행이 없으면 미계산으로 표시)

**연속 평가 (extended, De Lange 2023 / Caccia 2022)**
- A(k, n): 스텝 n에서 태스크 k 정확도 (`stream_acc`)
- min-ACC = 태스크 k 학습 종료 이후 모든 n에서 min A(k, n)의 평균 (k < T−1)
- WC-ACC = (1/T)·A(T−1, 끝) + (1 − 1/T)·min-ACC
- AAA (Average Anytime Accuracy) = 모든 평가 시점에서 (본 태스크 평균 정확도)의 평균
- Stability Gap = 태스크 k 종료 시 정확도 − 다음 태스크 학습 중 태스크 k의 최소 정확도, k에 대해 평균
- WF(Δ) (Windowed Forgetting) = 길이 Δ 스텝 윈도우 내 최대 정확도 낙폭, 태스크 평균 (기본 Δ = 태스크 길이의 10%)
- LCA_β = (1/(β+1)) Σ_{b=0..β} Z_b, Z_b = 태스크 평균 배치 b 시점 현재 태스크 정확도 (Chaudhry 2019)

**표현 수준 (extended)**
- Probe 정확도 P[i, j]: 태스크 i 종료 시 고정 특징으로 태스크 j 선형 분류기(로지스틱 회귀, probe 세트 80/20 분할) 정확도
- 표현 망각 = P[j, j] − P[T−1, j], 헤드 망각 = (R[j, j] − R[T−1, j]) − 표현 망각, 각 태스크 평균 (Davari 2022)
- CKA[j]: 태스크 j probe 특징의 (태스크 j 종료 시 vs 최종) linear CKA, 및 태스크 종료 시점 간 CKA 행렬 (Kornblith 2019)
- 특징 드리프트: 같은 샘플 특징의 정규화 L2 이동량 평균

**가소성 손실 진단 (extended; 반드시 최적화 기반 지표와 함께 보고)**
- Dormant 비율: 뉴런 점수 s_i = E|h_i| / mean_k E|h_k| ≤ τ 인 비율 (τ=0.025, Sokar 2023)
- Effective rank: 특징 행렬 특이값 정규화 분포의 엔트로피 지수 (Roy & Vetterli 2007)
- srank_δ: Σ_{i≤k} σ_i / Σ σ ≥ 1 − δ 를 만족하는 최소 k (δ=0.01, Kumar 2021)
- 가중치 노름: 레이어별 Frobenius 노름 추이
- 짝 지표: LA, LCA_β (신규 태스크 학습 능력) — 리포트에서 같은 표에 배치

**편향·신뢰도 (core)**
- Task-recency bias: Class-IL에서 이전 태스크 샘플 중 가장 최근 태스크 클래스로 예측된 비율 (Wu 2019, Masana 2022)
- 태스크별 예측 분포: 정답 태스크 × 예측 태스크 행렬
- ECE: 15 bin, 태스크별 및 전체

**복구성 (full)**
- Recovery_j = (재학습 후 정확도 − R[T−1, j]) / (R[j, j] − R[T−1, j]), 분모 ≤ 0.01이면 제외
- 높은 Recovery는 "피상적 망각"(지식 보존, 정렬 손상)의 증거로 해석 (Spurious Forgetting, ICLR 2025)

**손실 지형·간섭 (full)**
- Loss barrier_k = max_α L_k((1−α)θ_k + αθ_{k+1}) − [(1−α)L_k(θ_k) + αL_k(θ_{k+1})], α ∈ {0, 0.1, …, 1}, L_k는 태스크 k 검증 손실 (Mirzadeh 2020)
- 그래디언트 간섭 행렬: `task_grads` 간 코사인 유사도

**LLM 전용 (extended)**
- General retention Δ = general_ppl(최종) − general_ppl(θ₀) 및 태스크별 추이 (TRACE의 General Ability Delta를 cpu 규모에서 perplexity로 대체)

**효율 (core)**
- 총 학습 시간, samples/s, 피크 메모리(CPU: psutil RSS 최대값, CUDA: `max_memory_allocated`), 학습 파라미터 수, 버퍼 바이트, 모델 크기 증가(어댑터/전문가 포함), FLOPs(`torch.utils.flop_counter.FlopCounterMode`로 1스텝 측정 × 스텝 수)

### 6.4 통계
- 시드별 값 → 평균 ± 표준편차, 부트스트랩 95% CI(10,000회 재표본)
- 기준 메서드(설정, 기본 Naive) 대비 paired 검정: 시드 수 ≥ 6이면 Wilcoxon signed-rank, 미만이면 정확 permutation test
- 벤치마크 간: 지표별 평균 순위, Friedman 검정, Nemenyi critical difference (Demšar 2006)
- 시드 수가 적을 때(< 5) 리포트에 검정력 한계 경고 표시

### 6.5 정성 산출물
정확도 행렬 히트맵, 연속 평가 곡선(stability gap 구간 강조), 혼동 행렬 및 정답 태스크×예측 태스크 행렬, CKA 히트맵, 레이어별 가중치 드리프트 막대, 그래디언트 간섭 히트맵, probe 특징 PCA 및 t-SNE, 손실 보간 곡선, CD 다이어그램, 처리량 비교 막대(bench).

## 7. 러너·저장·리포트·에러 처리

### 7.1 CLI
| 명령 | 기능 |
|---|---|
| `contilab run <cfg.yaml> [key=val ...]` | 실험(매트릭스) 실행 |
| `contilab report <runs/exp>` | `report.md` + `report.html`(그림 PNG 임베드) 생성 |
| `contilab metrics <runs/exp> --add <name> [--tier ...]` | 지표 사후 계산 |
| `contilab bench [--model ...] [--E 1,2,4,8]` | eager / compile / vmap(E) 처리량 비교 |
| `contilab data prepare <benchmark>` | 데이터 사전 다운로드·캐시 |

### 7.2 설정
- YAML → dataclass 검증. 알 수 없는 키는 에러
- 프로파일: `configs/profiles/{smoke,cpu,gpu}.yaml`가 데이터 규모·모델·에폭·티어 기본값을 정의, 실험 YAML이 `profile:`로 참조 후 덮어씀
- 매트릭스: `matrix: {benchmark: [...], method: [...], seed: [...]}` 데카르트 곱. `sweep:`은 메서드 하이퍼파라미터 그리드
- `method`에 `consolidate:ta`, `consolidate:ties`, `consolidate:distill` 지정 가능 (전이 데이터 단계는 `consolidate.transfer`)

### 7.3 저장 구조
```
runs/<exp>/
  experiment.yaml        # 전개 전 원본
  <run_id>/
    config.yaml          # 해석 완료 설정
    env.json             # 버전, 하드웨어, compile/precision 결정과 사유, git 해시
    status.json          # pending|running|done|failed|diverged + traceback
    raw/*.jsonl, raw/*.safetensors
    ckpt/task_<i>.safetensors
    metrics.json
  results.csv            # 전 run × 지표
  report.md, report.html, figs/
```
- `run_id` = 설정 해시. `status=done`이면 재실행 시 건너뜀 → 스윕 재시작 멱등
- 중단 후 재시작 시 마지막 태스크 체크포인트부터 재개

### 7.4 스케줄러
- 시드만 다른 run들을 하나의 E축 실행으로 묶음 (`engine.max_E`, 기본 8, 메모리 한도 초과 시 분할)
- 묶인 실행은 `workers × threads_per_worker ≤ 코어 수`로 프로세스 병렬. 기본: 비전 2×2, 텍스트 1×4
- 결과는 멤버별 run 디렉터리에 분리 저장 (vmap 사용 여부와 무관하게 동일 구조)

### 7.5 에러 처리
- run 단위 격리: 예외 시 `status=failed` + traceback, 스윕은 계속
- 발산: 손실 NaN/Inf 또는 손실이 초기값의 1e3배 초과 → 해당 멤버 `diverged`, 해당 멤버 그래디언트를 0으로 고정하고 나머지 멤버는 계속, 리포트에 표시
- compile 실패 → eager 폴백, `env.json`에 사유
- `vmappable=False` 전략 또는 vmap 실패 → E=1 프로세스 병렬로 자동 폴백, 사유 기록
- 데이터 다운로드 실패 → 캐시 경로와 수동 배치 방법을 포함한 에러 메시지
- 지표 계산 실패 → 해당 지표만 `null`과 사유, 나머지 지표는 계속

### 7.6 리포트
- 주 표: 벤치마크별 메서드 × {ACC, AIA, BWT, FWT, F, LA, min-ACC, WC-ACC, 시간, 메모리} (평균 ± 표준편차, 기준 대비 유의성 표시)
- 축별 표: 안정성 / 가소성 / 표현 / 가소성 손실 / 편향·신뢰도 / 복구성 / 효율
- 순위표와 CD 다이어그램, 6.5절 그림 전부
- 실행 환경 요약과 한계(시드 수, 서브샘플 규모, Class-IL 병합 헤드 스케일 문제)

## 8. 테스트

| 종류 | 내용 |
|---|---|
| 지표 단위 | 손으로 계산한 정확도 행렬·스트림 곡선·혼동 행렬로 6.3절 각 공식 검증, 경계 조건(T=1, Joint 부재) |
| 통계 단위 | 알려진 입력에 대한 permutation/Wilcoxon/Friedman 결과를 scipy 기준값과 비교 |
| 성능 코어 등가성 | vmap E=k 실행의 각 멤버 == 같은 시드의 E=1 개별 실행 (fp32, atol 1e-5, 몇 스텝), compile == eager |
| 전략 정합성 | EWC(λ=0) == Naive, ER(M=0) == Naive, DER++(α=β=0) == Naive, A-GEM에서 내적 ≥ 0이면 그래디언트 불변 |
| 통합 정합성 | 전문가 1개 TA(λ=1) == 전문가, TIES(k=100%, 전문가 1개, λ=1) == TA, LoRA ΔW 병합 모델 == 어댑터 활성 모델 출력 |
| 데이터 | 분할의 클래스 구성, 순열 고정성, 캐시 해시 재사용, 버퍼 인덱스 유효성 |
| 통합 | `smoke` 프로파일로 전 메서드·전 통합 기법·report·metrics 재계산이 60초 내 완주 |
| 실증 | `cpu` 프로파일 결과로 1.5절 성공 기준 3·4 확인 (자동 테스트가 아닌 실증 리포트) |

## 9. 위험과 대응
- **vmap + compile 호환성**: 일부 연산이 vmap/dynamo 미지원일 수 있음 → 7.5절 폴백, `bench`로 실제 이득 확인 후 기본값 결정
- **CPU 텍스트 실험 시간**: Pythia-70M, 길이 128, 태스크당 2k 샘플 기준으로 설계. 실측이 예산을 넘으면 `cpu` 프로파일의 텍스트 샘플 수를 줄이고 리포트에 명시
- **소형 LM의 낮은 절대 성능**: 메서드 간 상대 비교가 목적임을 리포트에 명시
- **시드 수 부족에 따른 통계 검정력**: 6.4절 경고 표시
- **HF 데이터셋 경로 변경**: 벤치마크 빌더가 데이터셋 ID를 설정으로 받음

# contilab Implementation Plan


**Goal:** 비전·LLM 공통 Continual Learning 실증 프레임워크(학습·증류/통합·풍부한 지표·리포트)를 성능 우선(E축 vmap, compile)으로 구현하고 CPU 실증 리포트까지 산출한다.

**Architecture:** 디바이스 상주 텐서 데이터(`TrainPool`) → 스택 파라미터 앙상블(`Ensemble`, 선행 E축, `vmap(functional_call)`) → 훅 기반 `Strategy` / 태스크 단위 `Consolidator` → 원자료만 기록하는 `Collectors`/`RawLog` → 실험 종료 후 순수 함수 지표(`metrics`) → 통계·리포트. 러너는 시드만 다른 run을 하나의 E축 실행으로 묶는다.

**Tech Stack:** Python 3.13, PyTorch(PyPI), transformers, datasets, safetensors, numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, pytest

**Spec:** `docs/design/2026-10-22-contilab-design.md` (절 번호 §로 참조)

## Global Constraints

- 패키지는 저장소 루트의 `contilab/`, 진입점 `contilab = contilab.runner.cli:main`
- 설치: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`; 모든 명령은 `.venv/bin/` 접두
- peft 사용 금지(자체 LoRA), `torch.utils.data.DataLoader` 사용 금지, BatchNorm 사용 금지(GroupNorm)
- 데이터 증강 없음
- 캐시 루트: 환경변수 `CONTILAB_CACHE`, 기본 `~/.cache/contilab`
- 네트워크가 필요한 테스트는 `@pytest.mark.network`(기본 skip, `--run-network`로 실행)
- 모든 학습 텐서는 선행 E축 `[E, ...]`; 멤버 i의 결과는 같은 시드로 E=1 실행한 결과와 같아야 함(fp32, atol 1e-5)
- 기본 하이퍼파라미터(§5.3): EWC λ=100(vision)/10(text), γ=1.0, Fisher N=1024; LwF T=2, α=1; ER/DER++/A-GEM M=500(vision)/200(text); DER++ α=0.1, β=0.5; A-GEM 참조 배치 256; LoRA r=8, α=16; TIES k=0.2; λ 그리드 {0.1,…,1.0}; 증류 T=2
- 옵티마이저 기본: vision SGD lr=0.03, text AdamW lr=1e-4, LoRA 학습 시 lr=1e-3; 배치 vision 32 / text 16
- 지표 수집 기본값(§6.2): ρ=50, 스트림 서브셋 태스크당 200, β=10, probe 태스크당 256, recovery n=50 / K=20, WF Δ=태스크 길이의 10%, ECE 15 bin, dormant τ=0.025, srank δ=0.01, ppl 200문장, 보간 11점
- 발산 판정: 손실 NaN/Inf 또는 첫 스텝 손실의 1e3배 초과
- 피크 메모리: CPU는 `resource.getrusage(RUSAGE_SELF).ru_maxrss*1024`(psutil 대신, 진짜 피크값), CUDA는 `torch.cuda.max_memory_allocated`. 이를 위해 워커는 `ProcessPoolExecutor(max_tasks_per_child=1)`

## Review Focus

1. **Task-IL에서 리플레이 샘플의 마스크** — 이전 태스크 샘플은 자기 태스크 클래스 마스크를 써야 함(현재 태스크 마스크면 CE가 -inf). → Task 11 `test_replay_uses_own_task_mask`
2. **태스크 1개 스트림 / Joint 부재** — BWT·FWT·Forgetting·Intransigence는 크래시 없이 `None`+사유. → Task 16 `test_single_task_metrics_none`
3. **중단 후 재실행** — `status=running`으로 남은 run이 마지막 태스크 체크포인트부터 재개되어 무중단 실행과 같은 정확도 행렬. → Task 24 `test_resume_matches_uninterrupted`
4. **시드 수가 max_E로 나누어떨어지지 않음** — 7시드, max_E=4 → [4, 3] 그룹. → Task 24 `test_group_runs_split`
5. **짧은/패딩 많은 텍스트** — last-token pooling이 마지막 비패딩 토큰을 선택(오른쪽 패딩). → Task 5 `test_last_token_pooling_ignores_padding`

---

## Phase A — 기반

### Task 1: 프로젝트 골격, 코어 타입, 시나리오 마스크

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `contilab/__init__.py`, `contilab/core/__init__.py`, `contilab/core/types.py`, `contilab/core/scenario.py`, `contilab/core/config.py`, `tests/conftest.py`, `tests/helpers.py`, `tests/core/test_types.py`, `tests/core/test_scenario.py`

**Interfaces:**
- Produces:
  - `TensorSplit(x: Tensor, y: Tensor, attn_mask: Tensor | None = None)`; `__len__`, `subset(idx: Tensor) -> TensorSplit`, `to(device) -> TensorSplit`
  - `Task(id: int, train: TensorSplit, val: TensorSplit, test: TensorSplit, classes: Tensor)`
  - `Stream(name: str, tasks: list[Task], scenario: Literal["task","class","domain"], num_classes: int, modality: Literal["vision","text"], lm_eval: TensorSplit | None = None)`; `class_table() -> Bool[T, C]`
  - `Batch(x, y, task_id, attn_mask=None, index=None, mask=None, extra: dict | None = None)` — 학습 시 모든 필드 `[E, B, ...]`
  - `RunContext(task_idx: int, step: int, device: torch.device, E: int, gens: list[torch.Generator], stream: Stream, active: Bool[E], diverged: set[int] = ∅, pool: TrainPool | None = None)`
  - `contilab/core/config.py` dataclass(전부 기본값 포함, 로딩·병합은 Task 23): `BenchmarkConfig(name, scenario, n_train=None, n_test=None, n_tasks=None, max_len=128, tokenizer=None, task_order_seed=None)`, `LoRAConfig(r=8, alpha=16, targets=None)`, `ModelConfig(name, hidden=400, depth=2, nf=20, hf_name=None, lora=LoRAConfig())`, `OptimConfig(name="sgd", lr=0.03, weight_decay=0.0, momentum=0.0, clip=None)`, `TrainConfig(epochs=1, batch_size=32, eval_batch_size=256)`, `EngineConfig(device="auto", compile="auto", precision="auto", max_E=8)`, `MethodConfig(name="naive", params={})`, `ConsolidateConfig(lam_grid=[0.1..1.0], ties_k=0.2, init="base", routing="entropy", transfer="buffer", M=500, temperature=2.0, epochs=1, experts_lora=False, init_seed=0)`, `MetricsConfig(tier="core", rho=50, stream_subset=200, beta=10, probe_size=256, recovery_n=50, recovery_steps=20, window_frac=0.1, ece_bins=15, dormant_tau=0.025, srank_delta=0.01, ppl_n=200, barrier_points=11)`, `RunConfig(experiment, preset, benchmark, model, optim, train, engine, method, consolidate, metrics, seed=0)`, `ExperimentConfig(name, profile, matrix: dict[str, list], overrides: dict, sweep: dict[str, list], workers=1, threads_per_worker=None, baseline="naive")`
  - `sample_mask(stream: Stream, task_ids: Tensor, current_task: int) -> Bool[*task_ids.shape, C]` (§4.5)

- [ ] **Step 1: 골격 작성** — `pyproject.toml`(deps: torch, transformers, datasets, safetensors, numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml; dev: pytest; script 진입점), `.gitignore`(`.venv/`, `runs/`, `__pycache__/`), `tests/conftest.py`(`--run-network` 옵션과 `network` 마커 skip 처리). 설치 명령 실행.

Run: `.venv/bin/python -c "import torch, transformers; print(torch.__version__)"`
Expected: 버전 출력

- [ ] **Step 2: 실패하는 테스트 작성**

```python
def test_class_table_and_masks():
    s = make_stream(classes=[[0, 1], [2, 3], [4, 5]], scenario="class", C=6)
    assert s.class_table().tolist()[1] == [False, False, True, True, False, False]
    m = sample_mask(s, torch.tensor([0, 2]), current_task=1)
    assert m.shape == (2, 6) and m[0].tolist() == [True]*4 + [False]*2 and torch.equal(m[0], m[1])
    s.scenario = "task"
    m = sample_mask(s, torch.tensor([0, 2]), current_task=2)
    assert m[0].tolist() == [True, True, False, False, False, False] and m[1, 4:].all()
    s.scenario = "domain"
    assert sample_mask(s, torch.tensor([[0, 1]]), 0).all()

def test_tensorsplit_subset_len():
    sp = TensorSplit(torch.arange(10.).view(10, 1), torch.arange(10))
    assert len(sp.subset(torch.tensor([1, 3]))) == 2 and sp.subset(torch.tensor([3])).y.item() == 3
```
(`make_stream`은 `tests/conftest.py`의 픽스처 헬퍼: 태스크별 클래스 목록으로 더미 Task 생성)

- [ ] **Step 3: 실패 확인** — Run: `.venv/bin/pytest tests/core -v` → FAIL (ImportError)
- [ ] **Step 4: `types.py`, `scenario.py` 구현** — `sample_mask`: task → `table[task_ids]`; class → `table[:current_task+1].any(0)`를 task_ids 형태로 broadcast; domain → 전부 True
- [ ] **Step 5: 통과 확인** — Run: `.venv/bin/pytest tests/core -v` → PASS
- [ ] **Step 6: 커밋** — `git add -A && git commit -m "feat: project scaffold, core types, scenario masks"`

### Task 2: 캐시, TrainPool 배칭, 합성 벤치마크, Joint 변환

**Files:**
- Create: `contilab/data/__init__.py`, `contilab/data/cache.py`, `contilab/data/batching.py`, `contilab/data/synthetic.py`, `contilab/data/transforms.py`, `contilab/data/splits.py`, `tests/data/test_batching.py`, `tests/data/test_synthetic.py`

**Interfaces:**
- Consumes: Task 1 타입
- Produces:
  - `cache_dir() -> Path`; `cached_tensors(namespace: str, key: dict, builder: Callable[[], dict[str, Tensor]]) -> dict[str, Tensor]` (sha256(정렬 JSON(key))[:16].safetensors)
  - `train_val_split(split: TensorSplit, frac: float = 0.1, seed: int = 0) -> tuple[TensorSplit, TensorSplit]`
  - `TrainPool.from_stream(stream) -> TrainPool` (필드 `x, y, task_id, attn_mask, offsets: Long[T+1]`); `gather(gidx: Long[E,B]) -> Batch` (`index=gidx`)
  - `epoch_perms(n: int, gens: list[torch.Generator]) -> Long[E, n]`
  - `task_batches(pool, task_idx: int, batch_size: int, gens) -> Iterator[Batch]` (전역 인덱스 = offsets[t] + 순열, 마지막 배치는 잘림 없이 작게)
  - `eval_batches(split: TensorSplit, batch_size: int) -> Iterator[tuple[Tensor, Tensor, Tensor | None]]`
  - `build_synthetic(n_tasks=5, classes_per_task=2, dim=32, n_train=200, n_test=100, seed=0, scenario="class") -> Stream` (가우시안 클러스터, modality "vision")
  - `build_synthetic_text(n_tasks=3, classes_per_task=2, vocab=128, seq_len=16, n_train=200, n_test=100, seed=0) -> Stream` (클래스별 고유 토큰 포함 시퀀스, `lm_eval`=무작위 시퀀스 64개, 오른쪽 패딩·`attn_mask` 포함)
  - `make_joint_stream(stream) -> Stream` (단일 태스크, `classes`=전체 합집합, 이름 `<name>-joint`)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_member_perms_follow_own_generator():
    g = lambda s: torch.Generator().manual_seed(s)
    p2 = epoch_perms(50, [g(1), g(2)]); p1 = epoch_perms(50, [g(2)])
    assert torch.equal(p2[1], p1[0]) and not torch.equal(p2[0], p2[1])

def test_task_batches_cover_task_once():
    s = build_synthetic(n_train=70); pool = TrainPool.from_stream(s)
    idx = torch.cat([b.index for b in task_batches(pool, 2, 32, [torch.Generator().manual_seed(0)])], 1)
    assert idx.shape == (1, 63)  # 70 - 10% val
    assert set(idx[0].tolist()) == set(range(pool.offsets[2].item(), pool.offsets[3].item()))
    assert (pool.task_id[idx] == 2).all()

def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path)); calls = []
    b = lambda: calls.append(1) or {"a": torch.ones(2)}
    cached_tensors("t", {"k": 1}, b); out = cached_tensors("t", {"k": 1}, b)
    assert calls == [1] and torch.equal(out["a"], torch.ones(2))

def test_joint_stream():
    j = make_joint_stream(build_synthetic())
    assert len(j.tasks) == 1 and len(j.tasks[0].classes) == 10
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/data -v` → FAIL
- [ ] **Step 3: 구현** — 합성 데이터는 `torch.Generator(seed)`로 결정적 생성; val은 `train_val_split`(frac 0.1, seed=태스크 id)
- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/pytest tests/data -v` → PASS
- [ ] **Step 5: 커밋** — `git commit -am "feat: data cache, TrainPool batching, synthetic benchmarks"` (새 파일 `git add` 포함)

### Task 3: 비전 벤치마크

**Files:**
- Create: `contilab/data/vision.py`, `contilab/data/registry.py`, `tests/data/test_vision.py`

**Interfaces:**
- Consumes: `cached_tensors`, `train_val_split`, Task 1 타입
- Produces:
  - `split_by_classes(x, y, groups: list[list[int]]) -> list[tuple[Tensor, Tensor]]`
  - `build_split_mnist(scenario: str, n_train=None, n_test=None) -> Stream` (5×2, `ylecun/mnist`, 평탄화 784, 정규화 0.1307/0.3081)
  - `build_permuted_mnist(n_tasks=10, n_train=None, n_test=None, seed=0) -> Stream` (scenario "domain", 태스크 0은 항등 순열)
  - `build_split_cifar10(scenario, n_train=None, n_test=None)`, `build_split_cifar100(scenario, ...)` (`uoft-cs/cifar10` 컬럼 `img,label`; `uoft-cs/cifar100` 컬럼 `img,fine_label`; 정규화 mean (0.4914,0.4822,0.4465) std (0.2470,0.2435,0.2616); `[3,32,32]`)
  - `build_stream(cfg: BenchmarkConfig) -> Stream` 레지스트리(이름 → 빌더; synthetic/synthetic_text 포함; `task_order_seed` 지정 시 태스크 셔플)
  - `n_train/n_test`는 태스크당 서브샘플 수(None=전체), 서브샘플은 고정 시드 0

- [ ] **Step 1: 실패하는 테스트 작성** (네트워크 없는 단위 + 네트워크 통합)

```python
def test_split_by_classes():
    y = torch.tensor([0, 1, 2, 3, 0, 2]); x = torch.arange(6.).view(6, 1)
    parts = split_by_classes(x, y, [[0, 1], [2, 3]])
    assert parts[0][1].tolist() == [0, 1, 0] and parts[1][0].view(-1).tolist() == [2., 3., 5.]

@pytest.mark.network
def test_split_mnist_shapes():
    s = build_split_mnist("class", n_train=100, n_test=50)
    assert len(s.tasks) == 5 and s.tasks[0].train.x.shape[1] == 784 and s.num_classes == 10
    assert s.tasks[4].classes.tolist() == [8, 9] and len(s.tasks[0].test) == 50

@pytest.mark.network
def test_permuted_mnist_first_task_identity():
    s = build_permuted_mnist(n_tasks=3, n_train=50, n_test=20)
    assert s.scenario == "domain" and s.tasks[0].classes.tolist() == list(range(10))
    assert not torch.equal(s.tasks[1].test.x, s.tasks[0].test.x)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/data/test_vision.py -v --run-network` → FAIL
- [ ] **Step 3: 구현** — `datasets.load_dataset`로 로드 → numpy → 텐서, 전처리 결과는 `cached_tensors("vision", {...})`로 캐시. 다운로드 실패 시 `RuntimeError`에 캐시 경로와 "`contilab data prepare <name>`을 네트워크 가능한 환경에서 실행 후 캐시 디렉터리를 복사" 문구 포함(§7.5)
- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/pytest tests/data/test_vision.py -v --run-network` → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: vision benchmarks (split/permuted MNIST, split CIFAR)"`

### Task 4: 텍스트 벤치마크 text5, 프록시 데이터

**Files:**
- Create: `contilab/data/text.py`, `contilab/data/proxy.py`, `tests/data/test_text.py`
- Modify: `contilab/data/registry.py` (text5 등록)

**Interfaces:**
- Produces:
  - `TEXT5_SOURCES`: 순서대로 `(dataset_id, text_fn, label_offset, n_classes)` — AG(`fancyzhx/ag_news`, 0, 4), Yelp(`Yelp/yelp_review_full`, 4, 5), Amazon(`SetFit/amazon_reviews_multi_en`, 4, 5), DBpedia(`fancyzhx/dbpedia_14`, 9, 14; text = title + " " + content), Yahoo(`community-datasets/yahoo_answers_topics`, 23, 10; text = question_title + " " + question_content + " " + best_answer). 총 33 클래스
  - `tokenize(texts: list[str], tokenizer_name: str, max_len: int) -> tuple[Long[N,L], Long[N,L]]` (오른쪽 패딩, pad 토큰 없으면 eos 사용)
  - `build_text5(tokenizer_name: str, scenario="class", n_train=2000, n_test=500, max_len=128) -> Stream` (`lm_eval` = `Salesforce/wikitext` `wikitext-2-raw-v1` test의 비어있지 않은 첫 200줄)
  - `build_proxy(stream_name: str, n: int, tokenizer_name: str | None = None) -> TensorSplit` (§5.5: MNIST 계열 → CIFAR-10 그레이스케일 28×28 평탄화·MNIST 정규화, split_cifar10 → CIFAR-100, split_cifar100 → CIFAR-10, text5 → wikitext train 문장, synthetic → 표준정규 노이즈, synthetic_text → 무작위 토큰)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_text5_label_layout():
    offs = [(o, n) for _, _, o, n in TEXT5_SOURCES]
    assert offs == [(0, 4), (4, 5), (4, 5), (9, 14), (23, 10)]

@pytest.mark.network
def test_build_text5_small():
    s = build_text5("EleutherAI/pythia-70m", n_train=20, n_test=10, max_len=32)
    assert s.num_classes == 33 and s.tasks[1].classes.tolist() == s.tasks[2].classes.tolist()
    x, m = s.tasks[0].train.x, s.tasks[0].train.attn_mask
    assert x.shape[1] == 32 and m[:, 0].all() and s.lm_eval is not None

def test_proxy_synthetic_shape():
    assert build_proxy("synthetic", 16).x.shape == (16, 32)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/data/test_text.py -v --run-network` → FAIL
- [ ] **Step 3: 구현** — 토크나이즈 결과를 `cached_tensors("text", {...tokenizer, max_len, n_train, n_test})`로 캐시
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: text5 benchmark and proxy transfer data"`

### Task 5: 모델 (MLP, ReducedResNet18, HFClassifier)

**Files:**
- Create: `contilab/models/__init__.py`, `contilab/models/base.py`, `contilab/models/mlp.py`, `contilab/models/resnet.py`, `contilab/models/hf.py`, `contilab/models/registry.py`, `tests/models/test_models.py`

**Interfaces:**
- Produces:
  - `CLModel(nn.Module)`: 속성 `head: nn.Linear`; `forward(x, attn_mask=None, mode: str = "logits")` — mode ∈ {`"logits"` → `[B,C]`, `"features"` → `[B,D]`, `"acts"` → `list[Tensor[B,units]]`, `"lm"` → `[B,L,V]`(HF 전용)}. `functional_call`이 forward만 부르므로 모든 경로를 mode로 노출
  - `MLP(in_dim, num_classes, hidden=400, depth=2)` — acts = 각 은닉층 ReLU 출력
  - `ReducedResNet18(num_classes, nf=20, in_ch=3)` — GroupNorm(groups=min(32, ch)), acts = 각 stage 출력의 global-avg-pool
  - `HFClassifier(hf_name, num_classes, pooling="last_token")` — `AutoModelForCausalLM` 보유; features = 마지막 hidden state의 마지막 비패딩 토큰(`attn_mask.sum(-1)-1`); `"lm"`은 원래 LM 헤드 사용; acts = 각 레이어 hidden state의 토큰 평균. `hf_name`이 `"tiny-"`로 시작하면 다운로드 없이 `GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)`로 생성
  - `build_model(model_cfg, stream: Stream) -> CLModel` (`model_cfg.name` ∈ {mlp, resnet18r, hf}; in_dim은 stream에서 추론)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_mlp_modes():
    m = MLP(32, 10); x = torch.randn(4, 32)
    assert m(x).shape == (4, 10) and m(x, mode="features").shape == (4, 400) and len(m(x, mode="acts")) == 2

def test_resnet_no_batchnorm():
    m = ReducedResNet18(10)
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert not any(isinstance(l, nn.modules.batchnorm._BatchNorm) for l in m.modules())

def test_last_token_pooling_ignores_padding():
    m = HFClassifier("tiny-gptneox", 6).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0], [5, 6, 7, 9, 9]]); mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    f_pad = m(ids, mask, mode="features")[0]
    f_short = m(ids[:1, :3], mask[:1, :3], mode="features")[0]
    assert torch.allclose(f_pad, f_short, atol=1e-5)
    assert m(ids, mask, mode="lm").shape == (2, 5, 128)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/models/test_models.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: MLP, ReducedResNet18, HF classifier models"`

### Task 6: 자체 LoRA

**Files:**
- Create: `contilab/models/lora.py`, `tests/models/test_lora.py`

**Interfaces:**
- Produces:
  - `LoRALinear(base: nn.Linear, r, alpha, n_adapters)`, `LoRAConv2d(base: nn.Conv2d, r, alpha, n_adapters)` — 파라미터 이름 `lora_A.t{i}`, `lora_B.t{i}`(ParameterDict); A는 kaiming-uniform, B=0; Conv는 A `[r, in·k·k]`, B `[out, r]`, ΔW를 커널 형태로 reshape
  - `inject_lora(model, targets: list[str] | None, r: int, alpha: int, n_adapters: int, modality: str) -> list[str]` — 기본 대상: text는 이름에 `query_key_value`/`q_proj`/`v_proj` 포함 Linear, vision은 `head`를 제외한 모든 Linear/Conv2d. 감싼 모듈 이름 목록 반환
  - `set_active(model, adapters: list[int], scale: float = 1.0) -> None` — 출력 = base(x) + scale·Σ_t (α/r)·B_t A_t 경로
  - `lora_param_names(model, adapter: int) -> list[str]`
  - `lora_task_vector(params: dict[str, Tensor], adapter: int, lora_modules: list[str], r: int, alpha: int) -> dict[str, Tensor]` — 키 = `<module>.base.weight`, 값 = ΔW (선행 E축 유지)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_lora_starts_as_identity_and_merges():
    torch.manual_seed(0); m = MLP(8, 4, hidden=16); ref = copy.deepcopy(m); x = torch.randn(3, 8)
    names = inject_lora(m, None, r=2, alpha=4, n_adapters=2, modality="vision")
    set_active(m, [0]); assert torch.allclose(m(x), ref(x))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B.t0" in n: p.normal_()
    p = {k: v.unsqueeze(0) for k, v in m.state_dict().items()}
    tv = lora_task_vector(p, 0, names, r=2, alpha=4)
    merged = copy.deepcopy(m); set_active(merged, [])
    with torch.no_grad():
        for k, d in tv.items(): merged.get_parameter(k).add_(d[0])
    assert torch.allclose(m(x), merged(x), atol=1e-5)

def test_lora_conv_shape():
    c = LoRAConv2d(nn.Conv2d(3, 5, 3, padding=1), r=2, alpha=4, n_adapters=1)
    assert c(torch.randn(1, 3, 8, 8)).shape == (1, 5, 8, 8)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/models/test_lora.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: in-house LoRA for Linear/Conv2d with ΔW task vectors"`

## Phase B — 성능 코어

### Task 7: 디바이스·정밀도 선택, Ensemble

**Files:**
- Create: `contilab/engine/__init__.py`, `contilab/engine/device.py`, `contilab/engine/ensemble.py`, `tests/engine/test_ensemble.py`

**Interfaces:**
- Consumes: `CLModel`
- Produces:
  - `select_device(pref: str) -> torch.device` (auto: cuda > mps > cpu); `select_precision(pref: str, device) -> tuple[torch.dtype | None, str]` (dtype None=fp32, 두 번째는 사유; CPU bf16은 `torch.backends.cpu.get_cpu_capability()`가 AVX512 계열/AMX일 때만); `peak_memory_bytes(device) -> int`
  - `Ensemble(factory: Callable[[int], CLModel], init_seeds: list[int], device)`: 속성 `E`, `template`, `params: dict[str, Tensor[E,...]]`(모두 leaf, requires_grad=True), `buffers`
  - `Ensemble.apply(params, x, attn_mask=None, mode="logits", shared_input=False) -> Tensor[E, ...]` — E>1이면 `vmap(functional_call)`(x의 in_dim은 shared_input이면 None), E==1이면 `p[0]` 인덱싱 후 직접 호출하고 unsqueeze(0) (§4.4 fast path). mode="acts"는 리스트 반환
  - `Ensemble.member_state(i) -> dict[str, Tensor]`, `load_member_state(i, sd)`, `snapshot() -> dict[str, Tensor]`(detach clone), `load_params(p: dict)`(in-place copy)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_members_match_individual_modules():
    fac = lambda s: (torch.manual_seed(s), MLP(8, 4, hidden=16))[1]
    ens = Ensemble(fac, [1, 2, 3], torch.device("cpu")); x = torch.randn(5, 8)
    out = ens.apply(ens.params, x, shared_input=True)
    for i, s in enumerate([1, 2, 3]):
        assert torch.allclose(out[i], fac(s)(x), atol=1e-6)

def test_e1_fast_path_and_per_member_inputs():
    fac = lambda s: (torch.manual_seed(s), MLP(8, 4, hidden=16))[1]
    e1 = Ensemble(fac, [7], torch.device("cpu")); x = torch.randn(1, 5, 8)
    assert torch.allclose(e1.apply(e1.params, x)[0], fac(7)(x[0]), atol=1e-6)
    e2 = Ensemble(fac, [7, 8], torch.device("cpu")); xs = torch.randn(2, 5, 8)
    assert e2.apply(e2.params, xs).shape == (2, 5, 4)

def test_select_device_cpu():
    assert select_device("cpu").type == "cpu" and select_precision("fp32", torch.device("cpu"))[0] is None
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/engine/test_ensemble.py -v` → FAIL
- [ ] **Step 3: 구현** — `torch.func.stack_module_state`로 스택 후 각 텐서를 `detach().requires_grad_()`
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: device/precision selection and stacked-parameter Ensemble"`

### Task 8: Strategy 기반, Naive, 옵티마이저, TrainStep, 평가기

**Files:**
- Create: `contilab/strategies/__init__.py`, `contilab/strategies/base.py`, `contilab/strategies/naive.py`, `contilab/engine/optim.py`, `contilab/engine/step.py`, `contilab/engine/evaluate.py`, `tests/engine/test_step.py`

**Interfaces:**
- Consumes: `Ensemble`, `Batch`, `RunContext`, `sample_mask`
- Produces:
  - `MemberFns`: `logits(p, x, attn_mask, mask) -> Tensor` (`masked_fill(~mask, -1e9)` — `-inf` 대신 유한값으로 CE/KL 안정성 확보), `features(p, x, attn_mask) -> Tensor`
  - `Strategy` (§5.1) — 멤버 단위 핫패스 시그니처 확정:
    - `loss(self, p, s, batch, replay, f: MemberFns) -> tuple[Tensor, Tensor]` (스칼라 손실, 현재 배치 로짓)
    - `ref_loss(self, p, s, replay, f) -> Tensor | None` (기본 None)
    - `transform_grad(self, grads: dict[str, Tensor[E,...]], ref_grads: dict | None) -> dict` (스택 그래디언트, 호스트 측)
    - `prepare_model(self, model: CLModel, stream: Stream) -> CLModel` (기본 항등)
    - `trainable_names(self, names: list[str], task_idx: int) -> list[str]` (기본 전부)
    - `state(self) -> dict[str, Tensor[E,...]]`(핫패스로 전달), `state_dict() -> dict`, `load_state_dict(d) -> None` (기본: `self.tensors`, `self.meta`)
    - 호스트 훅 `before_task/replay_batch/after_step/after_task`, `eval_setup(ctx, task_eval: int) -> None`(기본 no-op, 평가기가 태스크별 평가 직전 호출), 클래스 속성 `vmappable = True`, `name`
  - `register_strategy(name)` 데코레이터, `build_strategy(name: str, params: dict, modality: str) -> Strategy` (클래스 속성 `DEFAULTS = {"vision": {...}, "text": {...}}`와 병합)
  - `Naive`: CE(+replay가 있어도 무시)
  - `build_optimizer(params: list[Tensor], cfg: OptimConfig) -> torch.optim.Optimizer` (sgd/adamw, `foreach=True`); `clip_per_member_(grads: dict, max_norm: float) -> None`
  - `StepOutput(loss: Tensor[E], logits: Tensor[E,B,C])` (detach)
  - `TrainStep(ens, strategy, stream, engine_cfg, optim_cfg)`: `set_task(task_idx)`(trainable 이름 갱신·옵티마이저 재생성), `__call__(batch, replay, ctx) -> StepOutput`, `compile_decision: dict` (`requested`, `used`, `reason`). 손실 함수 `vmap(member_loss)`를 `maybe_compile`로 감싸고, 그래디언트는 `torch.autograd.grad(loss.sum(), trainable)`; ref_loss가 있으면 두 번째 grad; `transform_grad` → `ctx.active`로 비활성 멤버 grad를 0(`nan_to_num` 후 곱) → `.grad` 할당 → step. `select_precision`이 bf16을 반환하면 손실 계산을 `torch.autocast(device.type, dtype=torch.bfloat16)`로 감쌈
  - `maybe_compile(fn, mode: str) -> tuple[Callable, dict]` — auto: 첫 호출에서 컴파일 시도, 예외 시 eager로 교체하고 사유 기록
  - `evaluate_split(ens, params, split, mask_row: Bool[C], batch_size) -> EvalResult(correct: Bool[E,N], pred: Long[E,N], conf: Tensor[E,N], logits: Tensor[E,N,C] | None)` (`shared_input=True`, `torch.inference_mode`)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def run_naive(seeds, steps=3, compile="off"):
    s = build_synthetic(); ens = Ensemble(lambda k: (torch.manual_seed(k), MLP(32, 10, hidden=16))[1], seeds, CPU)
    ctx = make_ctx(s, seeds); step = TrainStep(ens, Naive(), s, EngineConfig(compile=compile), OptimConfig(lr=0.1))
    step.set_task(0); pool = TrainPool.from_stream(s)
    for b, _ in zip(task_batches(pool, 0, 32, ctx.gens), range(steps)):
        b.mask = sample_mask(s, b.task_id, 0); step(b, None, ctx)
    return ens

def test_vmap_members_equal_individual_runs():
    e3 = run_naive([1, 2, 3])
    for i, sd in enumerate([1, 2, 3]):
        e1 = run_naive([sd])
        for k in e3.params: assert torch.allclose(e3.params[k][i], e1.params[k][0], atol=1e-5)

def test_compile_equals_eager():
    a, b = run_naive([1], compile="off"), run_naive([1], compile="on")
    for k in a.params: assert torch.allclose(a.params[k], b.params[k], atol=1e-5)

def test_inactive_member_frozen_others_unaffected():
    # ctx.active[1]=False 이후 member 1 파라미터 불변, member 0은 E=1 실행과 동일
    ...  # run_naive 변형: active=[True, False]; 위 두 조건 assert

def test_evaluate_split_accuracy_shape():
    ens = run_naive([1, 2]); s = build_synthetic()
    r = evaluate_split(ens, ens.params, s.tasks[0].test, s.class_table()[0], 64)
    assert r.correct.shape == (2, 100) and r.pred.max() <= 1
```
(`make_ctx`는 `tests/conftest.py` 헬퍼: `RunContext(task_idx=0, step=0, device=CPU, E=len(seeds), gens=[Generator(s)], stream=s, active=ones(E, bool))`)

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/engine/test_step.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS (compile 테스트가 환경상 컴파일 불가로 폴백되면 `compile_decision["used"]`를 출력하고 `pytest.skip`)
- [ ] **Step 5: 커밋** — `git commit -m "feat: strategy base, naive, compiled vmapped train step, evaluator"`

### Task 9: 학습 루프와 발산 처리

**Files:**
- Create: `contilab/engine/trainer.py`, `tests/engine/test_trainer.py`

**Interfaces:**
- Consumes: `TrainStep`, `TrainPool`, `task_batches`, `Strategy`, `sample_mask`
- Produces:
  - `CollectorHooks` 프로토콜: `on_run_start(ctx)`, `on_task_start(ctx)`, `on_step(ctx, out: StepOutput)`, `on_task_end(ctx)`, `on_run_end(ctx)`; `NullHooks`
  - `train_stream(ens, strategy, stream, train_cfg, engine_cfg, optim_cfg, hooks: CollectorHooks, ctx: RunContext, start_task: int = 0, on_task_done: Callable[[int], None] | None = None) -> TrainStep` — 태스크마다 `strategy.before_task` → `step.set_task` → 에폭·배치 루프(마스크 설정, `replay_batch`, step, `after_step`, 발산 검사, `hooks.on_step`) → `after_task` → `hooks.on_task_end` → `on_task_done(t)`(체크포인트용)
  - 발산 검사: 멤버별 첫 손실 기록, `~isfinite | loss > 1e3·first` → `ctx.active[e]=False`, `ctx.diverged`에 추가
  - `tests/helpers.py`(이후 태스크 공용): `STREAM = build_synthetic()`; `train_on(stream, strategy, seeds, epochs=1, tasks=None, poison_member=None, model="mlp") -> tuple[Ensemble, RunContext]`(MLP hidden=16 또는 `tiny-gptneox`, SGD lr 0.1, `tasks=k`면 앞 k개 태스크만 학습); `params_equal(a: Ensemble, b: Ensemble, atol=1e-6) -> bool`; `final_acc(stream, strategy, seeds) -> float`(멤버 평균 최종 ACC); `per_task_final_acc(stream, strategy, seeds) -> list[float]`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_naive_learns_task_il():
    s = build_synthetic(scenario="task"); ens, ctx = train_on(s, Naive(), seeds=[0], epochs=3)
    acc = [evaluate_split(ens, ens.params, t.test, s.class_table()[t.id], 256).correct.float().mean().item() for t in s.tasks]
    assert min(acc) > 0.9

def test_nan_member_marked_diverged():
    s = build_synthetic(); ens, ctx = train_on(s, Naive(), seeds=[0, 1], epochs=1, poison_member=1)
    assert ctx.diverged == {1} and all(torch.isfinite(p[0]).all() for p in ens.params.values())
```
(`poison_member`는 첫 스텝 전 해당 멤버 파라미터 하나에 NaN 주입)

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/engine/test_trainer.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: stream training loop with divergence isolation"`

## Phase C — CL 메서드

### Task 10: EWC(Online), LwF

**Files:**
- Create: `contilab/strategies/ewc.py`, `contilab/strategies/lwf.py`, `tests/strategies/test_reg.py`

**Interfaces:**
- Consumes: `Strategy`, `MemberFns`, `Ensemble`
- Produces:
  - `EWC(lam, gamma=1.0, fisher_n=1024)`: `after_task`에서 현재 태스크 학습 샘플 N개(멤버별 gen)로 per-sample grad를 `vmap(grad)`(멤버×샘플 중첩 vmap, 128 청크) → 제곱 평균 → `F ← γF + F_new`, `θ* ← 현재 파라미터`; `state()` = `{"F.<name>", "star.<name>"}`; 손실 `CE + λ/2·ΣF(θ−θ*)²`(첫 태스크는 페널티 없음)
  - `LwF(T=2.0, alpha=1.0)`: `after_task`에서 파라미터 스냅샷을 `state()`에 `old.<name>`로 보관, 이전 태스크까지 본 클래스 마스크 `old_mask: Bool[C]`; 손실 `CE + α·T²·KL(softmax(z_old/T) ‖ softmax(z/T))`(이전 클래스로 제한, 첫 태스크는 KD 없음)
  - 둘 다 `register_strategy("ewc")`, `("lwf")`, `DEFAULTS` = Global Constraints 값

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_ewc_lambda0_equals_naive():
    assert params_equal(train_on(STREAM, EWC(lam=0.0), [0])[0], train_on(STREAM, Naive(), [0])[0])

def test_fisher_nonnegative_and_shaped():
    ens, _ = train_on(STREAM, ewc := EWC(lam=1.0), [0, 1], tasks=1)
    F = {k: v for k, v in ewc.state().items() if k.startswith("F.")}
    assert all((v >= 0).all() and v.shape[0] == 2 for v in F.values())

def test_lwf_alpha0_equals_naive():
    assert params_equal(train_on(STREAM, LwF(alpha=0.0), [0])[0], train_on(STREAM, Naive(), [0])[0])

def test_ewc_reduces_forgetting_on_permuted_synthetic():
    # domain 시나리오 합성 스트림에서 첫 태스크 최종 정확도: EWC(λ=100) > Naive
    ...
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/strategies/test_reg.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: online EWC and LwF strategies"`

### Task 11: 리플레이 버퍼, ER, DER++

**Files:**
- Create: `contilab/strategies/buffer.py`, `contilab/strategies/er.py`, `contilab/strategies/derpp.py`, `tests/strategies/test_replay.py`

**Interfaces:**
- Consumes: `TrainPool.gather`, `sample_mask`, `Strategy`
- Produces:
  - `ReservoirBuffer(E, M, C | None, device)`: tensors `gidx: Long[E,M]`(-1=빈칸), `logits: Tensor[E,M,C]`(C 지정 시), meta `seen: list[int]`; `add(batch_index: Long[E,B], logits: Tensor[E,B,C] | None, gens)`(멤버별 표준 reservoir: seen<M이면 채우고, 아니면 j~U[0,seen]가 <M일 때 교체); `sample(n, gens) -> Long[E,n] | None`(채워진 칸에서 비복원, 부족하면 가능한 만큼, 비면 None); `size() -> Long[E]`
  - `ER(M)`: `replay_batch` = pool.gather(sample(batch_size)) + 각 샘플 **자기 task_id**로 `sample_mask` 계산; 손실 `CE(cat(cur, replay))`; `after_step`에서 현재 배치 인덱스 add
  - `DERpp(M, alpha=0.1, beta=0.5)`: 버퍼에 삽입 시점 로짓 저장; replay 두 개(로짓용·라벨용)를 한 Batch에 이어 붙여 전달; 손실 `CE(cur) + α·MSE(z_r1, z_buf) + β·CE(z_r2, y_r2)`; 저장 로짓은 `state()`가 아닌 replay Batch에 동반 필드로 전달(`Batch.extra: dict | None` 추가)
  - 공통: pool은 `ctx.pool`로 접근(`train_stream`이 설정), 버퍼는 첫 `before_task`에서 `ctx.E`로 생성

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_reservoir_fills_then_keeps_size():
    b = ReservoirBuffer(E=2, M=10, C=None, device=CPU); gens = [Generator(0), Generator(1)]
    for i in range(10): b.add(torch.arange(i*5, i*5+5).expand(2, 5), None, gens)
    assert b.size().tolist() == [10, 10] and not torch.equal(b.tensors["gidx"][0], b.tensors["gidx"][1])

def test_reservoir_is_roughly_uniform():
    # 1000개 스트림, M=100, 200회 반복 시 앞/뒤 절반 포함 비율 차이 < 0.05
    ...

def test_er_m0_equals_naive():
    assert params_equal(train_on(STREAM, ER(M=0), [0])[0], train_on(STREAM, Naive(), [0])[0])

def test_derpp_zero_coeffs_equals_naive():
    assert params_equal(train_on(STREAM, DERpp(M=50, alpha=0, beta=0), [0])[0], train_on(STREAM, Naive(), [0])[0])

def test_replay_uses_own_task_mask():
    s = build_synthetic(scenario="task"); ens, ctx = train_on(s, er := ER(M=50), [0], tasks=1)
    ctx.task_idx = 1; r = er.replay_batch(ctx)
    assert r.mask[0, :, :2].all() and not r.mask[0, :, 2:].any()   # 태스크 0 샘플만 버퍼에 있음 → 태스크 0 클래스 마스크

def test_er_beats_naive_class_il():
    s = build_synthetic(scenario="class")
    assert final_acc(s, ER(M=100), [0, 1]) > final_acc(s, Naive(), [0, 1]) + 0.1
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/strategies/test_replay.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: index-based reservoir buffer, ER, DER++"`

### Task 12: A-GEM

**Files:**
- Create: `contilab/strategies/agem.py`, `tests/strategies/test_agem.py`

**Interfaces:**
- Consumes: `ReservoirBuffer`, `Strategy.ref_loss`, `transform_grad`
- Produces:
  - `project_agem(g: dict[str, Tensor[E,...]], g_ref: dict[str, Tensor[E,...]]) -> dict` — 멤버별 평탄화 내적 d; d<0인 멤버만 `g − (d/‖g_ref‖²)·g_ref`
  - `AGEM(M, ref_batch=256)`: `replay_batch`는 참조 배치(크기 ref_batch); `loss`는 현재 배치 CE만; `ref_loss`는 replay CE; `transform_grad`는 `project_agem`(ref 없으면 항등)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_projection_only_when_conflicting():
    g = {"w": torch.tensor([[1., 0.], [1., 0.]])}; r = {"w": torch.tensor([[1., 1.], [-1., 0.]])}
    out = project_agem(g, r)["w"]
    assert torch.equal(out[0], g["w"][0])                       # 내적 ≥ 0 → 불변
    assert torch.allclose(out[1], torch.tensor([0., 0.]))        # 내적 < 0 → 투영
    assert (out[1] * r["w"][1]).sum() >= -1e-6

def test_agem_runs_and_differs_from_naive():
    assert not params_equal(train_on(STREAM, AGEM(M=50), [0])[0], train_on(STREAM, Naive(), [0])[0])
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/strategies/test_agem.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: A-GEM with per-member gradient projection"`

### Task 13: LoRA-per-task

**Files:**
- Create: `contilab/strategies/lora_per_task.py`, `tests/strategies/test_lora_pt.py`

**Interfaces:**
- Consumes: `inject_lora`, `set_active`, `lora_param_names`, `Strategy.prepare_model/trainable_names`, `evaluate_split`
- Produces:
  - `LoRAPerTask(r=8, alpha=16, targets=None, lam_grid=(0.1, …, 1.0))`: `prepare_model`에서 `n_adapters = len(stream.tasks)`로 주입; `trainable_names(names, t)` = 어댑터 t의 LoRA 파라미터 + `head.*`; `before_task`에서 `set_active(template, [t])`
  - `eval_setup(self, ctx, task_eval: int) -> None` — Task-IL: `set_active([task_eval])`; Class/Domain-IL: `set_active(range(t+1), scale=λ*)`, λ*는 `after_task`에서 본 태스크 검증 세트 평균 정확도 최대값으로 선택(멤버 0 기준, 동률은 작은 λ)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_base_frozen_and_only_active_adapter_trains():
    s = build_synthetic(scenario="task"); before = snapshot_of_initial(s)
    ens, _ = train_on(s, LoRAPerTask(r=2, alpha=4), [0], tasks=2)
    for k, v in ens.params.items():
        if "lora" not in k and not k.startswith("head"): assert torch.equal(v, before[k])
        if "lora_B.t4" in k: assert (v == 0).all()     # 아직 학습 안 된 어댑터

def test_task_il_adapter_selection_beats_chance():
    s = build_synthetic(scenario="task"); assert min(per_task_final_acc(s, LoRAPerTask(r=4, alpha=8), [0])) > 0.6
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/strategies/test_lora_pt.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: LoRA-per-task strategy with merged class-IL evaluation"`

## Phase D — 원자료 수집

### Task 14: RawLog와 core 수집기

**Files:**
- Create: `contilab/metrics/__init__.py`, `contilab/metrics/rawlog.py`, `contilab/metrics/collectors.py`, `tests/metrics/test_collectors_core.py`

**Interfaces:**
- Consumes: `CollectorHooks`, `evaluate_split`, `Strategy.eval_setup`, `peak_memory_bytes`
- Produces:
  - `RawLog(dir: Path)`: `append(key, record: dict)` → `raw/<key>.jsonl`; `put_tensor(key, name, t)` + `flush()` → `raw/<key>.safetensors`; `records(key) -> list[dict]`; `tensors(key) -> dict[str, Tensor]`; `has(key) -> bool`; `truncate_after_task(t: int)`(모든 jsonl에서 `task > t` 레코드 제거, 텐서 키 `t{i}.*` 중 i>t 제거)
  - `RAW_KEYS_BY_TIER: dict[str, set[str]]`(§6.2 표 그대로, 상위 티어는 하위 포함) — 수집 여부 결정
  - `Collectors(tier: str, logs: list[RawLog], stream, ens, strategy, metrics_cfg, train_cfg, eval_stream: Stream | None = None)` — `CollectorHooks` 구현. `eval_stream`이 주어지면 평가는 그 스트림의 태스크 기준(Joint용). core 키(§6.2): `init_acc`({"accs": [T]}), `acc_matrix`({"task": i, "accs": [T]} — 미래 태스크 포함 전 태스크 평가), `pred_hist`(전 태스크 test 연결에 대한 텐서 `t{i}.pred`, `t{i}.y`, `t{i}.conf`, `t{i}.task`), `cost`({"task": i, "wall_s", "samples", "samples_per_s", "peak_mem", "trainable_params", "buffer_bytes", "model_bytes"} + run 끝 `{"flops_per_step", "steps"}`)
  - `estimate_step_flops(step: TrainStep, batch: Batch, replay, ctx) -> int` (`torch.utils.flop_counter.FlopCounterMode`, eager 경로)
  - 수집 결과는 멤버별 RawLog에 분리 기록(멤버 e의 값만)
  - `tests/helpers.py`에 `run_with_collectors(stream, strategy, seeds, tier, logs, rho=50, beta=10, model="mlp") -> None` 추가

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_core_collection(tmp_path):
    logs = [RawLog(tmp_path / f"m{e}") for e in range(2)]
    run_with_collectors(build_synthetic(), Naive(), [0, 1], tier="core", logs=logs)
    rows = logs[0].records("acc_matrix")
    assert [r["task"] for r in rows] == [0, 1, 2, 3, 4] and all(len(r["accs"]) == 5 for r in rows)
    assert len(logs[1].records("init_acc")[0]["accs"]) == 5
    assert logs[0].tensors("pred_hist")["t4.pred"].shape == (100 * 5,)
    c = logs[0].records("cost"); assert c[-1]["flops_per_step"] > 0 and c[0]["samples_per_s"] > 0

def test_rawlog_truncate(tmp_path):
    l = RawLog(tmp_path); [l.append("acc_matrix", {"task": i, "accs": [i]}) for i in range(3)]
    l.truncate_after_task(0); assert [r["task"] for r in l.records("acc_matrix")] == [0]
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_collectors_core.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: RawLog storage and core-tier collectors"`

### Task 15: extended/full 수집기

**Files:**
- Modify: `contilab/metrics/collectors.py`
- Create: `tests/metrics/test_collectors_ext.py`

**Interfaces:**
- Produces (키 형식, 모두 멤버별):
  - extended: `stream_acc`({"step", "task_now", "accs": [seen 태스크별, 서브셋 200]}, ρ 스텝마다 + 각 태스크 종료 시), `lca_curve`({"task", "batch", "acc"}, 각 태스크 첫 β+1 배치(0~β) 직전 현재 태스크 서브셋 정확도), `features`(텐서 `t{i}.task{j}` = probe 세트 특징, `t{i}.task{j}.y` = 라벨, j ≤ T−1 전체; probe 크기 = min(256, test 크기)), `activations`(텐서 `t{i}.layer{l}` = 뉴런별 평균 |h|), `general_ppl`({"task": -1|i, "ppl"}; stream.lm_eval이 있을 때만, `-1`은 θ₀)
  - full: `param_snapshots`(체크포인트 재사용 — 수집기는 경로만 기록), `barrier_curve`({"task": k, "alphas": [11], "losses": [11]}; 태스크 k+1 종료 시 θ_k(이전 스냅샷 보관)와 θ_{k+1} 사이 보간, 태스크 k 검증 손실), `grad_cosine`({"task": i, "matrix": [[...]]}; 태스크 i 종료 시 본 태스크별 검증 배치(256) 손실 그래디언트 코사인 — 저장량 제한을 위해 그래디언트 대신 행렬만 기록), `recovery`({"task": j, "acc_before", "acc_after", "acc_at_learn"}; 실행 종료 후 모델 복사본을 태스크 j 학습 샘플 50개로 20 스텝 SGD(lr = 학습 lr) 재학습)
  - consolidation 경로에서 on_step 기반 키(`stream_acc`, `lca_curve`)는 `{"not_applicable": true}` 1건 기록
  - probe 세트: 각 태스크 test에서 고정 시드로 256개 선택(평가 서브셋도 같은 방식, 200개)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_extended_keys(tmp_path):
    log = RawLog(tmp_path); run_with_collectors(build_synthetic(), Naive(), [0], tier="extended", logs=[log], rho=5, beta=3)
    sa = log.records("stream_acc"); assert sa[0]["step"] == 0 and len(sa[-1]["accs"]) == 5
    assert len(log.records("lca_curve")) == 5 * 4
    assert log.tensors("features")["t4.task0"].shape == (100, 16)   # test 100개, MLP hidden 16
    assert "t0.layer0" in log.tensors("activations")

def test_full_keys(tmp_path):
    log = RawLog(tmp_path); run_with_collectors(build_synthetic(n_tasks=3), Naive(), [0], tier="full", logs=[log])
    assert len(log.records("barrier_curve")) == 2 and len(log.records("barrier_curve")[0]["losses"]) == 11
    m = log.records("grad_cosine")[-1]["matrix"]; assert len(m) == 3 and abs(m[0][0] - 1) < 1e-5
    assert len(log.records("recovery")) == 2

def test_general_ppl_for_text(tmp_path):
    log = RawLog(tmp_path); run_with_collectors(build_synthetic_text(), Naive(), [0], tier="extended", logs=[log], model="tiny")
    assert [r["task"] for r in log.records("general_ppl")] == [-1, 0, 1, 2]
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_collectors_ext.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: extended/full-tier raw data collectors"`

## Phase E — 지표와 통계

### Task 16: 지표 레지스트리, 정확도 행렬·효율 지표

**Files:**
- Create: `contilab/metrics/registry.py`, `contilab/metrics/accuracy.py`, `contilab/metrics/efficiency.py`, `tests/metrics/test_accuracy.py`

**Interfaces:**
- Consumes: `RawLog`
- Produces:
  - `MetricContext(scenario: str, class_table: Bool[T,C], joint_acc: list[float] | None, cfg: MetricsConfig)`
  - `@metric(name: str, requires: tuple[str, ...], tier: str)`; 함수 시그니처 `(log: RawLog, mctx: MetricContext) -> float | dict | None`; `MetricUnavailable(reason)` 예외로 `None`+사유
  - `compute_metrics(log, mctx, tier: str, only: list[str] | None = None) -> dict[str, dict]` — 각 값 `{"value": ..., "error": str | None}`; requires 원자료가 없으면 `error = "missing raw '<key>' (tier '<tier>')"`
  - 등록 지표(§6.3 정의 그대로): `acc`, `aia`, `bwt`, `fwt`, `forgetting`, `relative_forgetting`, `learning_acc`, `intransigence`; 효율 `wall_time`, `samples_per_s`, `peak_mem`, `trainable_params`, `buffer_bytes`, `model_bytes`, `total_flops`
  - `acc_matrix_of(log) -> np.ndarray[rows, T]`(Joint는 1행), `init_acc_of(log) -> np.ndarray[T]`; ACC는 항상 마지막 행 평균
  - 테스트 헬퍼(`tests/helpers.py`): `fake_log(tmp_path, R, B) -> RawLog`, `values(out) -> dict[str, value]`, `MCTX`(class, 3태스크), `MCTX1`(1태스크)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
R = [[0.9, 0.1, 0.0], [0.6, 0.8, 0.2], [0.5, 0.7, 0.9]]; B = [0.1, 0.1, 0.1]

def test_matrix_metrics(tmp_path):
    log = fake_log(tmp_path, R, B); v = values(compute_metrics(log, MCTX, "core"))
    assert v["acc"] == pytest.approx(0.7)
    assert v["aia"] == pytest.approx((0.9 + 0.7 + 0.7) / 3)
    assert v["bwt"] == pytest.approx(((0.5 - 0.9) + (0.7 - 0.8)) / 2)
    assert v["fwt"] == pytest.approx(((0.1 - 0.1) + (0.2 - 0.1)) / 2)
    assert v["forgetting"] == pytest.approx(((0.9 - 0.5) + (0.8 - 0.7)) / 2)
    assert v["relative_forgetting"] == pytest.approx((0.4 / 0.9 + 0.1 / 0.8) / 2)
    assert v["learning_acc"] == pytest.approx((0.9 + 0.8 + 0.9) / 3)

def test_intransigence_needs_joint(tmp_path):
    log = fake_log(tmp_path, R, B)
    assert compute_metrics(log, MCTX, "core")["intransigence"]["error"] is not None
    v = values(compute_metrics(log, replace(MCTX, joint_acc=[0.95, 0.9, 0.92]), "core"))
    assert v["intransigence"] == pytest.approx(((0.95 - 0.9) + (0.9 - 0.8) + (0.92 - 0.9)) / 3)

def test_single_task_metrics_none(tmp_path):
    out = compute_metrics(fake_log(tmp_path, [[0.8]], [0.1]), MCTX1, "core")
    assert out["acc"]["value"] == pytest.approx(0.8)
    for k in ("bwt", "fwt", "forgetting"): assert out[k]["value"] is None and out[k]["error"]
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_accuracy.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: metric registry, accuracy-matrix and efficiency metrics"`

### Task 17: 연속 평가·표현·가소성 지표

**Files:**
- Create: `contilab/metrics/continual.py`, `contilab/metrics/representation.py`, `contilab/metrics/plasticity.py`, `tests/metrics/test_continual_repr.py`

**Interfaces:**
- Produces (이름 = 등록 이름, 정의 §6.3):
  - continual(extended): `min_acc`, `wc_acc`, `aaa`, `stability_gap`, `windowed_forgetting`, `lca`
  - representation(extended): `probe_acc`(dict: 행렬 P), `repr_forgetting`, `head_forgetting`, `cka_final`(태스크별 리스트+평균), `cka_matrix`, `feature_drift`; 보조 함수 `linear_cka(X: ndarray, Y: ndarray) -> float`, `probe_accuracy(F: ndarray, y: ndarray, seed=0) -> float`(sklearn `LogisticRegression(max_iter=1000)`, 80/20)
  - plasticity(extended): `dormant_ratio`(태스크별 리스트+최종), `effective_rank`, `srank`, `weight_norms`(full; 체크포인트 사용); 보조 `effective_rank(F) -> float`, `srank(F, delta) -> int`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_stream_metrics(tmp_path):
    # 태스크 0 길이 10 스텝, 태스크 1 길이 10 스텝, ρ=5
    rows = [{"step": 0, "task_now": 0, "accs": [0.0]}, {"step": 5, "task_now": 0, "accs": [0.6]},
            {"step": 10, "task_now": 0, "accs": [0.9]}, {"step": 15, "task_now": 1, "accs": [0.4, 0.5]},
            {"step": 20, "task_now": 1, "accs": [0.7, 0.9]}]
    v = values(compute_metrics(fake_stream_log(tmp_path, rows, R2), MCTX2, "extended"))
    assert v["min_acc"] == pytest.approx(0.4)
    assert v["wc_acc"] == pytest.approx(0.5 * 0.9 + 0.5 * 0.4)
    assert v["stability_gap"] == pytest.approx(0.9 - 0.4)
    assert v["aaa"] == pytest.approx(np.mean([0.0, 0.6, 0.9, 0.45, 0.8]))

def test_cka_identity_and_rotation():
    X = np.random.default_rng(0).normal(size=(50, 8)); Q = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))[0]
    assert linear_cka(X, X) == pytest.approx(1.0) and linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)

def test_rank_measures():
    F = np.diag([1.0, 1.0, 0.0, 0.0]); assert effective_rank(F) == pytest.approx(2.0) and srank(F, 0.01) == 2

def test_dormant_ratio():
    # 뉴런 평균 |h| = [1, 1, 0, 0.02] → 점수 [1.98, 1.98, 0, 0.0396] → τ=0.025에서 1/4
    assert dormant_fraction(np.array([1, 1, 0, 0.02]), 0.025) == pytest.approx(0.25)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_continual_repr.py -v` → FAIL
- [ ] **Step 3: 구현** — `min_acc`: 태스크 k 종료 시점(해당 태스크의 마지막 레코드) **이후** 레코드의 최소값; `stability_gap`: 태스크 k 종료 시 정확도 − 태스크 k+1 학습 중 최소값, k 평균; WF: 윈도우 = 평균 태스크 길이×0.1 스텝
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: continual-evaluation, representation and plasticity metrics"`

### Task 18: 편향·신뢰도·복구성·지형·LLM 지표

**Files:**
- Create: `contilab/metrics/bias.py`, `contilab/metrics/recovery.py`, `contilab/metrics/landscape.py`, `contilab/metrics/llm.py`, `tests/metrics/test_misc_metrics.py`

**Interfaces:**
- Produces:
  - bias(core): `recency_bias`(Class-IL만; 이전 태스크 샘플 중 예측 클래스가 최근 태스크 클래스인 비율; 공유 클래스는 그 클래스를 포함하는 가장 최근 태스크로 귀속), `task_confusion`(dict 행렬), `ece`(전체 + 태스크별, 15 bin); 보조 `expected_calibration_error(conf, correct, bins=15) -> float`
  - recovery(full): `recovery`(§6.3 공식, 분모 ≤ 0.01 제외, 태스크별+평균)
  - landscape(full): `loss_barrier`(태스크별+평균), `grad_interference`(최종 행렬의 비대각 평균 + 행렬)
  - llm(extended): `general_ppl_delta`(최종 − θ₀ 및 태스크별 추이); stream.lm_eval 없으면 `MetricUnavailable`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_ece_perfect_and_overconfident():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0.9] * 10), np.array([1] * 5 + [0] * 5)) == pytest.approx(0.4)

def test_recency_bias(tmp_path):
    # 클래스 표 [[0,1],[2,3]], 최종 예측: 태스크0 샘플 4개 중 3개가 클래스 2/3으로 예측
    assert values(compute_metrics(fake_pred_log(tmp_path), MCTX_CLS, "core"))["recency_bias"] == pytest.approx(0.75)

def test_loss_barrier(tmp_path):
    v = values(compute_metrics(fake_barrier_log(tmp_path, alphas=[0, .5, 1], losses=[1.0, 3.0, 2.0]), MCTX, "full"))
    assert v["loss_barrier"] == pytest.approx(3.0 - 1.5)

def test_recovery_formula(tmp_path):
    rows = [{"task": 0, "acc_before": 0.4, "acc_after": 0.8, "acc_at_learn": 0.9}]
    assert values(compute_metrics(fake_recovery_log(tmp_path, rows), MCTX, "full"))["recovery"] == pytest.approx(0.8)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_misc_metrics.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: bias/calibration, recovery, landscape and LLM retention metrics"`

### Task 19: 통계

**Files:**
- Create: `contilab/metrics/stats.py`, `tests/metrics/test_stats.py`

**Interfaces:**
- Produces:
  - `summarize(xs: Sequence[float], n_boot=10000, seed=0) -> dict` (`mean, std(ddof=1, n=1이면 0), ci_low, ci_high, n`)
  - `paired_test(a: Sequence[float], b: Sequence[float]) -> dict` (`method` = "wilcoxon"(n ≥ 6, `scipy.stats.wilcoxon`) | "permutation"(n < 6, 부호 뒤집기 2ⁿ 전수, 양측), `statistic`, `p`)
  - `friedman_nemenyi(scores: ndarray[N_bench, k], higher_is_better=True, alpha=0.05) -> dict` (`avg_ranks`, `friedman_p`(N<2 또는 k<3이면 None), `cd` = q_α·√(k(k+1)/(6N)), q_α = `scipy.stats.studentized_range.ppf(1−α, k, inf)/√2`)
  - `low_power_warning(n: int) -> str | None` (n < 5이면 경고 문구)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_nemenyi_q_matches_demsar_table():
    assert nemenyi_q(3) == pytest.approx(2.343, abs=1e-3) and nemenyi_q(10) == pytest.approx(3.164, abs=1e-3)

def test_permutation_exact_small_n():
    r = paired_test([1, 2, 3], [0, 0, 0]); assert r["method"] == "permutation" and r["p"] == pytest.approx(0.25)

def test_friedman_ranks():
    out = friedman_nemenyi(np.array([[0.9, 0.5, 0.1], [0.8, 0.6, 0.2], [0.7, 0.4, 0.3]]))
    assert out["avg_ranks"].tolist() == [1.0, 2.0, 3.0] and out["friedman_p"] < 0.1

def test_summarize_single():
    s = summarize([0.5]); assert s["std"] == 0 and s["n"] == 1
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/metrics/test_stats.py -v` → FAIL
- [ ] **Step 3: 구현** (`nemenyi_q(k)` 공개 함수 포함)
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: statistics (bootstrap CI, paired tests, Friedman/Nemenyi CD)"`

## Phase F — 증류·사후 통합

### Task 20: 태스크 벡터 병합(TA/TIES), 헤드 조립, λ 선택

**Files:**
- Create: `contilab/consolidate/__init__.py`, `contilab/consolidate/merge.py`, `contilab/consolidate/heads.py`, `tests/consolidate/test_merge.py`

**Interfaces:**
- Produces:
  - `task_vector(base: dict, tuned: dict, names: list[str]) -> dict` (`tuned − base`, E축 유지)
  - `merge_ta(tvs: list[dict]) -> dict` (합)
  - `merge_ties(tvs: list[dict], k: float = 0.2) -> dict` — (1) 각 태스크 벡터에서 **멤버별·전 파라미터 평탄화 기준** 상위 k 비율 크기만 남김 (2) 합의 부호 = sign(Σ trimmed) (3) 합의 부호와 같은 부호의 비영 값들의 평균, 없으면 0
  - `apply_tv(base: dict, tv: dict, lam: Tensor[E] | float) -> dict`
  - `select_lambda(build: Callable[[float], dict], score: Callable[[dict], Tensor[E]], grid: Sequence[float]) -> tuple[Tensor[E], dict]` — 멤버별 최대 점수 λ(동률은 작은 λ), 결과 파라미터는 멤버별로 해당 λ의 것을 조립
  - `assemble_head(experts: list[dict], class_table: Bool[T,C], upto: int, scenario: str, prefix="head.") -> dict` — task/class: 클래스 c의 행 = c를 포함하는 태스크 중 가장 최근(≤upto) 전문가의 행; domain: 전문가 헤드 평균
  - `body_names(params: dict, prefix="head.") -> list[str]`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_single_expert_ta_is_expert():
    base = {"w": torch.zeros(1, 4)}; e = {"w": torch.tensor([[1., -2., 3., 0.]])}
    assert torch.equal(apply_tv(base, merge_ta([task_vector(base, e, ["w"])]), 1.0)["w"], e["w"])

def test_ties_full_k_single_expert_equals_ta():
    base = {"w": torch.zeros(1, 4)}; tv = [task_vector(base, {"w": torch.tensor([[1., -2., 3., 0.5]])}, ["w"])]
    assert torch.equal(merge_ties(tv, k=1.0)["w"], merge_ta(tv)["w"])

def test_ties_sign_election_and_disjoint_mean():
    a = {"w": torch.tensor([[2., 1., -1.]])}; b = {"w": torch.tensor([[4., -3., -1.]])}
    assert merge_ties([a, b], k=1.0)["w"].tolist() == [[3., -3., -1.]]

def test_ties_trim_per_member():
    t = {"w": torch.tensor([[1., 5., 2., 4., 3.]])}
    assert merge_ties([t], k=0.4)["w"].tolist() == [[0., 5., 0., 4., 0.]]

def test_select_lambda_per_member():
    score = lambda p: -(p["w"][:, 0] - torch.tensor([0.3, 0.8])).abs()
    lam, p = select_lambda(lambda l: {"w": torch.full((2, 1), l)}, score, [0.1 * i for i in range(1, 11)])
    assert torch.allclose(lam, torch.tensor([0.3, 0.8])) and torch.allclose(p["w"][:, 0], lam)

def test_assemble_head_rows():
    e0 = {"head.weight": torch.zeros(1, 4, 2)}; e1 = {"head.weight": torch.ones(1, 4, 2)}
    tbl = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    h = assemble_head([e0, e1], tbl, 1, "class")["head.weight"][0]
    assert h[:2].sum() == 0 and h[2:].sum() == 4
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/consolidate/test_merge.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: task arithmetic, TIES, head assembly, per-member lambda selection"`

### Task 21: 전문가 학습·캐시, TA/TIES Consolidator, 통합 실행 루프

**Files:**
- Create: `contilab/consolidate/experts.py`, `contilab/consolidate/consolidator.py`, `tests/consolidate/test_consolidator.py`

**Interfaces:**
- Consumes: `Ensemble`, `train_stream`, `Naive`, `inject_lora`, `lora_task_vector`, Task 20 함수, `Collectors`, `evaluate_split`
- Produces:
  - `Experts(base: dict[str, Tensor[S,...]], tuned: list[dict[str, Tensor[S,...]]], lora_modules: list[str] | None)`
  - `train_experts(stream, run_cfg, seeds: list[int]) -> Experts` — 모든 멤버의 초기값은 `init_seed`(기본 0) 공통(§5.4), 데이터 순서는 멤버별 seed. **학습 크기가 같은 태스크끼리 E = 태스크×시드로 묶어** 단일 태스크 스트림처럼 vmap 학습(멤버마다 자기 태스크 데이터 — `TrainPool.gather`의 전역 인덱스로 처리), 크기가 다른 태스크는 별도 그룹. LoRA 전문가(`consolidate.experts_lora=True`)는 어댑터 0 하나만 주입·학습 + 헤드 학습. 결과는 `cache_dir()/experts/<sha(benchmark, model, optim, train, seeds, init_seed, experts_lora)>/`에 safetensors로 캐시, 존재 시 로드
  - `ConsolidationContext(stream, ens: Ensemble, val_score: Callable[[dict], Tensor[S]])` — `val_score`는 본 태스크 val 평균 정확도
  - `Consolidator(cfg: ConsolidateConfig, student_factory=None)`: `on_new_expert(expert, ctx)`(no-op), `consolidate(experts, upto, cctx) -> dict[str, Tensor[S,...]]`
  - `TAConsolidator`, `TIESConsolidator` — 본체는 TA/TIES(LoRA면 ΔW 태스크 벡터를 base 가중치에 적용), 헤드는 `assemble_head`, λ는 `select_lambda`(그리드 `cfg.lam_grid`)
  - `build_consolidator(method_name: str, cfg) -> Consolidator` (`consolidate:ta|ties|distill`)
  - 테스트 헬퍼: `smoke_cfg(**overrides) -> RunConfig`(syn 프리셋 상당 값), `run_consolidation_for_test(stream, method, seeds, log, experts_lora=False, transfer="buffer") -> None`
  - `run_consolidation(stream, run_cfg, seeds, hooks: Collectors, ctx) -> None` — `train_experts` → `hooks.on_run_start` → 각 t: `hooks.on_task_start` → `ens.load_params(consolidate(experts, t, cctx))` → `hooks.on_task_end`; `hooks.on_run_end`. 비용 레코드에 전문가 학습 시간·FLOPs 합산과 `stored_experts_bytes` 기록(§5.5)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_experts_share_init_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path)); s = build_synthetic(n_tasks=3)
    ex = train_experts(s, smoke_cfg(), [0, 1])
    assert torch.equal(ex.base["head.weight"][0], ex.base["head.weight"][1]) and len(ex.tuned) == 3
    t0 = time.perf_counter(); train_experts(s, smoke_cfg(), [0, 1]); assert time.perf_counter() - t0 < 1.0

def test_ta_consolidation_task_il_reasonable(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path)); s = build_synthetic(n_tasks=3, scenario="task")
    log = RawLog(tmp_path / "r"); run_consolidation_for_test(s, "consolidate:ta", [0], log)
    assert len(log.records("acc_matrix")) == 3 and np.mean(log.records("acc_matrix")[-1]["accs"]) > 0.6

def test_lora_experts_merge_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path)); s = build_synthetic(n_tasks=2, scenario="task")
    log = RawLog(tmp_path / "r"); run_consolidation_for_test(s, "consolidate:ties", [0], log, experts_lora=True)
    assert len(log.records("acc_matrix")) == 2
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/consolidate/test_consolidator.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: expert training/cache and TA/TIES consolidation loop"`

### Task 22: Multi-teacher 증류와 전이 데이터

**Files:**
- Create: `contilab/consolidate/transfer.py`, `contilab/consolidate/distill.py`, `tests/consolidate/test_distill.py`

**Interfaces:**
- Consumes: `Experts`, `TrainPool`, `build_proxy`, `Ensemble`, `TrainStep` 구성요소, Task 20/21
- Produces:
  - `TransferSet(gidx: Long[S,N] | None, proxy: TensorSplit | None)`; `build_transfer(stream, pool, upto, kind: "full"|"buffer"|"proxy", M: int, gens) -> TransferSet` — full: 태스크 ≤ upto 학습 인덱스 전부(멤버 공통), buffer: 태스크당 ⌊M/(upto+1)⌋개를 멤버별 gen으로 비복원 추출, proxy: `build_proxy(stream.name, M)`
  - `teacher_targets(experts, upto, transfer, routing: "routed"|"entropy", T: float, scenario, class_table) -> Tensor[S,N,C]` — teacher i의 분포는 자기 태스크 클래스 위에서만 정의(그 외 0); routed: 샘플의 소속 태스크 teacher(proxy에는 task가 없으므로 entropy로 대체); entropy: w_i = softmax_i(−H_i), 혼합 p = Σ w_i p_i; domain: 클래스 공유이므로 teacher 분포를 그대로 사용. **1회 계산 후 메모리 캐시**
  - `DistillConsolidator(cfg)`: 학생 초기값 `cfg.init` ∈ {base, ta, ties}; 손실 `T²·KL(p ‖ softmax(z_s/T))`(마스크 = 본 클래스, task-IL은 샘플 태스크 클래스); `cfg.epochs` 에폭, 배치·옵티마이저는 run 설정 공유; `consolidate`는 학습된 학생 파라미터 반환

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_buffer_transfer_budget():
    s = build_synthetic(); pool = TrainPool.from_stream(s)
    t = build_transfer(s, pool, upto=3, kind="buffer", M=100, gens=[Generator(0), Generator(1)])
    assert t.gidx.shape == (2, 100) and set(pool.task_id[t.gidx[0]].tolist()) == {0, 1, 2, 3}

def test_entropy_routing_prefers_confident_teacher():
    p = mix_teachers(torch.tensor([[[5., 0., 0., 0.]], [[0., 0., 0.1, 0.]]]), TABLE_2x2, routing="entropy", T=1.0)
    assert p[0, 0].argmax() == 0 and torch.allclose(p.sum(-1), torch.ones(1, 1))

def test_distill_student_learns_from_teachers(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTILAB_CACHE", str(tmp_path)); s = build_synthetic(n_tasks=2, scenario="task")
    log = RawLog(tmp_path / "r"); run_consolidation_for_test(s, "consolidate:distill", [0], log, transfer="full")
    assert np.mean(log.records("acc_matrix")[-1]["accs"]) > 0.8
```
(`mix_teachers(logits: Tensor[n_teachers, N, C], class_table, routing, T, task_of=None) -> Tensor[N, C]`는 `teacher_targets`의 내부 공개 함수)

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/consolidate/test_distill.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: multi-teacher distillation with cached teacher targets"`

## Phase G — 러너·리포트·CLI

### Task 23: 설정, 프로파일, 매트릭스 전개

**Files:**
- Create: `contilab/runner/__init__.py`, `contilab/runner/config.py`, `configs/profiles/smoke.yaml`, `configs/profiles/cpu.yaml`, `configs/profiles/gpu.yaml`, `configs/experiments/smoke_all.yaml`, `configs/experiments/cpu_main.yaml`, `tests/runner/test_config.py`

**Interfaces:**
- Produces:
  - Consumes: `contilab/core/config.py` dataclass(Task 1)
  - 프로파일 YAML 구조: `{defaults: {...RunConfig 부분}, presets: {<preset>: {...}}}`
  - 병합 순서: dataclass 기본 ← profile.defaults ← preset ← experiment.overrides ← matrix 스칼라 축(점 경로) ← sweep ← CLI `key=val`(값은 `yaml.safe_load`로 파싱)
  - `load_experiment(path: Path, cli_overrides: list[str]) -> ExperimentConfig`; `expand(exp) -> list[RunConfig]`(matrix의 `preset` 축은 프리셋 이름 리스트, 나머지 축은 점 경로); `run_id(cfg) -> str` = `f"{preset}-{method.replace(':','_')}-s{seed}-{sha1(canonical json)[:8]}"`; `group_key(cfg) -> str`(seed 제외 해시)
  - 알 수 없는 키 → `ValueError("unknown config key: <path>")`
  - 프로파일 내용(§4.1, §5.3): smoke 프리셋 `syn_class`, `syn_task`, `syn_text`(tiny-gptneox, AdamW lr 1e-3), 티어 full, compile off; cpu 프리셋 `split_mnist_class`, `split_mnist_task`, `permuted_mnist`(n_train 10000, n_test 2000), `split_cifar10_class`(epochs 3, resnet18r), `text5_class`(pythia-70m, n_train 2000, n_test 500, AdamW 1e-4, batch 16), 티어 extended; gpu 프리셋은 cpu와 같은 이름 + `split_cifar100_class`, text5 n_train None, `Qwen/Qwen2.5-1.5B`
  - `smoke_all.yaml`: preset [syn_class, syn_task, syn_text] × method 11종(naive, joint, ewc, lwf, er, derpp, agem, lora_per_task, consolidate:ta, consolidate:ties, consolidate:distill) × seed [0, 1]
  - `cpu_main.yaml`: preset 5종 × method 11종 × seed [0, 1, 2], workers 2, threads_per_worker 2. full 티어는 split_mnist_class에만 적용하므로 cpu 프로파일의 `split_mnist_class` 프리셋 안에 `metrics.tier: full`을 둠

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_merge_order_and_expand(tmp_path):
    exp = load_experiment(Path("configs/experiments/smoke_all.yaml"), ["train.epochs=2"])
    runs = expand(exp); assert len(runs) == 3 * 11 * 2
    r = next(r for r in runs if r.preset == "syn_text" and r.method.name == "ewc")
    assert r.train.epochs == 2 and r.model.hf_name == "tiny-gptneox" and r.metrics.tier == "full"

def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config key: train.epoch"):
        load_experiment(Path("configs/experiments/smoke_all.yaml"), ["train.epoch=2"])

def test_run_id_stable_and_seed_sensitive():
    a, b = expand(load_experiment(SMOKE, []))[:2]
    assert run_id(a) == run_id(copy.deepcopy(a)) and run_id(a) != run_id(b)
    assert group_key(a) == group_key(replace(a, seed=99))
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/runner/test_config.py -v` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: config dataclasses, profiles, experiment matrix expansion"`

### Task 24: 실행·저장·스케줄러·재개·사후 지표 계산

**Files:**
- Create: `contilab/runner/storage.py`, `contilab/runner/execute.py`, `contilab/runner/scheduler.py`, `contilab/runner/finalize.py`, `tests/runner/test_execute.py`

**Interfaces:**
- Consumes: 전 단계 전부
- Produces:
  - `RunDir(path)`: `status() -> str`, `set_status(s, error=None)`(`status.json`에 traceback 포함), `write_config(cfg)`, `write_env(dict)`, `ckpt_path(t)`, `last_ckpt() -> int | None`, `log -> RawLog`
  - `env_info(engine_decisions: dict) -> dict`(torch/python 버전, 디바이스, CPU capability, 스레드 수, compile·precision 결정과 사유, `git rev-parse HEAD`)
  - `execute_group(cfgs: list[RunConfig], exp_dir: Path) -> None` — 한 Ensemble(E=len)로 실행; Joint는 `make_joint_stream`으로 학습하고 `Collectors(eval_stream=원 스트림)`으로 평가(acc_matrix 1행 × T열 → 이것이 `joint_acc`); `consolidate:*`는 `run_consolidation`; 각 태스크 종료 시 멤버별 체크포인트(파라미터 + `strategy.state_dict()` + 멤버 gen 상태); 시작 시 모든 멤버의 `last_ckpt`가 같은 t면 로드 후 `RawLog.truncate_after_task(t)`하고 `start_task=t+1`로 재개; `strategy.vmappable=False`이거나 vmap 단계에서 예외가 나면 E=1로 쪼개 순차 재실행하고 `env.json`에 사유; 예외는 그룹 멤버 전부 `failed`; 발산 멤버는 `diverged`
  - `group_runs(cfgs, max_E) -> list[list[RunConfig]]` — `group_key`로 묶고 max_E 단위로 분할(시드 오름차순)
  - `run_experiment(exp: ExperimentConfig, root: Path) -> Path` — `root/<exp.name>/experiment.yaml` 저장, `done`인 run 제외, 그룹을 `ProcessPoolExecutor(workers, max_tasks_per_child=1)`로 실행(워커 초기화에서 `torch.set_num_threads(threads_per_worker or cores//workers)`). text 모달리티 그룹은 나머지 그룹이 끝난 뒤 workers=1, threads=코어 수로 실행(§7.4 텍스트 1×4). 끝나면 `finalize_experiment` 호출
  - `finalize_experiment(exp_dir, add: list[str] | None = None, tier: str | None = None) -> Path` — 각 run에 `compute_metrics`(Joint 결과를 같은 preset·seed에서 찾아 `joint_acc` 제공) → `metrics.json`, 전체를 `results.csv`(열: run_id, preset, method, seed, status, 각 지표 값) 로 저장

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_group_runs_split():
    cfgs = [replace(BASE, seed=s) for s in range(7)]
    assert [len(g) for g in group_runs(cfgs, max_E=4)] == [4, 3]

def test_failure_isolated(tmp_path, monkeypatch):
    exp = tiny_experiment(methods=["naive", "boom"])   # "boom": before_task에서 예외를 던지는 테스트 전략
    out = run_experiment(exp, tmp_path); st = statuses(out)
    assert st["naive"] == {"done"} and st["boom"] == {"failed"}
    assert "Traceback" in read_status(out, "boom")["error"]

def test_resume_matches_uninterrupted(tmp_path):
    full = run_experiment(tiny_experiment(methods=["er"]), tmp_path / "a")
    crash_after_task(1)                               # 태스크 1 체크포인트 직후 예외 주입(monkeypatch)
    part = run_experiment(tiny_experiment(methods=["er"]), tmp_path / "b")
    clear_crash(); resumed = run_experiment(tiny_experiment(methods=["er"]), tmp_path / "b")
    assert acc_matrix(full) == acc_matrix(resumed)

def test_done_runs_skipped(tmp_path):
    out = run_experiment(tiny_experiment(methods=["naive"]), tmp_path)
    m = mtimes(out); run_experiment(tiny_experiment(methods=["naive"]), tmp_path); assert mtimes(out) == m

def test_results_csv_has_intransigence(tmp_path):
    out = run_experiment(tiny_experiment(methods=["naive", "joint"]), tmp_path)
    df = read_csv(out / "results.csv"); assert df.loc[df.method == "naive", "intransigence"].notna().all()
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/runner/test_execute.py -v` → FAIL
- [ ] **Step 3: 구현** (테스트 헬퍼는 `tests/helpers.py`에: `tiny_experiment(methods, tier="core") -> ExperimentConfig`(syn_class, 태스크 3, seed [0,1], workers 1), `statuses(exp_dir) -> dict[method, set[str]]`, `read_status`, `acc_matrix(exp_dir) -> dict`, `mtimes`, `read_csv = pandas.read_csv`, `crash_after_task(t)`/`clear_crash()`(환경변수 `CONTILAB_TEST_CRASH_AFTER`를 `execute_group`이 체크포인트 직후 확인), 실패 전략 `boom`은 테스트에서 `register_strategy`로 등록)
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: run execution, checkpoint resume, scheduler, post-hoc metric finalization"`

### Task 25: 리포트

**Files:**
- Create: `contilab/report/__init__.py`, `contilab/report/tables.py`, `contilab/report/figures.py`, `contilab/report/render.py`, `tests/report/test_report.py`

**Interfaces:**
- Consumes: `results.csv`, 각 run `RawLog`, `summarize`, `paired_test`, `friedman_nemenyi`, `low_power_warning`
- Produces:
  - `main_table(df, preset) -> str`(마크다운; 메서드 × {acc, aia, bwt, fwt, forgetting, learning_acc, min_acc, wc_acc, wall_time, peak_mem}, `평균 ± 표준편차`, 기준(`exp.baseline`) 대비 p<0.05면 `*`)
  - `axis_tables(df, preset) -> dict[str, str]`(§7.6 축: stability, plasticity, representation, plasticity_loss, bias_calibration, recovery, efficiency — 해당 지표가 하나도 없으면 생략)
  - `rank_table(df, metric="acc") -> tuple[str, dict]`(프리셋 간 평균 순위, Friedman p, CD)
  - figures(`figs/<preset>/<name>.png`, 대표 seed 0): `acc_heatmap`, `stream_curves`(stability gap 구간 음영), `task_confusion`, `cka_heatmap`, `weight_drift`, `grad_interference`, `feature_pca`, `feature_tsne`, `loss_barrier`, `cd_diagram`, `bench_bars`(`bench.json` 있을 때) — 원자료가 없으면 해당 그림 생략
  - `build_report(exp_dir) -> tuple[Path, Path]` — `report.md`(표 + 그림 링크 + 환경 요약 + 한계: 시드 수 경고, 서브샘플 규모, Class-IL 병합 헤드 스케일 문제) 와 `report.html`(같은 내용, PNG base64 임베드, 단일 파일)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_report_from_tiny_experiment(tmp_path):
    out = run_experiment(tiny_experiment(methods=["naive", "er", "joint"], tier="full"), tmp_path)
    md, html = build_report(out); text = md.read_text()
    assert "| naive |" in text and "±" in text and "acc_heatmap.png" in text
    assert (out / "figs" / "syn_class" / "acc_heatmap.png").exists()
    assert "data:image/png;base64," in html.read_text()
    assert "통계 검정력" in text            # seed 2개 → 경고 문구

def test_main_table_marks_significance():
    df = fake_results(naive=[0.1] * 6, er=[0.5] * 6)
    assert "*" in main_table(df, "p")
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/report -v` → FAIL
- [ ] **Step 3: 구현** (matplotlib `Agg` 백엔드; 경고 문구 상수 `"통계 검정력 경고: 시드 수 {n}개 (< 5)"`)
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat: markdown/html report with tables and figures"`

### Task 26: CLI, bench, smoke 종단 테스트

**Files:**
- Create: `contilab/runner/cli.py`, `contilab/runner/bench.py`, `tests/test_smoke_e2e.py`, `README.md`

**Interfaces:**
- Consumes: `run_experiment`, `finalize_experiment`, `build_report`, `build_stream`, `Ensemble`, `TrainStep`
- Produces:
  - `main(argv: list[str] | None = None) -> int` — 하위 명령(§7.1): `run <cfg> [k=v...] [--root runs]`, `report <exp_dir>`, `metrics <exp_dir> --add <name>... [--tier T]`, `bench [--model mlp|resnet18r|hf] [--E 1,2,4,8] [--steps 50] [--compile off,on] [--out runs/bench]`, `data prepare <benchmark>`
  - `run_bench(model: str, E_list: list[int], steps: int, compile_modes: list[str], out: Path) -> list[dict]` — 각 (compile, E) 조합에서 워밍업 5스텝 후 `steps` 스텝 측정, 레코드 `{model, E, compile, compile_used, samples_per_s, step_ms, speedup_vs_E1_eager}`, 추가로 기준선 "E개 순차 실행" 처리량 대비 `vmap_speedup`; `bench.json`과 표 출력
  - README: 설치, smoke 실행, cpu 실행, 리포트, 지표 사후 추가, bench 사용법(각 1줄 명령)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
def test_smoke_all_under_60s(tmp_path):
    t0 = time.perf_counter()
    assert main(["run", "configs/experiments/smoke_all.yaml", "--root", str(tmp_path)]) == 0
    assert time.perf_counter() - t0 < 60
    df = read_csv(tmp_path / "smoke_all" / "results.csv")
    assert len(df) == 66 and set(df.status) == {"done"}
    assert main(["report", str(tmp_path / "smoke_all")]) == 0

def test_metrics_add_posthoc(tmp_path):
    main(["run", "configs/experiments/smoke_all.yaml", "--root", str(tmp_path), "matrix.method.name=[naive]"])
    assert main(["metrics", str(tmp_path / "smoke_all"), "--add", "cka_final"]) == 0

def test_bench_reports_speedup(tmp_path):
    rows = run_bench("mlp", [1, 4], steps=10, compile_modes=["off"], out=tmp_path)
    assert {r["E"] for r in rows} == {1, 4} and all(r["samples_per_s"] > 0 for r in rows)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/pytest tests/test_smoke_e2e.py -v` → FAIL
- [ ] **Step 3: 구현** — `matrix.<axis>=[...]` 형식 CLI 오버라이드로 매트릭스 축 교체 지원
- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/pytest -v` (전체) → PASS, smoke 60초 이내. 초과 시 smoke 프로파일 규모(n_train, epochs)를 줄여 맞추고 그 값을 커밋 메시지에 기록
- [ ] **Step 5: 커밋** — `git commit -m "feat: CLI, throughput bench, smoke end-to-end test, README"`

## Phase H — 실증

### Task 27: CPU 실증 실행과 결과 리포트

**Files:**
- Create: `docs/results/2026-10-cpu-main.md` (리포트 요약 + 성공 기준 판정)
- Modify: `configs/experiments/cpu_main.yaml` (시간 예산에 맞춘 조정이 필요할 때만)

**Interfaces:**
- Consumes: CLI 전체

- [ ] **Step 1: 데이터 준비** — Run: `for b in split_mnist permuted_mnist split_cifar10 text5; do .venv/bin/contilab data prepare $b; done` → 각 캐시 생성 메시지
- [ ] **Step 2: bench** — Run: `.venv/bin/contilab bench --model mlp --E 1,2,4,8 --compile off,on --out runs/bench && .venv/bin/contilab bench --model resnet18r --E 1,2,4 --compile off,on --out runs/bench` → `bench.json`; vmap E>1의 `vmap_speedup` > 1 확인(성공 기준 4). 1 이하이면 수치를 그대로 기록하고 원인(코어 수 대비 스레드 경합 등)을 결과 문서에 적음
- [ ] **Step 3: 시간 예산 추정** — `cpu_main.yaml`을 `matrix.seed=[0] matrix.method.name=[naive]`로 1회 실행해 프리셋별 소요 시간 측정 → 전체 예상 시간 = 프리셋별 시간 × 메서드 11 × ⌈시드/max_E⌉ / workers. 6시간 초과 시 text5 `n_train`을 1000으로, split_cifar10 epochs를 2로 낮추고 조정 내역을 결과 문서에 기록
- [ ] **Step 4: 본 실행** — Run (백그라운드): `.venv/bin/contilab run configs/experiments/cpu_main.yaml` → 모든 run `done` 또는 `diverged`(실패 run이 있으면 traceback 확인 후 수정·재실행 — 완료 run은 자동 건너뜀)
- [ ] **Step 5: 리포트** — Run: `.venv/bin/contilab report runs/cpu_main` → `report.md`, `report.html`
- [ ] **Step 6: 성공 기준 판정** — `docs/results/2026-10-cpu-main.md`에 §1.5의 5개 기준 각각 충족/미충족과 근거 수치(특히 split_mnist_class ACC 순서 Naive < {EWC, LwF} < {ER, DER++} < Joint), bench 수치, 조정 내역, 한계 기록
- [ ] **Step 7: 커밋** — `git add docs/results configs && git commit -m "docs: CPU empirical results"` (`runs/`는 커밋하지 않음)

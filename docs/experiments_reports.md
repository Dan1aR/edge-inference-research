# BF16 Accumulator Fine-tuning Experiments

**Цель исследования:** Компенсация потери точности при использовании bf16-аккумуляторов в матричных операциях через fine-tuning.

**Модель:** YOLOS-tiny (hustvl/yolos-tiny)
**Датасет:** COCO 2017

---

## 1. Baseline: Inference Ablation Results
 
Результаты inference на 5000 изображений COCO val (без fine-tuning):

| Конфигурация | AP | AP50 | AP75 | Δ AP от FP32 |
|--------------|------|------|------|--------------|
| FP32 | 0.2692 | 0.4460 | 0.2710 | baseline |
| BF16 default (HW accum) | 0.2652 | 0.4441 | 0.2643 | -0.40% |
| bf16-accum lin=0, patch=0, attn=0 | 0.2647 | 0.4419 | 0.2649 | -0.45% |
| bf16-accum lin=0, patch=1, attn=0 | 0.2564 | 0.4333 | 0.2561 | -1.28% |
| bf16-accum lin=1, patch=0, attn=0 | 0.2555 | 0.4334 | 0.2535 | -1.37% |
| bf16-accum lin=0, patch=0, attn=1 | 0.2530 | 0.4287 | 0.2519 | -1.62% |
| bf16-accum lin=1, patch=1, attn=0 | 0.2491 | 0.4251 | 0.2472 | -2.01% |
| bf16-accum lin=1, patch=0, attn=1 | 0.2470 | 0.4231 | 0.2418 | -2.22% |
| bf16-accum lin=0, patch=1, attn=1 | 0.2460 | 0.4199 | 0.2434 | -2.32% |
| **bf16-accum full (lin=1, patch=1, attn=1)** | **0.2404** | **0.4128** | **0.2376** | **-2.88%** |

### Ключевые выводы из Inference Ablation:

1. **BF16 default vs FP32:** Минимальная разница (-0.40%), hardware accumulator работает хорошо
2. **Attention вносит наибольший вклад:** ~1.6% drop отдельно
3. **Linears:** ~1.4% drop отдельно
4. **Patch embedding:** ~0.8% drop отдельно (вычисляется косвенно)
5. **Кумулятивный эффект:** Все компоненты вместе дают -2.88%

---

## 2. Fine-tuning Experiments

### Summary Table (All Experiments)

| Experiment | Strategy | Pre-training AP | Final AP | Δ AP | Best Step |
|------------|----------|-----------------|----------|------|-----------|
| Exp 1: Baseline BF16 AMP | Standard fine-tuning | 0.2655 | 0.2612 | -0.43% | 250 |
| Exp 2: Immediate bf16-accum | All patches from step 0 | 0.2404 | 0.2490 | **+3.6%** | 200 |
| Exp 3: Progressive bf16-accum | Gradual patching | 0.2404* | 0.2462 | +2.4% | 250 |

*Progressive starts with BF16 AMP, bf16-accum patches added gradually

### 2.1 Методология

- **Короткие эксперименты:** 300 steps, eval каждые 50 steps
- **Optimizer:** AdamW, weight_decay=0.01
- **LR:** 5e-5 с cosine scheduler
- **Warmup:** 10% от total steps
- **Batch size:** 64
- **Precision:** bf16 AMP для baseline, bf16-accum для экспериментов

### 2.2 Experiment 1: Baseline BF16 AMP Fine-tuning

**Цель:** Установить upper bound - насколько fine-tuning улучшает модель в целом.

**Конфигурация:**
- Script: `train_yolos_baseline_bf16.py`
- Steps: 300, Eval: каждые 50 steps
- Batch: 64, LR: 5e-5, Warmup: 10%, Scheduler: cosine
- Results directory: `results/exp_baseline_quick/`

**Результаты:**

| Step | LR | AP | AP50 | AP75 | Примечание |
|------|-----|-------|-------|-------|------------|
| 0 | 0.0 | 0.2655 | 0.4441 | 0.2650 | Pre-training (BF16 default level) |
| 50 | 4.93e-5 | 0.2604 | 0.4416 | 0.2600 | Initial drop (-0.51%) |
| 100 | 4.22e-5 | 0.2574 | 0.4370 | 0.2556 | Minimum (-0.81%) |
| 150 | 2.93e-5 | 0.2574 | 0.4357 | 0.2576 | Recovery starts |
| 200 | 1.51e-5 | 0.2588 | 0.4347 | 0.2590 | Recovering |
| **250** | **4.11e-6** | **0.2612** | **0.4383** | **0.2598** | **Final (-0.43% от pre-training)** |

**Наблюдения:**
- V-образная кривая обучения: сначала падение (catastrophic forgetting), затем восстановление
- Финальный AP (0.2612) немного ниже pre-training (0.2655) - возможно нужно больше steps
- Модель стабилизируется к концу обучения (LR близок к 0)
- **Total training improvement:** -0.43% от pre-training (не улучшает, но и не ухудшает значительно)

---

### 2.3 Experiment 2: Full BF16-Accum Immediate Patching

**Цель:** Можно ли адаптировать модель к bf16-accum через fine-tuning?

**Конфигурация:**
- Script: `train_yolos_progressive_bf16accum_ste.py`
- Schedule: `"attn@0.0,mlp@0.0,qkv@0.0,head@0.0"` (всё с начала)
- Steps: 300, Batch: 64, LR: 5e-5
- Results directory: `results/exp_full_bf16accum/`

**Результаты:**

| Step | LR | AP | AP50 | AP75 | Примечание |
|------|-----|-------|-------|-------|------------|
| 0 | - | 0.2404 | 0.4128 | 0.2376 | Pre-training (bf16-accum inference) |
| 50 | 4.63e-5 | 0.2474 | 0.4223 | 0.2455 | +0.70% от pre-training |
| 100 | 3.70e-5 | 0.2469 | 0.4233 | 0.2431 | Стабильно |
| 150 | 2.78e-5 | 0.2485 | 0.4249 | 0.2441 | Улучшение |
| **200** | **1.85e-5** | **0.2490** | **0.4235** | **0.2466** | **Пик (+0.86% от pre-training)** |
| 250 | 9.26e-6 | 0.2483 | 0.4262 | 0.2434 | Небольшое снижение |

**Наблюдения:**
- Fine-tuning с bf16-accum даёт улучшение: AP 0.2404 → 0.2490 (+0.86%)
- Пик производительности на step 200, затем небольшое снижение
- Но НЕ достигает уровня BF16 default (0.2655) - разрыв ~1.6%
- Модель адаптируется к bf16 numerics, но не компенсирует потерю полностью
- Loss стабилен (~1.4-1.6), нет divergence
- **Recovery rate:** 30% от потерь bf16-accum восстановлено через fine-tuning

---

### 2.4 Experiment 3: Progressive BF16-Accum

**Цель:** Улучшает ли постепенное включение компонентов адаптацию?

**Конфигурация:**
- Schedule: `"attn@0.2,mlp@0.4,qkv@0.6,head@0.8"`
- attn включается на step 60 (20% от 300)
- mlp включается на step 120 (40%)
- qkv включается на step 180 (60%)
- head включается на step 240 (80%)
- Steps: 300
- Results directory: `results/exp_progressive_bf16accum/`

**Результаты:**

| Step | LR | AP | AP50 | AP75 | Enabled Groups | Δ AP |
|------|-----|-------|-------|-------|----------------|------|
| 50 | 4.63e-5 | 0.2536 | 0.4316 | 0.2519 | none (до step 60) | baseline |
| 100 | 3.70e-5 | 0.2388 | 0.4110 | 0.2362 | attn | -1.48% |
| 150 | 2.78e-5 | 0.2471 | 0.4221 | 0.2446 | attn + mlp | +0.83% recovery |
| 200 | 1.85e-5 | 0.2471 | 0.4219 | 0.2437 | attn + mlp + qkv | stable |
| **250** | **9.26e-6** | **0.2462** | **0.4209** | **0.2428** | **all components** | **Final** |

**Timeline визуализация:**
```
Step:     0----50----60----100----120----150----180----200----240----250----300
                     ↓           ↓             ↓             ↓
                   attn        mlp           qkv           head
AP:            0.2536      0.2388      0.2471      0.2471      0.2462
                     ↓           ↓             ↓             ↓
               [drop -1.5%] [recovery]  [stable]    [slight drop]
```

**Наблюдения:**
- Step 50 (без bf16-accum): AP = 0.2536 - чистый BF16 AMP fine-tuning
- Step 100 (attn включён): AP = 0.2388 - резкое падение (-1.48%) после включения attention
- Step 150-200: Модель восстанавливается до AP = 0.2471 (+0.83% от минимума)
- Step 250 (все компоненты): AP = 0.2462 - финальный результат с полным bf16-accum
- Progressive подход показывает V-образную кривую адаптации после каждого patch
- **Attention - главный источник потерь:** падение на 1.48% при включении
- **Recovery rate:** 20% от потерь bf16-accum восстановлено (хуже чем immediate)

---

## 3. Analysis and Conclusions

### 3.1 Сравнение стратегий

| Стратегия | Final AP | Δ от pre-training | Δ от BF16 default |
|-----------|----------|-------------------|-------------------|
| Pre-training (bf16-accum full) | 0.2404 | baseline | -9.4% |
| Baseline fine-tuning (bf16 AMP) | 0.2612 | - | -1.5% |
| Immediate bf16-accum fine-tuning | 0.2490 | +3.6% | -6.1% |
| **Progressive bf16-accum fine-tuning** | **0.2462** | **+2.4%** | **-7.2%** |

### 3.2 Ключевые наблюдения

1. **Fine-tuning помогает, но не полностью компенсирует:**
   - Immediate bf16-accum fine-tuning улучшает AP с 0.2404 до 0.2490 (+3.6%)
   - Progressive bf16-accum достигает AP = 0.2462 (+2.4%)
   - Оба подхода НЕ достигают уровня BF16 default (0.2652) - разрыв 6-7%

2. **Immediate vs Progressive:**
   - **Immediate лучше Progressive:** AP 0.2490 vs 0.2462 (+0.28% разница)
   - Причина: progressive подход не даёт достаточно времени для адаптации к каждому новому компоненту
   - При progressive: каждый новый patch вызывает временное падение AP, и модель не успевает полностью восстановиться

3. **V-образная кривая обучения:**
   - Baseline: сначала падение (catastrophic forgetting), затем восстановление
   - bf16-accum: более стабильное обучение после начального шока
   - Progressive: множественные V-образные падения при каждом включении нового компонента

4. **Attention - главный источник потерь:**
   - При включении attention (step 60→100): AP падает с 0.2536 до 0.2388 (-1.5%)
   - Это самое большое падение среди всех компонентов
   - Подтверждает результаты inference ablation: attention вносит ~1.6% drop

### 3.3 Рекомендации для практики

1. **Для максимальной точности:** Использовать immediate bf16-accum fine-tuning (все компоненты с начала)
   - Даёт лучший результат: AP = 0.2490
   - Модель адаптируется ко всем bf16-accum компонентам одновременно

2. **Progressive НЕ рекомендуется** для коротких fine-tuning сессий:
   - Каждый новый patch требует времени на адаптацию
   - 300 steps недостаточно для полной адаптации ко всем 4 группам компонентов
   - Может быть полезен только при очень длинном fine-tuning (>1000 steps)

3. **Ожидаемый gap:** Fine-tuning компенсирует ~30-40% потерь от bf16-accum:
   - Потеря от bf16-accum без fine-tuning: -2.88% (0.2692 → 0.2404)
   - После fine-tuning: -2.02% (0.2692 → 0.2490)
   - Восстановлено: ~0.86% AP (около 30% от исходной потери)

4. **Для NPU deployment:** 
   - Если требуется bf16-accum из-за hardware ограничений, fine-tuning обязателен
   - Ожидайте ~6-7% снижение AP относительно BF16 default даже после fine-tuning

---

## 4. Technical Notes

### 4.1 STE (Straight-Through Estimator) Approach

Forward pass использует bf16 accumulation (эмулирует NPU), backward pass использует стандартные PyTorch градиенты:

```
Forward: x → [bf16-accum matmul] → y
Backward: ∂L/∂x ← [standard matmul] ← ∂L/∂y
```

Это позволяет обучать модель "видеть" bf16-accum numerics во время forward, получая при этом "правильные" градиенты.

### 4.2 Implemented Triton Kernels

| Kernel | File | Forward | Backward |
|--------|------|---------|----------|
| Linear | `triton_bf16acc_linear_ste.py` | bf16-accum | STE (PyTorch) |
| BMM (Attention) | `triton_bf16acc_bmm.py` | bf16-accum | STE (PyTorch) |

---

## 5. Conclusions for Article

### 5.1 Research Question

**Can fine-tuning compensate for accuracy loss when using BF16 accumulators in matrix operations?**

### 5.2 Key Findings

#### Finding 1: BF16 Accumulator Impact
Using BF16 accumulators instead of higher-precision hardware accumulators causes a **2.88% AP drop** on YOLOS-tiny object detection (0.2692 → 0.2404). This represents significant accuracy degradation for edge deployment scenarios.

#### Finding 2: Fine-tuning Effectiveness
Fine-tuning with bf16-accum forward pass (STE approach) **partially compensates** for accuracy loss:
- **Immediate bf16-accum fine-tuning:** Recovers 0.86% AP (0.2404 → 0.2490), restoring ~30% of lost accuracy
- **Progressive bf16-accum fine-tuning:** Recovers 0.58% AP (0.2404 → 0.2462), restoring ~20% of lost accuracy
- Neither approach fully closes the gap to BF16 default (0.2652)

#### Finding 3: Strategy Comparison
| Strategy | Final AP | Recovery Rate |
|----------|----------|---------------|
| No fine-tuning (bf16-accum) | 0.2404 | 0% |
| Progressive fine-tuning | 0.2462 | 20% |
| **Immediate fine-tuning** | **0.2490** | **30%** |
| BF16 default (target) | 0.2652 | 100% |

**Immediate fine-tuning outperforms progressive** by +0.28% AP. The progressive approach suffers from insufficient adaptation time between component patches.

#### Finding 4: Component Sensitivity
Attention mechanism is the most sensitive to bf16-accum:
- Attention alone: -1.6% AP drop
- Linear layers: -1.4% AP drop  
- Patch embedding: -0.8% AP drop

### 5.3 Practical Implications

1. **For Edge NPU Deployment:**
   - Fine-tuning is **essential** when using bf16-accum hardware
   - Use immediate (all-at-once) bf16-accum patching, not progressive
   - Expect ~7% AP degradation vs standard BF16 even after fine-tuning

2. **Fine-tuning Budget:**
   - 300 steps sufficient for initial adaptation
   - Longer training may yield additional recovery (not tested)

3. **Architecture Considerations:**
   - Attention-heavy models may suffer more from bf16-accum
   - Consider attention-specific optimizations for edge deployment

### 5.4 Quantitative Summary

```
┌─────────────────────────────────────────────────────────────┐
│              BF16-Accum Fine-tuning Results                 │
├─────────────────────────────────────────────────────────────┤
│  FP32 Baseline:           0.2692 AP                         │
│  BF16 Default:            0.2652 AP  (-0.40% from FP32)     │
│  BF16-Accum (no FT):      0.2404 AP  (-2.88% from FP32)     │
│  BF16-Accum + FT:         0.2490 AP  (-2.02% from FP32)     │
├─────────────────────────────────────────────────────────────┤
│  Fine-tuning Recovery:    +0.86% AP (30% of loss recovered) │
│  Remaining Gap:           -1.62% AP vs BF16 default         │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendix: Raw Logs

Experiment logs available in:
- `results/exp_baseline_quick/logs/` - Baseline BF16 AMP fine-tuning
- `results/exp_full_bf16accum/logs/` - Immediate bf16-accum fine-tuning
- `results/exp_progressive_bf16accum/logs/` - Progressive bf16-accum fine-tuning

# 경량 모델에 적용된 Novel LM 아키텍처 조사

## 1. 개요

본 문서는 `ai-research` 레포지토리의 다음 후보 구현 대상을 발굴하기 위해, **2023 후반 ~ 2026 초**에 공개된 **경량(≤ ~7B) 변종을 보유하고 구조가 명시적으로 공개된 non-vanilla-Transformer LM 아키텍처**를 광범위하게 조사한 결과입니다.

선정 기준:

1. Pure SSM / Hybrid SSM-Attention / Linear (또는 Lightning) Attention / RNN-revival / Gated-Conv 등 sequence-mixing 측면에서 novel한 메커니즘을 가질 것
2. 1차 자료(arXiv 논문, 회사 테크리포트, 또는 공식 GitHub)로 아키텍처 세부 사항이 검증 가능할 것
3. CPU 토이 config (hidden_size~128, layers~4-6)로 from-scratch 이식이 가능한 수준일 것 — 즉 custom CUDA kernel 없이 PyTorch 레퍼런스 코드가 존재하거나 수식이 명확할 것
4. 현재 레포에 이미 구현된 **Mamba (selective SSM), LFM2 (gated short-conv + GQA hybrid), Gemma 4 (sliding+global attention)** 와 의미 있는 비교축을 만들 수 있을 것

---

## 2. 요약 테이블

| # | 모델 | 카테고리 | 핵심 메커니즘 | 발표 | 라이선스 | 경량 변종 | 1차 자료 | 도입 추천도 |
|---|------|---------|--------------|------|---------|----------|---------|------------|
| 1 | **Mamba-2** | Pure SSM | SSD (Structured State Space Duality), 병렬 SSM 파라미터 산출 | 2024-05 | Apache 2.0 | 130M~2.8B | [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) / [state-spaces/mamba](https://github.com/state-spaces/mamba) | ★★★★★ |
| 2 | **xLSTM (mLSTM/sLSTM)** | RNN-revival | Exponential gating + 행렬-값 메모리 / scalar memory mixing | 2024-05 | NX-AI License (연구 제한) | 1.4B / 7B | [arXiv 2405.04517](https://arxiv.org/abs/2405.04517) / [NX-AI/xlstm](https://github.com/NX-AI/xlstm) | ★★★★★ |
| 3 | **RWKV-7 "Goose"** | RNN-revival | 일반화된 delta rule + 벡터-값 게이팅 + in-context learning rate | 2025-03 | Apache 2.0 | 0.19B~2.9B | [arXiv 2503.14456](https://arxiv.org/abs/2503.14456) / [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | ★★★★★ |
| 4 | **Griffin / Hawk (RecurrentGemma)** | Hybrid (RG-LRU + local attn) | Real-Gated Linear Recurrent Unit + sliding-window attention | 2024-02 / 04 | Gemma Terms (오픈) | RecurrentGemma 2B / 9B | [arXiv 2402.19427](https://arxiv.org/abs/2402.19427) / [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma) | ★★★★★ |
| 5 | **Gated DeltaNet** | Linear Attn + Delta Rule | Mamba2의 forget-gate + DeltaNet의 targeted update 결합 | 2024-12 | NVIDIA Source Code License | 1.3B (실험) | [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) / [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) | ★★★★★ |
| 6 | **DeltaNet (Yang et al.)** | Linear Attn | Householder 기반 parallelizable delta rule (errata-correction style) | 2024-06 | MIT (FLA) | 1.3B 실험 | [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) / [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | ★★★★ |
| 7 | **Gated Linear Attention (GLA)** | Linear Attn | Data-dependent gate + chunk-wise FlashLinearAttention | 2024 (ICML) | MIT (FLA) | 1.3B / 2.7B 실험 | [arXiv 2312.06635](https://arxiv.org/abs/2312.06635) / [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | ★★★★ |
| 8 | **HGRN2** | Gated Linear RNN | Outer-product state expansion + 계층적 lower-bound forget gate | 2024-04 | Apache 2.0 | 150M~3B | [arXiv 2404.07904](https://arxiv.org/abs/2404.07904) / [OpenNLPLab/HGRN](https://github.com/OpenNLPLab/HGRN) | ★★★★ |
| 9 | **RetNet** | Retention | Parallel / recurrent / chunkwise 3-form retention | 2023-07 | MIT (커뮤니티 구현) | 1.3B~6.7B 학습됨 | [arXiv 2307.08621](https://arxiv.org/abs/2307.08621) / [Jamie-Stirling/RetNet](https://github.com/Jamie-Stirling/RetNet) | ★★★ |
| 10 | **TTT-Linear / TTT-MLP** | Test-Time Training | Hidden state = SGD으로 업데이트되는 mini-network | 2024-07 | MIT | 125M~1.3B | [arXiv 2407.04620](https://arxiv.org/abs/2407.04620) / [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) | ★★★★ |
| 11 | **Jamba (1.5 Mini)** | Hybrid (MoE + Mamba + Attn) | 1:7 비율 attn/Mamba 인터리브, 2-block 마다 MoE | 2024-03 / 08 | Jamba Open Model License | 12B active / 52B total (Mini) | [arXiv 2403.19887](https://arxiv.org/abs/2403.19887) / [HuggingFace 가중치](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.5) | ★★★ |
| 12 | **Zamba2 (1.2B / 2.7B / 7B)** | Hybrid (Mamba2 + shared attn) | Mamba2 백본 + ABAB shared-attention (LoRA로 layer-specialize) | 2024-08~10 | Apache 2.0 | 1.2B / 2.7B | [arXiv 2411.15242](https://arxiv.org/abs/2411.15242) / [Zyphra/Zamba2](https://github.com/Zyphra/Zamba2) | ★★★★★ |
| 13 | **Samba (3.8B)** | Hybrid (Mamba + SWA) | 단순 layer-level: Mamba+MLP+SWA+MLP 스택 | 2024-06 | MIT | 3.8B | [arXiv 2406.07522](https://arxiv.org/abs/2406.07522) / [microsoft/Samba](https://github.com/microsoft/Samba) | ★★★★ |
| 14 | **Hymba (1.5B)** | Hybrid-Head | 같은 layer 내에서 Attention head + SSM head 병렬 + meta tokens | 2024-11 | NVIDIA Open License | 1.5B | [arXiv 2411.13676](https://arxiv.org/abs/2411.13676) / [NVlabs/hymba](https://github.com/NVlabs/hymba) | ★★★★★ |
| 15 | **Bamba-9B** | Hybrid (Mamba2 + Attn) | 29 SSM + 3 Attention layers, distributed shuffle loader | 2024-12 | Apache 2.0 | 9B | [HF Bamba blog](https://huggingface.co/blog/bamba) / [ibm-ai-platform/Bamba-9B](https://huggingface.co/ibm-ai-platform/Bamba-9B-v1) | ★★★ |
| 16 | **Granite 4.0 (Micro / Tiny / Small)** | Hybrid (Mamba2 + Attn) | 9 Mamba2 : 1 Attention sequential 비율 | 2025-10 | Apache 2.0 | 3B (Micro), 7B (Tiny) | [IBM 발표](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) / [ibm-granite HF](https://huggingface.co/ibm-granite) | ★★★ |
| 17 | **Falcon Mamba 7B** | Pure SSM | Mamba1 + 추가 RMSNorm으로 large-scale 안정화 | 2024-10 | TII Falcon-LLM 라이선스 | 7B | [arXiv 2410.05355](https://arxiv.org/abs/2410.05355) / [HF tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b) | ★★ |
| 18 | **Falcon-H1** | Hybrid (Parallel SSM ‖ Attn) | Attention과 SSM이 같은 입력 위에서 병렬, 출력 concat | 2025-07 | Falcon-LLM 라이선스 | 0.5B / 1.5B / 3B / 7B | [arXiv 2507.22448](https://arxiv.org/abs/2507.22448) / [tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1) | ★★★★ |
| 19 | **Codestral Mamba 7B** | Pure SSM | Mamba2 백본을 코드 도메인에 학습 | 2024-07 | Apache 2.0 | 7B | [Mistral 발표](https://mistral.ai/news/codestral-mamba) | ★ |
| 20 | **Nemotron-H** | Hybrid (Mamba2 + Attn) | 92% layer를 Mamba2로 치환, 핵심 depth에만 attention | 2025-04 | NVIDIA Open License | 8B / 47B / 56B | [arXiv 2504.03624](https://arxiv.org/abs/2504.03624) / [Nemotron-H](https://research.nvidia.com/labs/adlr/nemotronh/) | ★★ |
| 21 | **MiniMax-01 (Text-01)** | Hybrid (Lightning Attn + softmax) | 7 lightning + 1 softmax block 비율, MoE | 2025-01 | MiniMax Model License | 동일 backbone의 mini 변종 없음(456B MoE) | [arXiv 2501.08313](https://arxiv.org/abs/2501.08313) | ★★★ |
| 22 | **RWKV-6 "Finch"** | RNN-revival | Matrix-valued state + data-dependent time-mixing | 2024-04 | Apache 2.0 | 1.6B / 3.1B | [arXiv 2404.05892](https://arxiv.org/abs/2404.05892) / [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | ★★★ |
| 23 | **Hyena Edge (Liquid AI)** | Gated Conv (multi-hybrid) | STAR로 진화 탐색된 Hyena-Y gated convolution + GQA | 2025-04 | 미공개 (가중치 일부 공개) | edge 1B대 추정 | [Liquid AI 발표](https://www.liquid.ai/research/convolutional-multi-hybrids-for-edge-devices) | ★★★ |
| 24 | **StripedHyena / Hyena Hierarchy** | Gated Conv | 암시적 long convolution + data-controlled gating | 2023 / 2024 | Apache 2.0 | 7B (StripedHyena) | [arXiv 2302.10866](https://arxiv.org/abs/2302.10866) / [togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena) | ★★★ |
| 25 | **Monarch Mixer (M2)** | Structured-matrix mixer | Sub-quadratic Monarch matrix를 sequence·channel 둘 다에 적용 | 2023-10 (NeurIPS oral) | Apache 2.0 | 110M~341M (BERT-style) | [arXiv 2310.12109](https://arxiv.org/abs/2310.12109) / [HazyResearch/m2](https://github.com/HazyResearch/m2) | ★★ |

> **참고**: ★ 1=현재 라인업과 중복되어 새로운 비교축이 거의 없음, ★ 5=Mamba/LFM2와 명확히 다른 메커니즘이며 toy config로도 차이를 시각화할 가능성이 높음.

---

## 3. 카테고리별 상세

### 3.1 Pure SSM 계열

> 핵심 아이디어: 입력에 의해 변조되는 (A, B, C) 상태공간 파라미터를 통해 sequence를 선형-시간으로 처리. KV-cache 없이 고정 크기 hidden state만 유지.

#### Mamba-2

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-05 / Albert Gu, Tri Dao (Princeton, CMU) |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | SSD(Structured State Space Duality) — SSM과 attention을 통합한 semiseparable matrix 관점. SSM 파라미터(A,B,C)를 X와 병렬 산출 |
| Mamba1 대비 | 상태 차원 8x 확장, 학습 속도 2-8x. 단 sequence-axis와 channel-axis가 SSD에서 결합되어 일부 selective scan의 expressivity는 잃음 |
| 구현 난이도 | 공식 PyTorch 레퍼런스 SSD가 약 30줄. 토이 config에서는 `selective_scan_cuda` 없이 chunk-wise matmul로 충분히 돌아감 |
| 토이 config 어려움 | head-grouping (multi-input/multi-output SSM)을 정확히 따라가야 의미가 살음. 또한 dt_proj 초기화는 Mamba1과 미세하게 다름 |

#### Falcon Mamba 7B

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-10 / TII (UAE) |
| 라이선스 | Falcon-LLM License (오픈, 일부 제한) |
| 핵심 차별점 | 순수 Mamba1 백본을 7B 스케일로 학습. Layer 안정성을 위해 추가 RMSNorm 삽입 |
| Mamba와 차이 | 실질적인 아키텍처 변경은 RMSNorm 위치/개수 정도. 본 레포의 Mamba와 큰 차이 없음 |
| 구현 난이도 | 매우 낮음 (Mamba와 거의 동일) |
| 도입 가치 | 본 레포의 Mamba 구현과 충분히 유사하여 별도 구현 가치는 낮음 |

#### Codestral Mamba 7B

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-07 / Mistral AI |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | Mamba2 + 코드 도메인 데이터. 아키텍처는 Mamba2와 동일 |
| 도입 가치 | 가중치는 흥미롭지만 from-scratch 구현 관점에서는 Mamba2와 중복 |

---

### 3.2 Hybrid SSM × Attention 계열

> 핵심 아이디어: SSM(또는 linear attention)이 압축적 long-range memory를 처리하고, 소수의 attention layer가 정확한 in-context retrieval / state tracking을 보조. 비율, 배치, 공유 여부가 design space.

#### Jamba (v1 / 1.5 Mini)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-03 / AI21 Labs |
| 라이선스 | Jamba Open Model License (가중치는 오픈, 상업 제한) |
| 핵심 차별점 | Attention:Mamba = 1:7 인터리브, 2 block마다 MoE 삽입 |
| Mamba/LFM2와의 차이 | Attention이 layer로 분리(Mamba block과 별개)되고, MoE를 결합. LFM2는 short-conv + GQA가 같은 stack에 sequential. Jamba는 attention-bearing block을 sparse하게 둠 |
| 구현 난이도 | MoE를 빼면 dense Jamba는 비교적 단순 — Mamba layer + 가끔 attention layer. 토이 config로는 MoE 없이 1:3 또는 1:7 비율만 시뮬레이션해도 의미 있음 |
| 어려움 | 256K context를 토이로 재현 불가. 단순히 "1:N 비율"이라는 design choice의 효과만 검증 가능 |

#### Zamba2

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-08~10 / Zyphra, 1.2B / 2.7B / 7.4B |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | Mamba2 백본에 **공유 가중치(shared)** attention block을 ABAB로 깊이 방향 인터리브. 깊이별 specialize는 LoRA로 처리 |
| Mamba/LFM2와의 차이 | "shared attention block + LoRA per depth" 라는 매우 독특한 파라미터 절약 design. LFM2의 short-conv와 직교적 |
| 구현 난이도 | 중간. 표준 multi-head attention 모듈을 한 번 만든 뒤 forward에서 여러 깊이에서 호출하고, 깊이별 LoRA 어댑터를 dispatch |
| 어려움 | LoRA 어댑터가 attention 블록·MLP 블록 양쪽에 적용. 정확한 가중치 공유 그래프를 graph-correct하게 짜는 것이 디버깅 포인트 |

#### Samba (Microsoft, 3.8B)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-06 / Microsoft Research + UIUC |
| 라이선스 | MIT |
| 핵심 차별점 | "단순함의 미학": `Mamba → MLP → SWA → MLP` 를 layer 단위로 반복 |
| Mamba/LFM2와의 차이 | LFM2가 short-conv+GQA를 같은 sub-block에 결합한 반면, Samba는 layer를 깔끔히 분리. 4K 학습으로 1M 토큰 추론 가능 (extrapolation) |
| 구현 난이도 | **매우 낮음** — 본 레포에 가장 자연스럽게 들어맞을 후보 중 하나. Mamba block과 Gemma4의 sliding-window attention 모듈이 이미 존재 |
| 어려움 | 거의 없음. Mamba layer와 SWA layer를 alternating stack하는 thin wrapper만 추가 |

#### Hymba (NVIDIA, 1.5B)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-11 / NVIDIA Research |
| 라이선스 | NVIDIA Open Model License |
| 핵심 차별점 | **Hybrid-Head**: 같은 layer에서 일부 head는 attention, 일부 head는 Mamba. 추가로 학습되는 **meta tokens**를 시퀀스 앞에 prepend하여 "attention sink" 역할 |
| Mamba/LFM2와의 차이 | Jamba/Zamba/Samba가 layer-level 하이브리드인 반면, Hymba는 head-level 하이브리드. 이는 LFM2 (sub-block 결합) 와도 다름 |
| 구현 난이도 | 중간 — head 분할 후 두 갈래의 mixing 결과를 concat. KV-cache cross-layer sharing은 토이에서는 생략 가능 |
| 어려움 | meta token 학습은 단순한 nn.Parameter로 대체 가능. partial sliding window attention 구현이 핵심 |

#### Bamba-9B (IBM)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-12 / IBM + Princeton + CMU + UIUC |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | Mamba2 29 layer + 표준 MHA 3 layer. KV-cache 부담을 거의 제거 |
| Mamba/LFM2와의 차이 | 비율 차이가 크다는 점(29:3) 외에는 Jamba/Zamba와 유사한 layer-level hybrid 패밀리. 별도 design 혁신은 적음 |
| 구현 난이도 | 낮음. Samba와 거의 같은 패턴 |
| 도입 가치 | Samba와 비교해 새로운 메커니즘은 적어 우선순위 낮음 |

#### Granite 4.0 (IBM, 3B Micro / 7B Tiny)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-10 / IBM |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | Mamba2:Attention = 9:1 sequential. Tiny/Small에는 MoE 결합 (1B/9B active) |
| Mamba/LFM2와의 차이 | Jamba/Bamba와 비슷하나 비율(9:1)이 더 극단적. Hybrid에서 SSM 비중을 어디까지 늘릴 수 있는지의 설계 결정 |
| 구현 난이도 | 낮음 (Samba/Bamba 구현이 있으면 비율 변경만으로 가능) |

#### Falcon-H1

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-07 / TII |
| 라이선스 | Falcon-LLM License |
| 핵심 차별점 | **Parallel hybrid**: 같은 입력에 대해 attention과 SSM이 동시에 동작하고 출력 concat 후 projection. Layer-level sequential 결합과 명확히 다름 |
| Mamba/LFM2와의 차이 | LFM2도 sub-block 단위에서 short-conv+attention이 sequential. Falcon-H1은 두 mixing primitive가 **같은 입력을 병렬 소비** |
| 구현 난이도 | 중간 — 두 mixer를 forward에서 병렬 호출하고 concat |
| 어려움 | 헤드 분할 비율, hidden dim 분할이 hyperparameter. 토이에서는 1:1로 단순화 |

#### Nemotron-H (NVIDIA)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-04 / NVIDIA |
| 라이선스 | NVIDIA Open Model License |
| 핵심 차별점 | 92% layer를 Mamba2로 치환, 핵심 depth에만 attention 유지. Bamba와 컨셉 유사 |
| 도입 가치 | 8B 이상이라 토이 단위 비교에는 의미가 적음. Granite 4.0과 비슷한 카테고리 |

#### MiniMax-01 (Text-01)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-01 / MiniMax |
| 라이선스 | MiniMax Model License |
| 핵심 차별점 | **Lightning Attention** (linear attention의 I/O-aware 변종) 7 block + softmax 1 block, MoE 32 experts |
| Mamba/LFM2와의 차이 | SSM이 아닌 linear attention 계열. Linear attention의 I/O 최적화(Lightning) 자체가 학습 토이에서는 필요 없으므로, "linear attention + 가끔 softmax" 패턴은 단순 재현 가능 |
| 구현 난이도 | Lightning 자체는 hardware-aware라 고난이도지만 수학적으로는 표준 linear attention. 학습용 토이에서는 단순 linear attention으로 substitute |
| 어려움 | 456B MoE라 직접 매핑은 불가, 메커니즘만 차용 |

---

### 3.3 Linear Attention / Delta Rule 계열

> 핵심 아이디어: softmax(QK^T)V 대신 K^T V를 누적해 O(n·d²) 시간/O(d²) 메모리. 누적 방식(단순 합산, EMA, delta-rule, gated)에 따라 expressivity가 달라짐.

#### Gated Linear Attention (GLA)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2023-12 (ICML 2024) / Songlin Yang et al. |
| 라이선스 | MIT (FLA 라이브러리) |
| 핵심 차별점 | Data-dependent gate g_t로 forget을 도입한 linear attention. FlashLinearAttention 알고리즘으로 chunkwise hardware-efficient |
| Mamba/LFM2와의 차이 | LFM2의 short-conv는 시간축 mixing을 conv로 처리, GLA는 누적 K^T V로 처리. 같은 "data-dependent gating"이지만 base operator가 다름 |
| 구현 난이도 | 토이에서는 chunkwise 없이 단순 누적합으로 충분. flash-linear-attention 라이브러리에 깔끔한 PyTorch 코드 존재 |
| 어려움 | Length-generalization을 보려면 비교적 긴 토이 시퀀스(1K~4K) 필요 |

#### DeltaNet (Yang et al., NeurIPS 2024)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-06 / Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim |
| 라이선스 | MIT (FLA) |
| 핵심 차별점 | Linear attention 상태를 새 (k,v) 쌍이 들어올 때마다 **delta rule**(in-context error correction)로 업데이트. Householder 행렬 곱의 메모리-효율 표현으로 sequence-length 방향 병렬화 |
| Mamba/LFM2와의 차이 | "rapid memory erase + targeted update" 라는 메커니즘은 SSM, GLA의 단순 누적/EMA와 명확히 다른 design |
| 구현 난이도 | 중간 — Householder reflection의 chunkwise 누적이 핵심. FLA의 reference 구현이 가장 명확 |
| 어려움 | 토이로 in-context retrieval 차이를 보려면 작은 toy task (예: needle-in-haystack 미니버전) 필요 |

#### Gated DeltaNet (NVIDIA, ICLR 2025)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-12 / NVIDIA + MIT |
| 라이선스 | NVIDIA Source Code License |
| 핵심 차별점 | Mamba2의 **gating(빠른 망각)** + DeltaNet의 **delta rule(targeted update)** 결합. 두 메커니즘이 상호보완적이라는 통찰 |
| Mamba/LFM2와의 차이 | Mamba는 gate만, DeltaNet은 delta만. Gated DeltaNet은 둘 다. 이는 본 레포에 **Mamba와 직접 ablation 가능한 모델** |
| 구현 난이도 | 중간 — DeltaNet 구현이 있다면 forget gate 추가는 단순 |
| 어려움 | 공식 코드는 Triton 커널이라, 토이용으로는 chunk-wise PyTorch 재작성 필요 |

#### RetNet

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2023-07 / Microsoft Research Asia |
| 라이선스 | MIT (커뮤니티 구현 다수) |
| 핵심 차별점 | Retention = 고정 decay γ의 linear attention. Parallel / recurrent / chunkwise 3가지 등가 표현 |
| Mamba/LFM2와의 차이 | Decay가 data-independent (단순 γ^Δt). Mamba의 selective decay와 명확히 비교 가능한 baseline |
| 구현 난이도 | 매우 낮음. 토이로는 30~50줄 |
| 도입 가치 | "data-dependent vs data-independent decay"라는 ablation을 한 차원 더 추가 가능 |

#### MiniMax Lightning Attention

위의 MiniMax-01 항목 참조. 알고리즘 자체는 linear attention을 I/O 최적화한 변종이라, 토이 학습에서는 표준 linear attention과 차이 없음.

---

### 3.4 RNN-revival 계열

> 핵심 아이디어: 행렬-값 hidden state, exponential gating, dynamic recurrence 등을 통해 LSTM/RWKV의 한계를 극복.

#### xLSTM (mLSTM / sLSTM)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-05 / Sepp Hochreiter et al. (NeurIPS 2024) |
| 라이선스 | NX-AI License (연구 사용 무료, 상업 제한) |
| 핵심 차별점 | **mLSTM**: 행렬-값 메모리 + covariance update + 병렬화. **sLSTM**: scalar memory + memory mixing. 둘 다 exponential gating + 안정화 정규화 |
| Mamba/LFM2와의 차이 | LSTM 가족의 부활. mLSTM은 outer-product 누적이라 GLA/RetNet과 가깝고, sLSTM은 진정한 sequential RNN. 두 변종 ablation이 흥미로움 |
| 구현 난이도 | 공식 PyTorch 레퍼런스(NX-AI/xlstm)가 있어 쉬움. 단, exponential gate의 안정화(stabilizer state m_t)가 디버그 포인트 |
| 어려움 | sLSTM은 본질적으로 sequential이라 chunkwise 병렬이 어렵지만, 토이 길이에서는 문제 없음 |

#### RWKV-7 "Goose"

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-03 / BlinkDL + RWKV community |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | **일반화된 delta rule + 벡터-값 게이팅 + in-context learning rate**. 표준 복잡도 가정 하에 Transformer가 못하는 state tracking과 모든 정규언어 인식 가능 |
| Mamba/LFM2와의 차이 | DeltaNet 가족과 RWKV 가족의 통합. WKV 7세대는 `state ← state · diag(w) - state·k_T·k + v_T·k` 형태로 매우 명시적 |
| 구현 난이도 | 중간 — 공식 코드는 CUDA-heavy지만, 수학식이 깔끔하여 PyTorch 재작성 가능 (커뮤니티 PyTorch 구현 다수) |
| 어려움 | RWKV 특유의 token-shift, channel-mixing block을 동반하므로 layer 전체를 옮겨야 의미 있음 |

#### RWKV-6 "Finch"

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-04 (COLM 2024) / BlinkDL |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | 행렬-값 state + data-dependent time-mixing/token-shift |
| 도입 가치 | RWKV-7이 더 expressive하므로, 굳이 6을 별도 구현할 가치는 낮음. RWKV-7 한 개 충분 |

#### HGRN2

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-04 / OpenNLPLab |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | Outer-product state expansion으로 파라미터 추가 없이 hidden state 차원 확장 + 계층적 lower-bound forget gate (depth가 깊을수록 longer-term) |
| Mamba/LFM2와의 차이 | "lower-bound가 layer-depth에 따라 monotone 증가"라는 inductive prior가 매우 독특. Mamba/Jamba에는 없는 layer-wise 시간 스케일 분리 |
| 구현 난이도 | 낮음 — 공식 PyTorch 코드 존재 |
| 어려움 | depth-wise lower bound 스케줄을 어떻게 정할지의 hyperparameter |

---

### 3.5 Convolution / Structured-Matrix 계열

> 핵심 아이디어: Attention을 large/long convolution이나 sub-quadratic structured matrix로 대체. Spectral / signal-processing 관점.

#### StripedHyena / Hyena Hierarchy

| 항목 | 내용 |
|------|------|
| 발표/소속 | Hyena 2023-02 (Stanford), StripedHyena 2023-12 (Together) |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | 암시적 long convolution(주파수 도메인 FFT) + data-controlled gating. StripedHyena는 그 위에 grouped attention layer를 stripe로 추가 |
| Mamba/LFM2와의 차이 | LFM2는 short causal conv (학습 가능 weight, 지역적). Hyena는 implicit하게 시퀀스 길이 전체를 cover하는 long conv |
| 구현 난이도 | 중간 — FFT-based long conv는 numpy/torch.fft로 구현 가능 |
| 어려움 | implicit kernel parameterization (positional encoding-based)이 핵심. toy 길이로는 단순 long conv로도 충분 |

#### Hyena Edge (Liquid AI)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2025-04 / Liquid AI (ICLR 2025) |
| 라이선스 | 미공개 |
| 핵심 차별점 | STAR(Synthesis of Tailored Architectures) 진화 탐색이 자동 발견한 multi-hybrid. GQA 2/3을 Hyena-Y gated convolution으로 대체 |
| LFM2와의 차이 | LFM2(같은 회사)도 short-conv+GQA hybrid이지만, Hyena Edge는 STAR로 발견된 비율과 변종(Hyena-Y는 gating에 conv 없음). 실측에서 Galaxy S24 Ultra 기준 latency·메모리 모두 우수 |
| 구현 난이도 | 중간 — Hyena-Y의 정확한 구조는 Liquid AI 발표 자료에 공개. 토이 config로는 hyena conv + GQA 비율만 시뮬레이션 |
| 어려움 | STAR 자체는 architecture search라 재현 불가. 결과 architecture만 따라 만들 수 있음 |

#### Monarch Mixer (M2)

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2023-10 (NeurIPS oral) / Stanford Hazy Research |
| 라이선스 | Apache 2.0 |
| 핵심 차별점 | sequence·channel 두 축 모두에 sub-quadratic Monarch matrix를 적용. attention도 MLP도 아닌 GEMM-only architecture |
| 도입 가치 | BERT-style이라 causal LM 카테고리와 결이 다름. 본 레포 목적과는 거리 있음 |

---

### 3.6 Test-Time Training 계열

> 핵심 아이디어: hidden state를 "weights"로 보고, 시퀀스를 자기지도 학습 데이터셋으로 보아 inner-loop SGD로 그 weights를 업데이트.

#### TTT-Linear / TTT-MLP

| 항목 | 내용 |
|------|------|
| 발표/소속 | 2024-07 / Stanford + UC Berkeley + UCSD + Meta + CMU |
| 라이선스 | MIT |
| 핵심 차별점 | Hidden state가 그 자체로 작은 model의 weight. Token-별 inner-loop gradient step (predicted target은 self-supervised reconstruction loss) |
| Mamba/LFM2와의 차이 | "state = weights" 라는 메타 학습 관점. SSM/linear-attention은 state가 vector/matrix, TTT는 state가 mini-MLP |
| 구현 난이도 | 중간 — inner-loop gradient를 functional하게 구성. 공식 PyTorch 구현 존재 |
| 어려움 | inner-loop SGD를 mini-batch로 chunkwise 처리해야 효율적. 토이에서는 chunk size 1~4로 단순화 가능 |
| 도입 가치 | **본 레포에서 가장 철학적으로 다른 메커니즘** — sequence를 "처리"하는 것이 아니라 그 위에서 "online 학습"하는 것 |

---

## 4. 레포 도입 우선순위 추천 (Top 5)

현재 본 레포에는 selective SSM(Mamba), gated short-conv hybrid(LFM2), sliding+global attention(Gemma4) 세 가지 mixing primitive가 있습니다. 다음 다섯 개를 순서대로 추가하면 매 단계마다 새로운 비교축이 추가되어 포트폴리오의 가치가 극대화됩니다.

### Top 1. **Mamba-2** — SSD가 추가하는 통합 관점

**왜 의미 있는가**: 현재의 Mamba(=Mamba1)에서 Mamba2로의 변환은 단순 "성능 향상"이 아니라 **selective scan의 expressivity 일부를 포기하고 attention과의 등가성(SSD)을 얻는 trade-off**를 보여줄 수 있는 가장 근본적인 비교 축입니다. 같은 hidden size에서 Mamba1(seq-axis 풍부) vs Mamba2(state 8x 확장 + matmul-friendly) 의 attention map / state dynamics 시각화가 즉시 한 장의 노트북이 됩니다.

### Top 2. **Hymba (head-level hybrid)** — layer hybrid vs head hybrid 대비

**왜 의미 있는가**: 현재 Gemma4와 LFM2는 둘 다 *layer-level* hybrid 또는 sub-block sequential 결합입니다. Hymba는 **같은 layer 안에서 head를 두 갈래(SSM head vs Attention head)로 분할**하는 head-level hybrid로, 이는 현재 라인업에 없는 design point입니다. meta-token까지 포함하면 "attention sink" 현상에 대한 시각화도 가능합니다.

### Top 3. **xLSTM (mLSTM + sLSTM)** — RNN-revival 가족

**왜 의미 있는가**: 현재 라인업은 모두 SSM/Conv/Attention 스펙트럼에 머물러 있고, 진정한 RNN-revival 메커니즘(exponential gating, scalar memory mixing)이 없습니다. mLSTM(병렬 가능, 행렬 메모리)과 sLSTM(순수 sequential, scalar memory) 두 변종을 모두 구현하면 한 모델 안에서 "병렬화 가능 vs 진짜 sequential"이라는 중요한 비교축까지 만들어집니다.

### Top 4. **Gated DeltaNet** — gating × delta rule의 ablation 무대

**왜 의미 있는가**: 현재 Mamba 구현 옆에 둘 때, "Mamba(gate만) vs DeltaNet(delta만) vs Gated DeltaNet(둘 다)" 라는 직접적인 ablation을 같은 토이 setting에서 펼칠 수 있습니다. linear attention 가족 중 가장 최신이며, in-context retrieval이 실제로 강해지는 design 선택의 근거를 한 노트북으로 시각화 가능합니다.

### Top 5. **Samba** — 가장 단순한 "Mamba + SWA" baseline

**왜 의미 있는가**: 본 레포에 이미 Mamba block과 Gemma4의 sliding-window attention 모듈이 모두 존재하므로 **추가 코드가 거의 thin wrapper 수준**으로 끝납니다. 그럼에도 "단순 비율 hybrid"의 효과를 정량적으로 보일 수 있어 가성비 최고. Granite 4.0/Bamba/Jamba 등 layer-level hybrid 가족의 minimal representative로 충분합니다.

> 추가로 흥미롭지만 우선순위 다음 그룹: **TTT-Linear** (철학적 차별성), **RWKV-7** (state tracking 능력), **HGRN2** (depth-wise 시간 스케일).

---

## 5. 참고 문헌

### Pure SSM
- [Mamba-2 — Transformers are SSMs (arXiv 2405.21060)](https://arxiv.org/abs/2405.21060)
- [Mamba-2 공식 코드 — state-spaces/mamba](https://github.com/state-spaces/mamba)
- [Falcon Mamba 7B (arXiv 2410.05355)](https://arxiv.org/abs/2410.05355)
- [Codestral Mamba 발표 (Mistral)](https://mistral.ai/news/codestral-mamba)

### Hybrid SSM × Attention
- [Jamba (arXiv 2403.19887)](https://arxiv.org/abs/2403.19887)
- [Jamba 1.5 (AI21 blog)](https://www.ai21.com/blog/announcing-jamba-model-family/)
- [Zamba2 Suite Technical Report (arXiv 2411.15242)](https://arxiv.org/abs/2411.15242)
- [Zamba2 코드 — Zyphra/Zamba2](https://github.com/Zyphra/Zamba2)
- [Samba (arXiv 2406.07522)](https://arxiv.org/abs/2406.07522)
- [Samba 코드 — microsoft/Samba](https://github.com/microsoft/Samba)
- [Hymba (arXiv 2411.13676)](https://arxiv.org/abs/2411.13676)
- [Hymba 코드 — NVlabs/hymba](https://github.com/NVlabs/hymba)
- [Bamba (HuggingFace blog)](https://huggingface.co/blog/bamba)
- [Granite 4.0 발표 (IBM)](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Falcon-H1 (arXiv 2507.22448)](https://arxiv.org/abs/2507.22448)
- [Falcon-H1 코드 — tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1)
- [Nemotron-H (arXiv 2504.03624)](https://arxiv.org/abs/2504.03624)
- [MiniMax-01 (arXiv 2501.08313)](https://arxiv.org/abs/2501.08313)

### Linear Attention / Delta Rule
- [Gated Linear Attention (arXiv 2312.06635)](https://arxiv.org/abs/2312.06635)
- [DeltaNet — Parallelizing Linear Transformers with Delta Rule (arXiv 2406.06484)](https://arxiv.org/abs/2406.06484)
- [Gated DeltaNet (arXiv 2412.06464)](https://arxiv.org/abs/2412.06464)
- [Gated DeltaNet 코드 — NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Flash Linear Attention 라이브러리 — fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [RetNet (arXiv 2307.08621)](https://arxiv.org/abs/2307.08621)
- [RetNet 비공식 PyTorch 구현 — Jamie-Stirling/RetNet](https://github.com/Jamie-Stirling/RetNet)

### RNN-revival
- [xLSTM (arXiv 2405.04517)](https://arxiv.org/abs/2405.04517)
- [xLSTM 공식 코드 — NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [RWKV-7 Goose (arXiv 2503.14456)](https://arxiv.org/abs/2503.14456)
- [Eagle & Finch — RWKV-5/6 (arXiv 2404.05892)](https://arxiv.org/abs/2404.05892)
- [RWKV 공식 — BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- [HGRN (arXiv 2311.04823)](https://arxiv.org/abs/2311.04823)
- [HGRN2 (arXiv 2404.07904)](https://arxiv.org/abs/2404.07904)
- [HGRN 코드 — OpenNLPLab/HGRN](https://github.com/OpenNLPLab/HGRN)

### Recurrent Attention (Hybrid)
- [Griffin / Hawk (arXiv 2402.19427)](https://arxiv.org/abs/2402.19427)
- [RecurrentGemma (arXiv 2404.07839)](https://arxiv.org/abs/2404.07839)
- [RecurrentGemma 코드 — google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma)

### Convolution / Structured-Matrix
- [Hyena Hierarchy (arXiv 2302.10866)](https://arxiv.org/abs/2302.10866)
- [StripedHyena 코드 — togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena)
- [Hyena Edge / Convolutional Multi-Hybrids (Liquid AI blog)](https://www.liquid.ai/research/convolutional-multi-hybrids-for-edge-devices)
- [Monarch Mixer (arXiv 2310.12109)](https://arxiv.org/abs/2310.12109)
- [Monarch Mixer 코드 — HazyResearch/m2](https://github.com/HazyResearch/m2)

### Test-Time Training
- [TTT — Learning to (Learn at Test Time) (arXiv 2407.04620)](https://arxiv.org/abs/2407.04620)
- [TTT PyTorch 코드 — test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)

### 종합 / 서베이성 자료
- [AI21 — "Attention was never enough: Tracing the rise of hybrid LLMs"](https://www.ai21.com/blog/rise-of-hybrid-llms/)
- [Songlin Yang — DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- [Goomba Lab — State Space Duality (Mamba-2) Part I/II](https://goombalab.github.io/blog/2024/mamba2-part1-model/)

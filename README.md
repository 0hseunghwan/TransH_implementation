# TransH Link Prediction (FB15k-237)

TransH 논문 (Wang, Zhang, Feng, Chen, *"Knowledge Graph Embedding by Translating on Hyperplanes"*, AAAI 2014)의 세 가지 실험 중 하나인 Link prediction 실험을 PyTorch로 재현하는 코드입니다.

## 구현 내용

- 각 relation마다 초평면(법선벡터 `w_r`)과 그 위에서의 translation `d_r`을 학습
- entity를 관계별 초평면에 투영(projection)한 뒤 TransE와 동일한 방식으로 스코어링
- soft constraint(엔티티 norm 제한, `w_r`-`d_r` 직교 제약)를 loss에 정규화 항으로 추가
- 평가 지표: Mean Rank(MR), MRR, Hits@10 (raw / filtered)

## 파일 구성

```
.
├── TransH.ipynb   # 메인 노트북
├── requirements.txt
├── .gitignore
└── README.md
```

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

GPU(CUDA)가 있으면 자동으로 사용. Google Colab에서 실행할 경우
`런타임 > 런타임 유형 변경 > GPU`로 설정하는 것을 권장

### 2. 노트북 실행

`TransH.ipynb` 순서대로 실행

1. **환경 설정** — torch 설치 및 device 확인
2. **데이터셋 다운로드** — FB15k-237 (`train.txt`, `valid.txt`, `test.txt`)를
   [dataset_FB15k-237](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/raw/master/data/FB15k-237)
   저장소에서 자동 다운로드하여 `fb15k237/` 디렉토리에 저장
3. **데이터 로딩 및 인덱싱** — entity/relation을 정수 ID로 매핑
4. **TransH 모델 구현** — hyperplane 투영, score function, soft constraints 포함
5. **Negative sampling** — head 또는 tail을 무작위로 교체 (uniform)
6. **Training** — margin ranking loss + SGD
7. **Training loss 시각화**
8. **Evaluation** — MR / MRR / Hits@10 (raw, filtered)

## 주요 하이퍼파라미터

기본값은 빠른 실습을 위한 설정이며, 논문에서 사용된 하이퍼파라미터 값과 다를 수 있음

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `DIM` | 100 | 임베딩 차원 |
| `BATCH_SIZE` | 1024 | 배치 크기 |
| `EPOCHS` | 20 | 논문 재현에는 500 epoch 이상 권장 |
| `LR` | 0.01 | SGD 학습률 |
| `MARGIN` | 1.0 | margin ranking loss의 margin |
| `C` | 0.25 | soft constraint 가중치 |


## 참고 / 튜닝 포인트

- `EPOCHS`, `DIM`, `LR`, `MARGIN`, `C`(soft constraint 가중치)는 TransE / TransH 논문 참고
- 평가(`evaluate`)에서 `max_test`를 늘리면(예: `None`으로 전체 test set) 논문과 동일한 전체 평가가 되지만,
  entity 수 × test triple 수만큼 forward pass가 필요해 CPU에서는 매우 느림. GPU 런타임 사용 권장
  (`런타임 > 런타임 유형 변경 > GPU`)
- negative sampling은 현재 uniform random 사용. 논문의 "bern" (relation의 하나의 head/tail 의 평균 tail/head 개수에 따라 확률적으로 head/tail 교체) 방식으로 바꾸면 성능이 더 오를 수도 있음.

## 참고 문헌

Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). *Knowledge Graph Embedding by Translating on Hyperplanes.*
Proceedings of the AAAI Conference on Artificial Intelligence, 28(1).

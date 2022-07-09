# object-tracking-2
object tracking 에 이어서 모델을 변경해 정확도 향상에 초점을 둔 프로젝트입니다.

[이전 프로젝트 링크](https://github.com/jjuun0/object-tracking)

## 1. 모델 변경
- SiamRPN++ → CSWinTT

## 2. 플렌옵틱 시퀀스에 맞게 코드 수정
- 첫 트래킹시에는 100개의 focal sequence 를 체크하기 때문에 conf score > 0.5 이상만 다음 state 에 추가가 된다. → state 는 리스트로 여러개로 만들어짐
    - 따라서 각 state 마다 search region 이 다르게 만들어져 더 많은 결과를 얻을 수 있게 됐다.
- 두번째 트래킹시 부터는 첫번째 트래킹시 conf score 가 높은 focal index 를 기준으로 앞뒤 5개의 index, 총 11개의 focal sequence 를 체크하게 된다.
- 각 focal sequence 마다 max state 를 추가해준다.
    - 여러 state가 있기 때문에 각 state 마다 conf score을 측정해 max state 를 찾아낸다.
- 이 11개의 max focals 를 conf 기준으로 내림차순으로 정렬해
    - 11 개중 높은 score 6 개를 다음 state 로 적용한다.(연산향을 줄이기 위해)

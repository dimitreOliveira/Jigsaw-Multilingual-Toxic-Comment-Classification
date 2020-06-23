## Model backlog (list of the developed model and metrics)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **ROC AUC**.
- **Public LB** is the Public Leaderboard score.
- **Private LB** is the Private Leaderboard score.

---

## Models

|Model|Train|Validation|Public LB|Private LB|
|-----|-----|----------|---------|----------|
| 1-Jigsaw-Train-DistilBERT ML Cased-toxic | 0.502 | 0.693 | 0.7015 | ??? |
| 2-Jigsaw-Train-DistilBERT ML Cased-toxic | 0.503 | 0.795 | 0.7621 | ??? |
| 3-Jigsaw-Train-DistilBERT ML-toxic AVG_MAX | 0.498 | 0.795 | 0.7726 | ??? |
| 4-Jigsaw-Train-DistilBERT ML Cased-bias | 0.502 | 0.528 | 0.7447 | ??? |
| 5-Jigsaw-Train-DistilBERT ML-bias AVG_MAX | 0.500 | 0.529 | 0.7825 | ??? |
| 6-Jigsaw-Train-DistilBERT ML-toxic pb | 0.502 | 0.796 | 0.7742 | ??? |
| 7-Jigsaw-Train-DistilBERT ML-bias AVG_MAX focal | 0.507 | 0.674 | 0.6734 | ??? |
| 8-Jigsaw-Train-BERT ML-toxic pb | 0.498 | 0.828 | 0.8202 | ??? |
| 9-Jigsaw-Train-BERT ML-toxic pb | 0.502 | 0.688 | 0.6806 | ??? |
| 10-Jigsaw-Train-3Fold-XLM_roBERTa_base-toxic | 0.932 | 0.907 | 0.6713 | ??? |
| 11-Jigsaw-Train-3Fold-XLM_roBERTa_base | 0.852 | 0.852 | 0.7699 | ??? |
| 12-Jigsaw-Train-3Fold-XLM_roBERTa_base | 0.800 | 0.799 | 0.6339 | ??? |
| 13-Jigsaw-Train-1Fold-XLM_roBERTa large | 0.772 | 0.771 | 0.5870 | ??? |
| 14-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9328 | ??? |
| 15-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9337 | ??? |
| 16-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9322 | ??? |
| 17-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9309 | ??? |
| 18-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9002 | ??? |
| 19-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.5663 | ??? |
| 20-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9267 | ??? |
| 21-Jigsaw-Train-2Fold-XLM_roBERTa large | ??? | ??? | 0.8507 | ??? |
| 22-Jigsaw-Train-2Fold-XLM_roBERTa large numpy | ??? | ??? | 0.9137 | ??? |
| 23-Jigsaw-Train-2Fold-XLM_roBERTa large seed | ??? | ??? | 0.9100 | ??? |
| 24-Jigsaw-Train-2Fold-XLM_roBERTa large se | ??? | ??? | 0.9108 | ??? |
| 25-Jigsaw-Train-1Fold-XLM_roBERTa base | ??? | ??? | 0.8321 | ??? |
| 26-Jigsaw-Train-1Fold-XLM_roBERTa large | ??? | ??? | 0.9109 | ??? |
| 27-Jigsaw-Train-1Fold-XLM_roBERTa large false rema | ??? | ??? | 0.8979 | ??? |
| 28-Jigsaw-Train-1Fold-XLM_roBERTa large att mask | ??? | ??? | 0.9153 | ??? |
| 29-Jigsaw-Train-1Fold-XLM_roBERTa large seed3 | ??? | ??? | 0.9128 | ??? |
| 30-Jigsaw-Train-1Fold-XLM_roBERTa large Avg | ??? | ??? | 0.9027 | ??? |
| 31-Jigsaw-Train-1Fold-XLM_roBERTa large optimized | ??? | ??? | 0.9133 | ??? |
| 32-Jigsaw-Train-1Fold-XLM_roBERTa large mixed prec | ??? | ??? | 0.9120 | ??? |
| 33-Jigsaw-Train-1Fold-XLM_roBERTa large XLA | ??? | ??? | 0.5025 | ??? |
| 34-Jigsaw-Train-1Fold-XLM_roBERTa large mixed XLA | ??? | ??? | 0.9167 | ??? |
| 35-Jigsaw-Train-1Fold-XLM_roBERTa large ratio_1 | ??? | ??? | 0.9214 | ??? |
| 36-Jigsaw-Train-1Fold-XLM_roBERTa large ratio_2 | ??? | ??? | 0.9166 | ??? |
| 37-Jigsaw-Train-1Fold-XLM_roBERTa large ratio_4 | ??? | ??? | 0.5030 | ??? |
| 38-Jigsaw-Train-1Fold-XLM_roBERTa large ratio_1 | ??? | ??? | 0.9195 | ??? |
| 39-Jigsaw-Train-1Fold-XLM_roBERTa large avg_max | ??? | ??? | 0.9197 | ??? |
| 40-Jigsaw-Train-1Fold-XLM_roBERTa large clean | ??? | ??? | 0.9234 | ??? |
| 41-Jigsaw-Train-1Fold-XLM_roBERTa large clean tail | ??? | ??? | 0.9207 | ??? |
| 42-Jigsaw-Train-1Fold-XLM_roBERTa large clean tail | ??? | ??? | 0.9224 | ??? |
| 43-Jigsaw-Train-1Fold-XLM_roBERTa large trail cls | ??? | ??? | 0.9191 | ??? |
| 44-Jigsaw-Train-1Fold-XLM_roBERTa large tail | ??? | ??? | 0.9045 | ??? |
| 45-Jigsaw-Train-1Fold-XLM_roBERTa large tail float | ??? | ??? | 0.9182 | ??? |
| 46-Jigsaw-Train-1Fold-XLM_roBERTa large tail int | ??? | ??? | 0.9204 | ??? |
| 47-Jigsaw-Train-1Fold-XLM_roBERTa base | ??? | ??? | 0.8925 | ??? |
| 48-Jigsaw-Train-1Fold-XLM_roBERTa large LRSchedule | ??? | ??? | 0.9146 | ??? |
| 49-Jigsaw-1Fold-XLM_roBERTa large LRSchedule 08 | ??? | ??? | 0.9141 | ??? |
| 50-Jigsaw-2Fold-XLM_roBERTa large LR Constant | ??? | ??? | 0.9193 | ??? |
| 51-Jigsaw-1Fold-XLM_roBERTa large LR Const ratio_2 | ??? | ??? | 0.9198 | ??? |
| 52-Jigsaw-1Fold-XLM_roBERTa large LR Constant | ??? | ??? | 0.9127 | ??? |
| 53-Jigsaw-1Fold-XLM_roBERTa large step LR Constant | ??? | ??? | v | ??? |
| 54-Jigsaw-1Fold-XLM_roBERTa large step RAdam | ??? | ??? | 0.9174 | ??? |
| 55-Jigsaw-1Fold-XLM_roBERTa large step RAdam lower | ??? | ??? | 0.9185 | ??? |
| 56-Jigsaw-1Fold-XLM_roBERTa large linear decay | ??? | ??? | 0.9241 | ??? |
| 57-Jigsaw-1Fold-XLM_roBERTa large linear no warmup | ??? | ??? | 0.9221 | ??? |
| 58-Jigsaw-1Fold-XLM_roBERTa large lbl smoothing 01 | ??? | ??? | 0.9250 | ??? |
| 59-Jigsaw-1Fold-XLM_roBERTa large lbl smooth 02 | ??? | ??? | 0.9250 | ??? |
| 60-Jigsaw-Fold1-XLM_roBERTa large cosine decay | 0.987417 | 0.915397 | 0.9112 | ??? |
| 61-Jigsaw-Fold1-XLM_roBERTa large cosine warmup | 0.982849 | 0.916137 | 0.9007 | ??? |
| 62-Jigsaw-Fold1-XLM_roBERTa large exponential warm | 0.946594 | 0.922751 | 0.9214 | ??? |
| 63-Jigsaw-Fold1-XLM_roBERTa large exponential warm | 0.923352 | 0.923352 | 000 | ??? |
| 64-Jigsaw-Fold1-XLM_roBERTa large exponent upper | ??? | ??? | 0.9229 | ??? |
| 65-Jigsaw-Fold1-XLM_roBERTa large exponent no tail | ??? | ??? | 0.9230 | ??? |
| 66-Jigsaw-Fold1-XLM_roBERTa large mixed labels | ??? | ??? | 0.9270 | ??? |
| 67-Jigsaw-Fold1-XLM_roBERTa large ml data | ??? | ??? | 0.9391 | ??? |
| 68-Jigsaw-Fold1-XLM_roBERTa large cls token | ??? | ??? | 0.9388 | ??? |
| 69-Jigsaw-Fold1-XLM_roBERTa large LR 3e-5 | ??? | ??? | 0.9232 | ??? |
| 70-Jigsaw-Fold1-XLM_roBERTa large pred mixed label | ??? | ??? | 0.9375 | ??? |
| 71-Jigsaw-Fold1-XLM_roBERTa large focal loss | ??? | ??? | 0.9377 | ??? |
| 72-Jigsaw-Fold1-XLM_roBERTa large cosine restart | ??? | ??? | 0.4906 | ??? |
| 73-Jigsaw-Fold1-XLM_roBERTa large CLS exponential | ??? | 0.920458 | 0.9292 | ??? |
| 74-Jigsaw-Fold1-XLM_roBERTa large CLS exponential2 | ??? | 0.922624 | 0.9217 | ??? |
| 75-Jigsaw-Fold1-XLM_roBERTa large CLS tail | ??? | 0.922301 | 0.9403 | ??? |
| 76-Jigsaw-Fold1-XLM_roBERTa large CLS ml tail | ??? | 0.9232 | 0.9254 | ??? |
| 77-Jigsaw-Fold1-XLM_roBERTa large CLS 128 | ??? | 0.92202 | 0.9233 | ??? |
| 78-Jigsaw-Fold1-XLM_roBERTa large CLS 128 tail | ??? | 0.918278 | 0.9367 | ??? |
| 79-Jigsaw-Fold1-XLM_roBERTa large CLS 128 float | ??? | 0.918036 | 0.9364 | ??? |
| 80-Jigsaw-Fold1-XLM_roBERTa large 128 high float | ??? | 0.917543 | 000 | ??? |
| 81-Jigsaw-Fold1-XLM_roBERTa large 128 high int | ??? | 0.918644 | 0.9216 | ??? |
| 82-Jigsaw-Fold1-XLM_roBERTa large 128 high float08 | ??? | 0.915272 | 0.9220 | ??? |
| 83-Jigsaw-Fold1-XLM_roBERTa large 128 clean tail | ??? | 0.925274 | 0.9280 | ??? |
| 84-Jigsaw-Fold1-XLM_roBERTa large 128 alphanumeric | ??? | 0.921053 | 0.9277 | ??? |
| 85-Jigsaw-Fold1-XLM_roBERTa large 128 best | ??? | 0.921654 | 0.9244 | ??? |
| 86-Jigsaw-Fold1-XLM_roBERTa large 128 last | ??? | 0.922163 | 0.9187 | ??? |
| 87-Jigsaw-Fold1-XLM_roBERTa large 128 bs 256 | ??? | 0.920576 | 0.9196 | ??? |
| 88-Jigsaw-Fold1-XLM_roBERTa large 128 bs 256 | ??? | 0.921483 | 0.9220 | ??? |
| 89-Jigsaw-Fold1-XLM_roBERTa large 128 int | 0.931689 | 0.918185 | 0.9202 | ??? |
| 90-Jigsaw-Fold1-XLM_roBERTa large 128 lbl smooth01 | 0.947793 | 0.917091 | 0.9187 | ??? |
| 91-Jigsaw-Fold1-XLM_roBERTa large 128 last | 0.940337 | 0.915416 | 0.9132 | ??? |
| 92-Jigsaw-Fold1-XLM_roBERTa large 128 cls hidden11 | 0.942628 | 0.914737 | 0.9218 | ??? |
| 93-Jigsaw-Fold1-XLM_roBERTa large 128 AVG | 0.944806 | 0.917717 | 0.9206 | ??? |
| 94-Jigsaw-Fold1-XLM_roBERTa large 128 AVG hidden11 | 0.946852 | 0.918410 | 0.9179 | ??? |
| 95-Jigsaw-Fold1-XLM_roBERTa large best | 000 | 0.921937 | 0.9403 | ??? |
| 96-Jigsaw-Fold1-XLM_roBERTa large 128 best | 000 | 0.918719 | 0.9299 | ??? |
| 97-Jigsaw-Fold1-XLM_roBERTa large 192 | 0.947843 | 0.923894 | 0.9259 | ??? |
| 98-Jigsaw-Fold1-XLM_roBERTa large 224 | 0.948865 | 0.927250 | 0.9194 | ??? |
| 99-Jigsaw-Fold1-XLM_roBERTa large best | 0.951657 | 0.923948 | 0.9400 | ??? |
| 100-Jigsaw-Fold2-XLM_roBERTa large best | 0.952586 | 0.925539 | 0.9403 | ??? |
| 101-Jigsaw-Fold3-XLM_roBERTa large best | 0.952752 | 0.924753 | 0.9395 | ??? |
| 102-Jigsaw-Fold4-XLM_roBERTa large best | 0.952200 | 0.924454 | 0.9407 | ??? |
| 103-Jigsaw-Fold5-XLM_roBERTa large best | 0.951592 | 0.924500 | 0.9393 | ??? |
| 104-Jigsaw-Fold1-XLM_roBERTa large best3 | 0.950109 | 0.923406 | 0.9228 | ??? |
| 105-Jigsaw-Fold1-XLM_roBERTa large best4 | 0.955983 | 0.925805 | 0.9372 | ??? |
| 106-Jigsaw-Fold1-XLM_roBERTa large best5 | 0.953012 | 0.923134 | 0.9386 | ??? |
| 107-Jigsaw-Fold1-XLM_roBERTa large best6 | 0.946119 | 0.923696 | 0.9403 | ??? |
| 108-Jigsaw-Fold1-XLM_roBERTa large 128 polish | 0.945167 | 0.921218 | 0.9370 | ??? |
| 109-Jigsaw-Fold1-XLM_roBERTa large 224 polish | 0.954515 | 0.924343 | 0.9402 | ??? |
| 110-Jigsaw-Fold1-XLM_roBERTa cosine 4 epochs | 000 | 0.917335 | 0.9068 | ??? |
| 111-Jigsaw-Fold1-XLM_roBERTa exponential 4 epochs | 000 | 0.921949 | 0.9409 | ??? |
| 112-Jigsaw-Fold1-XLM_roBERTa exponential best ep | 000 | 0.922049 | 0.9398 | ??? |
| 113-Jigsaw-Fold1-XLM_roBERTa exponential 3epoch | 000 | 0.924234 | 0.9234 | ??? |
| 114-Jigsaw-Fold1-XLM_roBERTa exponential 2epochs | 000 | 0.923881 | 0.9220 | ??? |
| 115-Jigsaw-Fold1-XLM_roBERTa exponential 4 epochs | 000 | 0.919703 | 0.9207 | ??? |
| 116-Jigsaw-Fold1-XLM_roBERTa exponential 4 epochs | 000 | 0.922264 | 0.9412 | ??? |
| 117-Jigsaw-Fold1-XLM_roBERTa exponential 4 epochs | 000 | 0.923012 | 0.9290 | ??? |
| 118-Jigsaw-Fold1-XLM_roBERTa exponential 3 epochs | 000 | 0.921504 | 0.9377 | ??? |
| 119-Jigsaw-Fold1-XLM_roBERTa exponential 3 epochs | 000 | 0.918649 | 0.9379 | ??? |
| 120-Jigsaw-Fold1-XLM_roBERTa ratio_1 exp 3 epochs | 000 | 0.921277 | 0.9390 | ??? |
| 121-Jigsaw-Fold1-XLM_roBERTa ratio_4 exp 3 epochs | 000 | 0.561200 | 0.5372 | ??? |
| 122-Jigsaw-Fold1-XLM_roBERTa ratio_1 10_warmup | 000 | 0.920408 | 0.9364 | ??? |
| 123-Jigsaw-Fold1-XLM_roBERTa ratio_1 2_optimizers | 000 | 0.919934 | 0.9362 | ??? |
| 124-Jigsaw-Fold1-XLM_roBERTa ratio_1 AVG | 000 | 0.913944 | 0.9375 | ??? |
| 125-Jigsaw-Fold1-XLM_roBERTa ratio_1 MAX | 000 | 0.925247 | 0.9374 | ??? |
| 126-Jigsaw-Fold1-XLM_roBERTa ratio_1 AVG_MAX | 000 | 0.917274 | 0.9371 | ??? |
| 127-Jigsaw-Fold1-XLM_roBERTa ratio_1 AVG_MAX norm | 000 | 0.898280 | 0.9281 | ??? |
| 128-Jigsaw-Fold1-XLM_roBERTa ratio_1 AVG_MAX dropo | 000 | 0.919696 | 0.9419 | ??? |
| 129-Jigsaw-Fold2-XLM_roBERTa ratio_1 AVG_MAX dropo | 000 | 0.923416 | 0.9398 | ??? |
| 130-Jigsaw-Fold3-XLM_roBERTa ratio_1 AVG_MAX dropo | 000 | 0.922633 | 0.9357 | ??? |
| 131 | 000 | 000 | 000 | ??? |
| 132 | 000 | 000 | 000 | ??? |
| 133-Jigsaw-Fold1-XLM_roBERTa ratio_1 double drop | 000 | 0.922888 | 0.9402 | ??? |
| 134-Jigsaw-Fold1-XLM_roBERTa ratio_1 conv pooling | 000 | 0.917031 | 0.9396 | ??? |
| 135-Jigsaw-Fold1-XLM_roBERTa ratio_1 2-sample drop | 000 | 0.915444 | 0.9396 | ??? |
| 136-Jigsaw-Fold1-XLM_roBERTa ratio_1 8-sample drop | 000 | 0.921630 | 0.9400 | ??? |
| 137-Jigsaw-Fold1-XLM_roBERTa ratio_1 16sample drop | 000 | 0.913923 | 0.9273 | ??? |
| 138-Jigsaw-Fold2-XLM_roBERTa ratio_1 8-sample drop | 000 | 0.922340 | 000 | ??? |
| 139-Jigsaw-Fold3-XLM_roBERTa ratio_1 8-sample drop | 000 | 0.917284 | 000 | ??? |
| 140-Jigsaw-Fold4-XLM_roBERTa ratio_1 8-sample drop | 000 | 0.921714 | 000 | ??? |
| 141-Jigsaw-Fold5-XLM_roBERTa ratio_1 8-sample drop | 000 | 0.923529 | 000 | ??? |
| 142-Fold1-XLM_roBERTa custom head 8-sample dropout | 000 | 0.922390 | 000 | ??? |
| 143-Fold1-XLM_roBERTa hidden_23 8-sample dropout | 000 | 0.920463 | 000 | ??? |
| 144-Fold1-XLM_roBERTa cls_hidden_23 8-sample drop | 000 | 0.920163 | 000 | ??? |
| 145-Fold1-XLM_roBERTa open_sub pretrai 8-samp drop | 000 | 0.925158 | 000 | ??? |

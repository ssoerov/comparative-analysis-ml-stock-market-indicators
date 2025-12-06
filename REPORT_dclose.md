# Отчёт по моделям прогнозирования (почасовые данные)

Версия: автогенерация из артефактов в папке outputs.

## Методология без утечек для IMOEX

Прогнозируем почасовой ряд индекса IMOEX, добавляя экзогенные факторы только в виде лагов. Используемые формулы:

$$r_t = \mu + \beta^\top X_{t-1} + \varepsilon_t, \qquad \sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2 + \gamma^\top X_{t-1}.$$

Вектор $X_{t-1}$ содержит лаги Brent, USD/RUB, ключевой ставки, RSI(50), ATR(50) и лаги цены. Параметры CatBoost/RF и LSTM обучаются только на стандартизованных лаговых матрицах; гибридная LSTM использует прогноз SARIMAX и σ_t из GARCH как признаки риска. Ни один признак не знает будущего, эмбарго и walk-forward разбиения исключают утечки.

## Матрица признаков (Y и Xi)

```
Y=y | X1=Value | X2=SMA_5 | X3=SMA_10 | X4=SMA_20 | X5=EMA_5 | X6=EMA_10 | X7=EMA_20 | X8=BBH | X9=BBL | X10=RSI_50 | X11=StochK | X12=StochD | X13=ATR_50 | X14=OBV | X15=MACD | X16=MACD_SIGNAL | X17=MACD_DIFF | X18=ADX_14 | X19=CCI_20 | X20=ROC_10 | X21=WILLR_14 | X22=lag1 | X23=lag2 | X24=lag3 | X25=lag4 | X26=lag5 | X27=lag6 | X28=lag7 | X29=lag8 | X30=lag9 | X31=lag10 | X32=lag11 | X33=lag12 | X34=lag13 | X35=lag14 | X36=lag15 | X37=lag16 | X38=lag17 | X39=lag18 | X40=lag19 | X41=lag20 | X42=lag21 | X43=lag22 | X44=lag23 | X45=lag24 | X46=lag25 | X47=lag26 | X48=lag27 | X49=lag28 | X50=lag29 | X51=lag30 | X52=lag31 | X53=lag32 | X54=lag33 | X55=lag34 | X56=lag35 | X57=lag36 | X58=lag37 | X59=lag38 | X60=lag39 | X61=lag40 | X62=lag41 | X63=lag42 | X64=lag43 | X65=lag44 | X66=lag45 | X67=lag46 | X68=lag47 | X69=lag48 | X70=lag49 | X71=lag50 | X72=lag51 | X73=lag52 | X74=lag53 | X75=lag54 | X76=lag55 | X77=lag56 | X78=lag57 | X79=lag58 | X80=lag59 | X81=lag60 | X82=Brent_lag1 | X83=Brent_lag2 | X84=Brent_lag3 | X85=Brent_lag4 | X86=Brent_lag5 | X87=Brent_lag6 | X88=Brent_lag7 | X89=Brent_lag8 | X90=Brent_lag9 | X91=Brent_lag10 | X92=Brent_lag11 | X93=Brent_lag12 | X94=Brent_lag13 | X95=Brent_lag14 | X96=Brent_lag15 | X97=Brent_lag16 | X98=Brent_lag17 | X99=Brent_lag18 | X100=Brent_lag19 | X101=Brent_lag20 | X102=Brent_lag21 | X103=Brent_lag22 | X104=Brent_lag23 | X105=Brent_lag24 | X106=USD_lag1 | X107=USD_lag2 | X108=USD_lag3 | X109=USD_lag4 | X110=USD_lag5 | X111=USD_lag6 | X112=USD_lag7 | X113=USD_lag8 | X114=USD_lag9 | X115=USD_lag10 | X116=USD_lag11 | X117=USD_lag12 | X118=USD_lag13 | X119=USD_lag14 | X120=USD_lag15 | X121=USD_lag16 | X122=USD_lag17 | X123=USD_lag18 | X124=USD_lag19 | X125=USD_lag20 | X126=USD_lag21 | X127=USD_lag22 | X128=USD_lag23 | X129=USD_lag24
```

## Сводные метрики по тикерам и моделям

```
   Tk     Model       MAE      RMSE     MASE    CumRet    MaxDD
IMOEX    Hybrid  9.847342 13.408129 0.670888 -0.002862 0.013580
IMOEX LSTM_base  9.861054 13.722488 0.710452 -0.005809 0.015423
IMOEX  LSTM_att 10.166408 13.803463 0.780939 -0.008133 0.016650
IMOEX   SARIMAX 10.333646 13.985901 0.800075 -0.011939 0.022295
IMOEX        RF 11.181735 14.232439 0.928106  0.000623 0.011846
IMOEX  CatBoost 11.269434 14.593115 1.012452 -0.010504 0.017857
IMOEX    sNaive 13.335667 18.595654 1.029239 -0.011021 0.019217
IMOEX     Naive 15.244333 20.059274 1.000000  0.012547 0.006466
```

## Лучшие модели по MAE на тикер

```
   Tk  Model      MAE      RMSE     MASE    CumRet   MaxDD
IMOEX Hybrid 9.847342 13.408129 0.670888 -0.002862 0.01358
```

## Лучшие модели по MASE на тикер

```
   Tk  Model      MAE      RMSE     MASE    CumRet   MaxDD
IMOEX Hybrid 9.847342 13.408129 0.670888 -0.002862 0.01358
```

## Попарные DM-тесты по метрикам

```
   Tk    Model1    Model2 Metric   DM_stat    p_val
IMOEX    Hybrid     Naive    MAE -3.480960 0.000500
IMOEX LSTM_base     Naive    MAE -3.211452 0.001321
IMOEX  LSTM_att     Naive    MAE -3.160995 0.001572
IMOEX     Naive   SARIMAX    MAE  3.025976 0.002478
IMOEX  CatBoost LSTM_base    MAE  2.929134 0.003399
IMOEX    Hybrid    sNaive    MAE -2.667046 0.007652
IMOEX LSTM_base    sNaive    MAE -2.613588 0.008960
IMOEX  CatBoost  LSTM_att    MAE  2.551718 0.010719
IMOEX     Naive        RF    MAE  2.545579 0.010910
IMOEX  LSTM_att    sNaive    MAE -2.359113 0.018319
IMOEX  CatBoost     Naive    MAE -2.259162 0.023873
IMOEX   SARIMAX    sNaive    MAE -2.228640 0.025838
IMOEX  CatBoost    Hybrid    MAE  2.055479 0.039833
IMOEX    Hybrid        RF    MAE -1.946494 0.051595
IMOEX        RF    sNaive    MAE -1.767085 0.077214
IMOEX LSTM_base        RF    MAE -1.527072 0.126743
IMOEX  CatBoost    sNaive    MAE -1.522562 0.127868
IMOEX  CatBoost   SARIMAX    MAE  1.455060 0.145653
IMOEX  LSTM_att        RF    MAE -1.365998 0.171939
IMOEX        RF   SARIMAX    MAE  1.339649 0.180360
IMOEX     Naive    sNaive    MAE  0.977071 0.328534
IMOEX  LSTM_att LSTM_base    MAE  0.944817 0.344752
IMOEX LSTM_base   SARIMAX    MAE -0.877929 0.379982
IMOEX    Hybrid   SARIMAX    MAE -0.767339 0.442880
IMOEX    Hybrid  LSTM_att    MAE -0.736012 0.461724
IMOEX  LSTM_att   SARIMAX    MAE -0.335127 0.737529
IMOEX  CatBoost        RF    MAE  0.102276 0.918538
IMOEX    Hybrid LSTM_base    MAE -0.022375 0.982149
IMOEX    Hybrid    sNaive   MAPE -3.162600 0.001564
IMOEX    Hybrid        RF   MAPE -3.153125 0.001615
IMOEX LSTM_base     Naive   MAPE -3.061683 0.002201
IMOEX  LSTM_att     Naive   MAPE -3.039047 0.002373
IMOEX  CatBoost    Hybrid   MAPE  2.728544 0.006361
IMOEX  LSTM_att    sNaive   MAPE -2.711589 0.006696
IMOEX     Naive   SARIMAX   MAPE  2.711121 0.006706
IMOEX LSTM_base    sNaive   MAPE -2.675099 0.007471
IMOEX    Hybrid     Naive   MAPE -2.661960 0.007769
IMOEX  LSTM_att        RF   MAPE -2.498519 0.012471
IMOEX LSTM_base        RF   MAPE -2.363987 0.018079
IMOEX        RF    sNaive   MAPE -2.363117 0.018122
IMOEX   SARIMAX    sNaive   MAPE -2.348561 0.018846
IMOEX  CatBoost  LSTM_att   MAPE  2.297279 0.021603
IMOEX  CatBoost LSTM_base   MAPE  2.173658 0.029731
IMOEX        RF   SARIMAX   MAPE  1.788853 0.073639
IMOEX  CatBoost   SARIMAX   MAPE  1.778824 0.075269
IMOEX  LSTM_att   SARIMAX   MAPE -1.516414 0.129415
IMOEX LSTM_base   SARIMAX   MAPE -1.411710 0.158035
IMOEX  CatBoost    sNaive   MAPE -1.186255 0.235522
IMOEX     Naive    sNaive   MAPE -1.134766 0.256473
IMOEX    Hybrid   SARIMAX   MAPE -1.041688 0.297556
IMOEX  CatBoost        RF   MAPE  0.809908 0.417993
IMOEX    Hybrid LSTM_base   MAPE  0.615256 0.538386
IMOEX  LSTM_att LSTM_base   MAPE  0.587699 0.556734
IMOEX    Hybrid  LSTM_att   MAPE  0.497478 0.618852
IMOEX  CatBoost     Naive   MAPE  0.318855 0.749836
IMOEX     Naive        RF   MAPE  0.094036 0.925081
IMOEX    Hybrid    sNaive  MDAPE -3.162600 0.001564
IMOEX    Hybrid        RF  MDAPE -3.153125 0.001615
IMOEX LSTM_base     Naive  MDAPE -3.061683 0.002201
IMOEX  LSTM_att     Naive  MDAPE -3.039047 0.002373
IMOEX  CatBoost    Hybrid  MDAPE  2.728544 0.006361
IMOEX  LSTM_att    sNaive  MDAPE -2.711589 0.006696
IMOEX     Naive   SARIMAX  MDAPE  2.711121 0.006706
IMOEX LSTM_base    sNaive  MDAPE -2.675099 0.007471
IMOEX    Hybrid     Naive  MDAPE -2.661960 0.007769
IMOEX  LSTM_att        RF  MDAPE -2.498519 0.012471
IMOEX LSTM_base        RF  MDAPE -2.363987 0.018079
IMOEX        RF    sNaive  MDAPE -2.363117 0.018122
IMOEX   SARIMAX    sNaive  MDAPE -2.348561 0.018846
IMOEX  CatBoost  LSTM_att  MDAPE  2.297279 0.021603
IMOEX  CatBoost LSTM_base  MDAPE  2.173658 0.029731
IMOEX        RF   SARIMAX  MDAPE  1.788853 0.073639
IMOEX  CatBoost   SARIMAX  MDAPE  1.778824 0.075269
IMOEX  LSTM_att   SARIMAX  MDAPE -1.516414 0.129415
IMOEX LSTM_base   SARIMAX  MDAPE -1.411710 0.158035
IMOEX  CatBoost    sNaive  MDAPE -1.186255 0.235522
IMOEX     Naive    sNaive  MDAPE -1.134766 0.256473
IMOEX    Hybrid   SARIMAX  MDAPE -1.041688 0.297556
IMOEX  CatBoost        RF  MDAPE  0.809908 0.417993
IMOEX    Hybrid LSTM_base  MDAPE  0.615256 0.538386
IMOEX  LSTM_att LSTM_base  MDAPE  0.587699 0.556734
IMOEX    Hybrid  LSTM_att  MDAPE  0.497478 0.618852
IMOEX  CatBoost     Naive  MDAPE  0.318855 0.749836
IMOEX     Naive        RF  MDAPE  0.094036 0.925081
IMOEX    Hybrid     Naive   RMSE -2.336679 0.019456
IMOEX     Naive        RF   RMSE  2.265070 0.023508
IMOEX  LSTM_att     Naive   RMSE -2.243441 0.024868
IMOEX     Naive   SARIMAX   RMSE  2.216583 0.026652
IMOEX LSTM_base     Naive   RMSE -2.139243 0.032416
IMOEX  CatBoost     Naive   RMSE -2.072302 0.038237
IMOEX LSTM_base    sNaive   RMSE -2.039695 0.041381
IMOEX   SARIMAX    sNaive   RMSE -1.928334 0.053814
IMOEX  LSTM_att    sNaive   RMSE -1.901328 0.057259
IMOEX  CatBoost    sNaive   RMSE -1.791294 0.073246
IMOEX    Hybrid    sNaive   RMSE -1.766490 0.077314
IMOEX        RF    sNaive   RMSE -1.755024 0.079255
IMOEX  CatBoost  LSTM_att   RMSE  1.079928 0.280174
IMOEX  CatBoost    Hybrid   RMSE  0.693598 0.487934
IMOEX  CatBoost   SARIMAX   RMSE  0.669511 0.503170
IMOEX  CatBoost LSTM_base   RMSE  0.557502 0.577185
IMOEX    Hybrid        RF   RMSE -0.471774 0.637088
IMOEX     Naive    sNaive   RMSE  0.456762 0.647842
IMOEX  LSTM_att        RF   RMSE -0.347796 0.727993
IMOEX        RF   SARIMAX   RMSE  0.309456 0.756975
IMOEX  LSTM_att LSTM_base   RMSE -0.267016 0.789457
IMOEX    Hybrid LSTM_base   RMSE -0.222122 0.824219
IMOEX    Hybrid   SARIMAX   RMSE -0.203785 0.838522
IMOEX    Hybrid  LSTM_att   RMSE -0.159200 0.873511
IMOEX LSTM_base        RF   RMSE -0.156945 0.875288
IMOEX  LSTM_att   SARIMAX   RMSE -0.150476 0.880389
IMOEX  CatBoost        RF   RMSE  0.127779 0.898324
IMOEX LSTM_base   SARIMAX   RMSE  0.122098 0.902821
IMOEX    Hybrid  LSTM_att  SMAPE -3.523122 0.000426
IMOEX  LSTM_att LSTM_base  SMAPE  3.024679 0.002489
IMOEX  CatBoost    Hybrid  SMAPE  2.436089 0.014847
IMOEX  LSTM_att    sNaive  SMAPE  2.334458 0.019572
IMOEX  LSTM_att     Naive  SMAPE  2.256089 0.024065
IMOEX  CatBoost LSTM_base  SMAPE  2.177610 0.029435
IMOEX    Hybrid        RF  SMAPE -2.122440 0.033801
IMOEX    Hybrid   SARIMAX  SMAPE -1.799389 0.071957
IMOEX  CatBoost    sNaive  SMAPE  1.705526 0.088096
IMOEX  CatBoost     Naive  SMAPE  1.586898 0.112536
IMOEX        RF    sNaive  SMAPE  1.502367 0.133002
IMOEX  LSTM_att   SARIMAX  SMAPE  1.365742 0.172020
IMOEX   SARIMAX    sNaive  SMAPE  1.184740 0.236120
IMOEX     Naive        RF  SMAPE -1.132155 0.257569
IMOEX  LSTM_att        RF  SMAPE  1.113313 0.265574
IMOEX    Hybrid LSTM_base  SMAPE -1.101289 0.270771
IMOEX    Hybrid     Naive  SMAPE -1.068323 0.285375
IMOEX  CatBoost  LSTM_att  SMAPE -1.028481 0.303724
IMOEX     Naive   SARIMAX  SMAPE -1.007669 0.313614
IMOEX LSTM_base        RF  SMAPE -0.933281 0.350675
IMOEX LSTM_base   SARIMAX  SMAPE -0.930122 0.352308
IMOEX    Hybrid    sNaive  SMAPE -0.842369 0.399582
IMOEX  CatBoost   SARIMAX  SMAPE  0.706034 0.480167
IMOEX  CatBoost        RF  SMAPE  0.464716 0.642135
IMOEX        RF   SARIMAX  SMAPE  0.212137 0.832000
IMOEX LSTM_base    sNaive  SMAPE  0.173986 0.861876
IMOEX     Naive    sNaive  SMAPE  0.168264 0.866376
IMOEX LSTM_base     Naive  SMAPE  0.025996 0.979261
IMOEX    Hybrid    sNaive   WAPE -3.162600 0.001564
IMOEX    Hybrid        RF   WAPE -3.153125 0.001615
IMOEX LSTM_base     Naive   WAPE -3.061683 0.002201
IMOEX  LSTM_att     Naive   WAPE -3.039047 0.002373
IMOEX  CatBoost    Hybrid   WAPE  2.728544 0.006361
IMOEX  LSTM_att    sNaive   WAPE -2.711589 0.006696
IMOEX     Naive   SARIMAX   WAPE  2.711121 0.006706
IMOEX LSTM_base    sNaive   WAPE -2.675099 0.007471
IMOEX    Hybrid     Naive   WAPE -2.661960 0.007769
IMOEX  LSTM_att        RF   WAPE -2.498519 0.012471
IMOEX LSTM_base        RF   WAPE -2.363987 0.018079
IMOEX        RF    sNaive   WAPE -2.363117 0.018122
IMOEX   SARIMAX    sNaive   WAPE -2.348561 0.018846
IMOEX  CatBoost  LSTM_att   WAPE  2.297279 0.021603
IMOEX  CatBoost LSTM_base   WAPE  2.173658 0.029731
IMOEX        RF   SARIMAX   WAPE  1.788853 0.073639
IMOEX  CatBoost   SARIMAX   WAPE  1.778824 0.075269
IMOEX  LSTM_att   SARIMAX   WAPE -1.516414 0.129415
IMOEX LSTM_base   SARIMAX   WAPE -1.411710 0.158035
IMOEX  CatBoost    sNaive   WAPE -1.186255 0.235522
IMOEX     Naive    sNaive   WAPE -1.134766 0.256473
IMOEX    Hybrid   SARIMAX   WAPE -1.041688 0.297556
IMOEX  CatBoost        RF   WAPE  0.809908 0.417993
IMOEX    Hybrid LSTM_base   WAPE  0.615256 0.538386
IMOEX  LSTM_att LSTM_base   WAPE  0.587699 0.556734
IMOEX    Hybrid  LSTM_att   WAPE  0.497478 0.618852
IMOEX  CatBoost     Naive   WAPE  0.318855 0.749836
IMOEX     Naive        RF   WAPE  0.094036 0.925081
```

## Анализ важности факторов

Средняя значимость признаков усреднена по фолдам и моделям (RF, CatBoost). Список ограничен десятью ключевыми факторами.

```
  Feature  Importance
    Value    0.041156
   RSI_50    0.035177
   ROC_10    0.031409
MACD_DIFF    0.028091
   CCI_20    0.024596
   StochD    0.024061
 WILLR_14    0.022291
     MACD    0.021983
   StochK    0.020132
   ATR_50    0.019224
```

![feature_importance_top5_overall](outputs/reports/feature_importance_top5_overall.png)

![feature_importance_top5_CatBoost](outputs/reports/feature_importance_top5_CatBoost.png)

![feature_importance_top5_RF](outputs/reports/feature_importance_top5_RF.png)

## Временные ряды топ‑5 индикаторов

![Value](outputs/reports/indicator_series_Value.png)

![RSI_50](outputs/reports/indicator_series_RSI_50.png)

![ROC_10](outputs/reports/indicator_series_ROC_10.png)

![MACD_DIFF](outputs/reports/indicator_series_MACD_DIFF.png)

![CCI_20](outputs/reports/indicator_series_CCI_20.png)

![StochD](outputs/reports/indicator_series_StochD.png)

## Формулы индикаторов

```
Y (цель): dClose_t = Close_{t+1} − Close_t

Value: денежный оборот за бар (MOEX Value).

SMA_n: SMA_n(t) = mean(Close_{t−n+1..t})  (n ∈ {5,10,20})
EMA_n: EMA_n(t) = α·Close_t + (1−α)·EMA_n(t−1),  α = 2/(n+1)  (n ∈ {5,10,20})

BB (Bollinger, n=20, k=2):
  BBH = SMA_20 + 2·σ_20,   BBL = SMA_20 − 2·σ_20

RSI_50: RSI = 100 − 100/(1 + RS),  RS = EMA(Gain,50)/EMA(Loss,50)

Stochastics (окно n):
  %K = 100·(Close − L_n)/(H_n − L_n),  %D = SMA(%K, 3)
  H_n = max(High_{t−n+1..t}),  L_n = min(Low_{t−n+1..t})

ATR_50: ATR = RMA(TR,50),  TR_t = max(High−Low, |High−Close_{t−1}|, |Low−Close_{t−1}|)

OBV: OBV_t = OBV_{t−1} + sign(Close_t − Close_{t−1})·Volume_t

MACD: MACD = EMA_12(Close) − EMA_26(Close)
MACD_SIGNAL = EMA_9(MACD),  MACD_DIFF = MACD − MACD_SIGNAL

ADX_14: ADX = RMA(DX,14),  DX = 100·|+DI − −DI|/(+DI + −DI)
  +DI, −DI получаются из +DM, −DM и TR по схеме Уайлдера

CCI_20: CCI = (TP − SMA(TP,20)) / (0.015·MD_20),  TP=(H+L+C)/3,
  MD_20 = mean(|TP − SMA(TP,20)|) за 20 баров

ROC_10: ROC = 100·(Close_t/Close_{t−10} − 1)
WILLR_14: −100·(H_14 − Close)/(H_14 − L_14)

Лаги цены: lag_k = Close_{t−k},  k = 1..60
Лаги экзогенных факторов: Brent_lag_k, USD_lag_k, KeyRate_lag_k = соответствующий уровень на t−k,  k = 1..24
```

## Цепной анализ IMOEX

Сводка по месячным цепным приращениям (фрагмент):

```
  begin   Close  ChainIncrement  MeanRelGrowthPct  Volatility
2023-10 3200.97          -26.10         -0.808783         NaN
2023-11 3165.79          -35.18         -0.049107   15.525597
2023-12 3099.11          -66.68         -0.098335   24.368236
2024-01 3214.19          115.08          0.174412   11.553472
2024-02 3256.80           42.61          0.069252   26.942123
2024-03 3332.53           75.73          0.115965   14.845580
2024-04 3469.83          137.30          0.176723   15.905302
2024-05 3217.19         -252.64         -0.355336   30.897251
2024-06 3154.36          -62.83         -0.093060   47.056312
2024-07 2942.68         -211.68         -0.292823   41.044346
2024-08 2650.32         -292.36         -0.464896   39.177986
2024-09 2857.56          207.24          0.373861   46.640322
```

Дневные цепные приросты (фрагмент):

```
                     Date   Close  ChainIncrement  ChainGrowthCoef  RelativeGrowth  RelativeGrowthPct  ChainIndex  BaseAbsolute  BaseRelativeCoef  BaseRelativePct  ChainAbsoluteIndex  ChainRelativeIndex  StructuralShiftAbsolute  StructuralShiftRelativePct
2023-10-31 00:00:00+00:00 3200.97          -26.10         0.991912       -0.008088          -0.808783    0.991912        -26.10          0.991912        -0.808783              -26.10            0.991912                   -26.10                   -0.808783
2023-11-01 00:00:00+00:00 3206.52            5.55         1.001734        0.001734           0.173385    0.993632        -20.55          0.993632        -0.636801                5.55            1.001734                     5.55                    0.173385
2023-11-02 00:00:00+00:00 3197.22           -9.30         0.997100       -0.002900          -0.290034    0.990750        -29.85          0.990750        -0.924988               -9.30            0.997100                    -9.30                   -0.290034
2023-11-03 00:00:00+00:00 3208.63           11.41         1.003569        0.003569           0.356873    0.994286        -18.44          0.994286        -0.571416               11.41            1.003569                    11.41                    0.356873
2023-11-06 00:00:00+00:00 3235.11           26.48         1.008253        0.008253           0.825274    1.002491          8.04          1.002491         0.249142               26.48            1.008253                    26.48                    0.825274
2023-11-07 00:00:00+00:00 3246.34           11.23         1.003471        0.003471           0.347129    1.005971         19.27          1.005971         0.597136               11.23            1.003471                    11.23                    0.347129
2023-11-08 00:00:00+00:00 3245.43           -0.91         0.999720       -0.000280          -0.028032    1.005689         18.36          1.005689         0.568937               -0.91            0.999720                    -0.91                   -0.028032
2023-11-09 00:00:00+00:00 3239.92           -5.51         0.998302       -0.001698          -0.169777    1.003982         12.85          1.003982         0.398194               -5.51            0.998302                    -5.51                   -0.169777
2023-11-10 00:00:00+00:00 3242.06            2.14         1.000661        0.000661           0.066051    1.004645         14.99          1.004645         0.464508                2.14            1.000661                     2.14                    0.066051
2023-11-13 00:00:00+00:00 3248.27            6.21         1.001915        0.001915           0.191545    1.006569         21.20          1.006569         0.656943                6.21            1.001915                     6.21                    0.191545
2023-11-14 00:00:00+00:00 3212.39          -35.88         0.988954       -0.011046          -1.104588    0.995451        -14.68          0.995451        -0.454902              -35.88            0.988954                   -35.88                   -1.104588
2023-11-15 00:00:00+00:00 3215.11            2.72         1.000847        0.000847           0.084672    0.996294        -11.96          0.996294        -0.370615                2.72            1.000847                     2.72                    0.084672
```

Файл включает цепные/базисные относительные и абсолютные индексы и коэффициент структурного сдвига.

Ключевые показатели цепного роста:

```
                   Metric     Value
       mean_abs_increment 26.765967
        median_growth_pct  0.066402
           max_growth_pct  9.185789
           min_growth_pct -3.971219
mean_structural_shift_pct  0.911423
```

![IMOEX_chain_growth](outputs/imoex_analysis/IMOEX_chain_growth.png)

![IMOEX_trend_component.png](outputs/imoex_analysis/IMOEX_trend_component.png)

![IMOEX_seasonal_component.png](outputs/imoex_analysis/IMOEX_seasonal_component.png)

![IMOEX_acf_chain.png](outputs/imoex_analysis/IMOEX_acf_chain.png)

![IMOEX_pacf_chain.png](outputs/imoex_analysis/IMOEX_pacf_chain.png)

## Примеры графиков

Графики факта/прогнозов, ACF остатков, гистограммы и кривые капитала — см. папку `outputs/reports/`.

![IMOEX_IMOEX_f0_all_models.png](reports/IMOEX_IMOEX_f0_all_models.png)

![IMOEX_IMOEX_f1_all_models.png](reports/IMOEX_IMOEX_f1_all_models.png)

![IMOEX_IMOEX_f2_all_models.png](reports/IMOEX_IMOEX_f2_all_models.png)

![IMOEX_IMOEX_f3_all_models.png](reports/IMOEX_IMOEX_f3_all_models.png)

![IMOEX_IMOEX_f4_all_models.png](reports/IMOEX_IMOEX_f4_all_models.png)

![IMOEX_f0_actual_vs_CatBoost.png](reports/IMOEX_f0_actual_vs_CatBoost.png)

## Факт и прогноз на всём горизонте (все фолды)

![IMOEX_full_LSTM_att.png](outputs/reports/IMOEX_full_LSTM_att.png)

![IMOEX_full_CatBoost.png](outputs/reports/IMOEX_full_CatBoost.png)

![IMOEX_full_Hybrid.png](outputs/reports/IMOEX_full_Hybrid.png)

![IMOEX_full_SARIMAX.png](outputs/reports/IMOEX_full_SARIMAX.png)

## Кластеризация режимов по тикерам

```
   Tk  k  silhouette  n_points
IMOEX  5    0.318751      3262
```

![IMOEX_clusters_price.png](outputs/clustering/IMOEX_clusters_price.png)

![IMOEX_cluster_pie.png](outputs/clustering/IMOEX_cluster_pie.png)


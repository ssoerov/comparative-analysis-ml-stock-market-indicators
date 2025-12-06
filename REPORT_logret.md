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
   Tk     Model      MAE     RMSE     MASE    CumRet    MaxDD
IMOEX   SARIMAX 0.000163 0.000212 0.051069  0.012519 0.006501
IMOEX        RF 0.002264 0.003391 0.491579  0.003871 0.011395
IMOEX  CatBoost 0.002332 0.003224 0.523532  0.005996 0.011435
IMOEX    Hybrid 0.003246 0.004456 0.698365 -0.001010 0.011660
IMOEX LSTM_base 0.003284 0.004505 0.719699 -0.003703 0.014878
IMOEX  LSTM_att 0.003378 0.004574 0.826303 -0.008133 0.016650
IMOEX     Naive 0.004907 0.006500 1.000000 -0.028001 0.032155
IMOEX    sNaive 0.004913 0.006916 1.163511 -0.018970 0.024931
```

## Лучшие модели по MAE на тикер

```
   Tk   Model      MAE     RMSE     MASE   CumRet    MaxDD
IMOEX SARIMAX 0.000163 0.000212 0.051069 0.012519 0.006501
```

## Лучшие модели по MASE на тикер

```
   Tk   Model      MAE     RMSE     MASE   CumRet    MaxDD
IMOEX SARIMAX 0.000163 0.000212 0.051069 0.012519 0.006501
```

## Попарные DM-тесты по метрикам

```
   Tk    Model1    Model2 Metric    DM_stat         p_val
IMOEX  LSTM_att   SARIMAX    MAE   7.196429  6.181001e-13
IMOEX     Naive   SARIMAX    MAE   7.114773  1.120968e-12
IMOEX LSTM_base   SARIMAX    MAE   6.764075  1.341632e-11
IMOEX    Hybrid   SARIMAX    MAE   6.723191  1.777869e-11
IMOEX   SARIMAX    sNaive    MAE  -6.690355  2.226305e-11
IMOEX  CatBoost   SARIMAX    MAE   5.703760  1.171934e-08
IMOEX  CatBoost     Naive    MAE  -5.047228  4.482665e-07
IMOEX        RF   SARIMAX    MAE   4.973662  6.569993e-07
IMOEX     Naive        RF    MAE   4.773873  1.807161e-06
IMOEX  CatBoost    sNaive    MAE  -4.631151  3.636394e-06
IMOEX        RF    sNaive    MAE  -4.609598  4.034490e-06
IMOEX  CatBoost  LSTM_att    MAE  -4.108077  3.989679e-05
IMOEX  CatBoost LSTM_base    MAE  -3.637203  2.756149e-04
IMOEX    Hybrid    sNaive    MAE  -3.537739  4.035681e-04
IMOEX  LSTM_att        RF    MAE   3.448519  5.636702e-04
IMOEX  CatBoost    Hybrid    MAE  -3.421192  6.234735e-04
IMOEX LSTM_base    sNaive    MAE  -3.364547  7.666933e-04
IMOEX LSTM_base        RF    MAE   3.296547  9.788126e-04
IMOEX  LSTM_att    sNaive    MAE  -3.207525  1.338822e-03
IMOEX    Hybrid     Naive    MAE  -3.147697  1.645621e-03
IMOEX LSTM_base     Naive    MAE  -3.006854  2.639663e-03
IMOEX  LSTM_att     Naive    MAE  -2.855068  4.302768e-03
IMOEX    Hybrid        RF    MAE   2.813002  4.908140e-03
IMOEX    Hybrid  LSTM_att    MAE  -1.684559  9.207366e-02
IMOEX  LSTM_att LSTM_base    MAE   0.934570  3.500100e-01
IMOEX  CatBoost        RF    MAE   0.421744  6.732120e-01
IMOEX    Hybrid LSTM_base    MAE  -0.259528  7.952280e-01
IMOEX     Naive    sNaive    MAE  -0.008555  9.931746e-01
IMOEX  LSTM_att   SARIMAX   MAPE   7.799501  6.215242e-15
IMOEX  CatBoost   SARIMAX   MAPE   6.913784  4.718937e-12
IMOEX        RF   SARIMAX   MAPE   6.517726  7.138117e-11
IMOEX LSTM_base   SARIMAX   MAPE   6.393015  1.626460e-10
IMOEX     Naive   SARIMAX   MAPE   5.519312  3.403296e-08
IMOEX    Hybrid   SARIMAX   MAPE   5.339441  9.323377e-08
IMOEX   SARIMAX    sNaive   MAPE  -4.307733  1.649361e-05
IMOEX     Naive        RF   MAPE   4.111222  3.935700e-05
IMOEX    Hybrid    sNaive   MAPE  -3.510449  4.473500e-04
IMOEX  CatBoost    sNaive   MAPE  -3.440888  5.798094e-04
IMOEX        RF    sNaive   MAPE  -3.383281  7.162537e-04
IMOEX  CatBoost     Naive   MAPE  -3.372784  7.441228e-04
IMOEX  LSTM_att    sNaive   MAPE  -3.144216  1.665325e-03
IMOEX LSTM_base    sNaive   MAPE  -3.093870  1.975638e-03
IMOEX  LSTM_att     Naive   MAPE  -2.680059  7.360913e-03
IMOEX    Hybrid     Naive   MAPE  -2.677416  7.419254e-03
IMOEX LSTM_base     Naive   MAPE  -2.467872  1.359189e-02
IMOEX  CatBoost LSTM_base   MAPE  -2.161418  3.066304e-02
IMOEX  LSTM_att        RF   MAPE   2.073719  3.810545e-02
IMOEX LSTM_base        RF   MAPE   2.034712  4.187981e-02
IMOEX  CatBoost  LSTM_att   MAPE  -1.707009  8.782034e-02
IMOEX     Naive    sNaive   MAPE  -1.518396  1.289145e-01
IMOEX    Hybrid        RF   MAPE   1.398440  1.619809e-01
IMOEX  CatBoost    Hybrid   MAPE  -1.185612  2.357754e-01
IMOEX  CatBoost        RF   MAPE   0.866525  3.862024e-01
IMOEX    Hybrid LSTM_base   MAPE  -0.804339  4.212013e-01
IMOEX    Hybrid  LSTM_att   MAPE  -0.624104  5.325591e-01
IMOEX  LSTM_att LSTM_base   MAPE  -0.349054  7.270485e-01
IMOEX  LSTM_att   SARIMAX  MDAPE   7.799501  6.215242e-15
IMOEX  CatBoost   SARIMAX  MDAPE   6.913784  4.718937e-12
IMOEX        RF   SARIMAX  MDAPE   6.517726  7.138117e-11
IMOEX LSTM_base   SARIMAX  MDAPE   6.393015  1.626460e-10
IMOEX     Naive   SARIMAX  MDAPE   5.519312  3.403296e-08
IMOEX    Hybrid   SARIMAX  MDAPE   5.339441  9.323377e-08
IMOEX   SARIMAX    sNaive  MDAPE  -4.307733  1.649361e-05
IMOEX     Naive        RF  MDAPE   4.111222  3.935700e-05
IMOEX    Hybrid    sNaive  MDAPE  -3.510449  4.473500e-04
IMOEX  CatBoost    sNaive  MDAPE  -3.440888  5.798094e-04
IMOEX        RF    sNaive  MDAPE  -3.383281  7.162537e-04
IMOEX  CatBoost     Naive  MDAPE  -3.372784  7.441228e-04
IMOEX  LSTM_att    sNaive  MDAPE  -3.144216  1.665325e-03
IMOEX LSTM_base    sNaive  MDAPE  -3.093870  1.975638e-03
IMOEX  LSTM_att     Naive  MDAPE  -2.680059  7.360913e-03
IMOEX    Hybrid     Naive  MDAPE  -2.677416  7.419254e-03
IMOEX LSTM_base     Naive  MDAPE  -2.467872  1.359189e-02
IMOEX  CatBoost LSTM_base  MDAPE  -2.161418  3.066304e-02
IMOEX  LSTM_att        RF  MDAPE   2.073719  3.810545e-02
IMOEX LSTM_base        RF  MDAPE   2.034712  4.187981e-02
IMOEX  CatBoost  LSTM_att  MDAPE  -1.707009  8.782034e-02
IMOEX     Naive    sNaive  MDAPE  -1.518396  1.289145e-01
IMOEX    Hybrid        RF  MDAPE   1.398440  1.619809e-01
IMOEX  CatBoost    Hybrid  MDAPE  -1.185612  2.357754e-01
IMOEX  CatBoost        RF  MDAPE   0.866525  3.862024e-01
IMOEX    Hybrid LSTM_base  MDAPE  -0.804339  4.212013e-01
IMOEX    Hybrid  LSTM_att  MDAPE  -0.624104  5.325591e-01
IMOEX  LSTM_att LSTM_base  MDAPE  -0.349054  7.270485e-01
IMOEX     Naive   SARIMAX   RMSE   3.599990  3.182289e-04
IMOEX    Hybrid   SARIMAX   RMSE   3.420108  6.259617e-04
IMOEX  LSTM_att   SARIMAX   RMSE   3.304710  9.507474e-04
IMOEX        RF    sNaive   RMSE  -3.197069  1.388319e-03
IMOEX  CatBoost    sNaive   RMSE  -3.099813  1.936425e-03
IMOEX   SARIMAX    sNaive   RMSE  -3.073840  2.113231e-03
IMOEX LSTM_base   SARIMAX   RMSE   3.058881  2.221653e-03
IMOEX  CatBoost LSTM_base   RMSE  -2.997166  2.725025e-03
IMOEX  CatBoost  LSTM_att   RMSE  -2.953540  3.141521e-03
IMOEX  CatBoost     Naive   RMSE  -2.879064  3.988578e-03
IMOEX  CatBoost    Hybrid   RMSE  -2.740234  6.139551e-03
IMOEX     Naive        RF   RMSE   2.598523  9.362589e-03
IMOEX LSTM_base    sNaive   RMSE  -2.576635  9.976714e-03
IMOEX  LSTM_att    sNaive   RMSE  -2.497500  1.250723e-02
IMOEX    Hybrid    sNaive   RMSE  -2.453112  1.416264e-02
IMOEX    Hybrid     Naive   RMSE  -2.220091  2.641259e-02
IMOEX  LSTM_att     Naive   RMSE  -2.169877  3.001618e-02
IMOEX LSTM_base     Naive   RMSE  -2.115337  3.440122e-02
IMOEX  CatBoost   SARIMAX   RMSE   2.056418  3.974226e-02
IMOEX LSTM_base        RF   RMSE   1.920773  5.476029e-02
IMOEX        RF   SARIMAX   RMSE   1.799930  7.187163e-02
IMOEX  LSTM_att        RF   RMSE   1.685031  9.198264e-02
IMOEX    Hybrid        RF   RMSE   1.487422  1.369034e-01
IMOEX  CatBoost        RF   RMSE  -0.727626  4.668425e-01
IMOEX    Hybrid  LSTM_att   RMSE  -0.321069  7.481582e-01
IMOEX    Hybrid LSTM_base   RMSE  -0.258827  7.957684e-01
IMOEX  LSTM_att LSTM_base   RMSE  -0.190804  8.486792e-01
IMOEX     Naive    sNaive   RMSE  -0.182694  8.550381e-01
IMOEX  LSTM_att   SARIMAX  SMAPE  21.682430 3.005817e-104
IMOEX   SARIMAX    sNaive  SMAPE -15.608618  6.359958e-55
IMOEX LSTM_base   SARIMAX  SMAPE  14.043609  8.430751e-45
IMOEX     Naive   SARIMAX  SMAPE  13.471224  2.310041e-41
IMOEX    Hybrid   SARIMAX  SMAPE  12.173555  4.299398e-34
IMOEX  CatBoost   SARIMAX  SMAPE  11.182781  4.951357e-29
IMOEX        RF   SARIMAX  SMAPE   8.515672  1.656267e-17
IMOEX  LSTM_att        RF  SMAPE   6.728863  1.709944e-11
IMOEX  CatBoost  LSTM_att  SMAPE  -6.272465  3.553758e-10
IMOEX        RF    sNaive  SMAPE  -4.561738  5.073202e-06
IMOEX     Naive        RF  SMAPE   4.524807  6.045059e-06
IMOEX    Hybrid  LSTM_att  SMAPE  -4.311092  1.624506e-05
IMOEX LSTM_base        RF  SMAPE   4.300208  1.706378e-05
IMOEX  LSTM_att LSTM_base  SMAPE   4.222007  2.421371e-05
IMOEX  LSTM_att     Naive  SMAPE   3.388158  7.036370e-04
IMOEX  CatBoost    sNaive  SMAPE  -3.192931  1.408366e-03
IMOEX  CatBoost     Naive  SMAPE  -3.117383  1.824644e-03
IMOEX  CatBoost LSTM_base  SMAPE  -2.970836  2.969901e-03
IMOEX    Hybrid        RF  SMAPE   2.961463  3.061812e-03
IMOEX  LSTM_att    sNaive  SMAPE   2.866097  4.155668e-03
IMOEX  CatBoost        RF  SMAPE   2.210869  2.704493e-02
IMOEX  CatBoost    Hybrid  SMAPE  -2.102524  3.550736e-02
IMOEX    Hybrid    sNaive  SMAPE  -0.591761  5.540108e-01
IMOEX    Hybrid LSTM_base  SMAPE  -0.548379  5.834318e-01
IMOEX     Naive    sNaive  SMAPE  -0.481439  6.302047e-01
IMOEX LSTM_base     Naive  SMAPE   0.366950  7.136565e-01
IMOEX    Hybrid     Naive  SMAPE  -0.167320  8.671182e-01
IMOEX LSTM_base    sNaive  SMAPE  -0.105938  9.156313e-01
IMOEX  LSTM_att   SARIMAX   WAPE   7.799501  6.215242e-15
IMOEX  CatBoost   SARIMAX   WAPE   6.913784  4.718937e-12
IMOEX        RF   SARIMAX   WAPE   6.517726  7.138117e-11
IMOEX LSTM_base   SARIMAX   WAPE   6.393015  1.626460e-10
IMOEX     Naive   SARIMAX   WAPE   5.519312  3.403296e-08
IMOEX    Hybrid   SARIMAX   WAPE   5.339441  9.323377e-08
IMOEX   SARIMAX    sNaive   WAPE  -4.307733  1.649361e-05
IMOEX     Naive        RF   WAPE   4.111222  3.935700e-05
IMOEX    Hybrid    sNaive   WAPE  -3.510449  4.473500e-04
IMOEX  CatBoost    sNaive   WAPE  -3.440888  5.798094e-04
IMOEX        RF    sNaive   WAPE  -3.383281  7.162537e-04
IMOEX  CatBoost     Naive   WAPE  -3.372784  7.441228e-04
IMOEX  LSTM_att    sNaive   WAPE  -3.144216  1.665325e-03
IMOEX LSTM_base    sNaive   WAPE  -3.093870  1.975638e-03
IMOEX  LSTM_att     Naive   WAPE  -2.680059  7.360913e-03
IMOEX    Hybrid     Naive   WAPE  -2.677416  7.419254e-03
IMOEX LSTM_base     Naive   WAPE  -2.467872  1.359189e-02
IMOEX  CatBoost LSTM_base   WAPE  -2.161418  3.066304e-02
IMOEX  LSTM_att        RF   WAPE   2.073719  3.810545e-02
IMOEX LSTM_base        RF   WAPE   2.034712  4.187981e-02
IMOEX  CatBoost  LSTM_att   WAPE  -1.707009  8.782034e-02
IMOEX     Naive    sNaive   WAPE  -1.518396  1.289145e-01
IMOEX    Hybrid        RF   WAPE   1.398440  1.619809e-01
IMOEX  CatBoost    Hybrid   WAPE  -1.185612  2.357754e-01
IMOEX  CatBoost        RF   WAPE   0.866525  3.862024e-01
IMOEX    Hybrid LSTM_base   WAPE  -0.804339  4.212013e-01
IMOEX    Hybrid  LSTM_att   WAPE  -0.624104  5.325591e-01
IMOEX  LSTM_att LSTM_base   WAPE  -0.349054  7.270485e-01
```

## Анализ важности факторов

Средняя значимость признаков усреднена по фолдам и моделям (RF, CatBoost). Список ограничен десятью ключевыми факторами.

```
  Feature  Importance
   StochK    0.163360
 WILLR_14    0.156843
    Value    0.144700
   StochD    0.132171
   CCI_20    0.036378
   RSI_50    0.017723
MACD_DIFF    0.017460
   ROC_10    0.016438
     MACD    0.015675
   ATR_50    0.013782
```

![feature_importance_top5_overall](outputs/reports/feature_importance_top5_overall.png)

![feature_importance_top5_CatBoost](outputs/reports/feature_importance_top5_CatBoost.png)

![feature_importance_top5_RF](outputs/reports/feature_importance_top5_RF.png)

## Временные ряды топ‑5 индикаторов

![Value](outputs/reports/indicator_series_Value.png)

![StochK](outputs/reports/indicator_series_StochK.png)

![WILLR_14](outputs/reports/indicator_series_WILLR_14.png)

![StochD](outputs/reports/indicator_series_StochD.png)

![CCI_20](outputs/reports/indicator_series_CCI_20.png)

![RSI_50](outputs/reports/indicator_series_RSI_50.png)

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

![IMOEX_full_LSTM_att.png](outputs_logret/reports/IMOEX_full_LSTM_att.png)

![IMOEX_full_CatBoost.png](outputs_logret/reports/IMOEX_full_CatBoost.png)

![IMOEX_full_Hybrid.png](outputs_logret/reports/IMOEX_full_Hybrid.png)

![IMOEX_full_SARIMAX.png](outputs_logret/reports/IMOEX_full_SARIMAX.png)

## Кластеризация режимов по тикерам

```
   Tk  k  silhouette  n_points
IMOEX  5    0.318751      3262
```

![IMOEX_clusters_price.png](outputs/clustering/IMOEX_clusters_price.png)

![IMOEX_cluster_pie.png](outputs/clustering/IMOEX_cluster_pie.png)


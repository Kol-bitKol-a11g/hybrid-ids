# Методи: дані, ознаки, моделі

## 1 Набори даних

| Параметр | CICIDS2017 | NSL-KDD | UNSW-NB15 |
|---|---|---|---|
| Рік створення | 2017 | 2009 | 2015 |
| Кількість записів | 2 830 743 | 148 517 | 257 673 |
| Вхідних ознак | 78 | 41 | 49 |
| Категорій атак | 13 (+ норма) | 4 | 9 |
| Мітка часу | так | ні | ні |
| Поділ 70/15/15 | хронологічний | груповий | груповий |
| Призначення | основний набір | порівняння з літературою | незалежна перевірка |

## 2 Очищення CICIDS2017

| Етап | До | Після | Видалено |
|---|---|---|---|
| Видалення дублікатів | 2 830 743 | 2 746 892 | 83 851 |
| Обробка NaN/Inf | 2 746 892 | 2 741 308 | 5 584 |
| Видалення некоректних значень | 2 741 308 | 2 739 527 | 1 781 |
| Видалення надто малих класів | 2 739 527 | 2 739 516 | 11 |

Реалізація: `data_preparation.clean_flows`. Розміри вікна TCP (`Init_Win_bytes_*`) можуть дорівнювати −1 і не перевіряються на від'ємність.

## 3 Простір ознак

22 ознаки CICFlowMeter, відібрані з 78 (`config.FEATURES`):

| № | Ознака | № | Ознака |
|---|---|---|---|
| 1 | Flow Duration | 12 | Packet Length Mean |
| 2 | Total Fwd Packets | 13 | Packet Length Std |
| 3 | Total Backward Packets | 14 | Packet Length Variance |
| 4 | Total Length of Fwd Packets | 15 | Max Packet Length |
| 5 | Total Length of Bwd Packets | 16 | Avg Bwd Segment Size |
| 6 | Fwd Packet Length Max | 17 | Average Packet Size |
| 7 | Fwd Packet Length Mean | 18 | Flow Bytes/s |
| 8 | Bwd Packet Length Max | 19 | Flow Packets/s |
| 9 | Bwd Packet Length Mean | 20 | Init_Win_bytes_forward |
| 10 | Bwd Packet Length Min | 21 | Init_Win_bytes_backward |
| 11 | Bwd Packet Length Std | 22 | Down/Up Ratio |

Поля `Flow ID`, адреси, порти й `Timestamp` до вектора ознак не входять: вони ідентифікують конкретні вузли й сценарії запису та спричиняють витік інформації. Їх використовують лише для зіставлення з подіями Suricata і формування оповіщень.

## 4 Балансування класів

Класи з часткою менше 1 % ділять на малочисельні (0,1–1 %) і вкрай малочисельні (менше 0,1 %). SMOTE доповнює кожну групу до десятикратного розміру найбільшого класу групи; нормальний трафік і численні класи атак не змінюються. Додатково застосовуються ваги прикладів, обернено пропорційні частоті класу. Балансування виконується лише на навчальній частині (`data_preparation.smote_targets`, `balance_train`).

## 5 Гіперпараметри

**XGBoost** (`config.XGB_PARAMS`): `n_estimators = 500`, `max_depth = 8`, `learning_rate = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.7`, `reg_lambda = 1.0`, `reg_alpha = 0.5`, `min_child_weight = 5`, `objective = multi:softprob`, `tree_method = hist`.

**Автоенкодер** (`config.AE_PARAMS`): шари 22-128-64-16-[4]-16-64-128-22, активація ReLU, вихідний шар лінійний, dropout 0,2 в енкодері, L2 = 1e-5, Adam (lr = 1e-3), batch 512, до 50 епох з ранньою зупинкою (patience = 5), функція втрат MSE. Поріг $\tau_{base}$ обирається за максимумом F1 на валідаційній частині.

**Random Forest** (`config.RF_PARAMS`): 200 дерев, `max_depth = 32`, `min_samples_leaf = 5`, `max_features = sqrt`, `class_weight = balanced_subsample`.

**Контрольні моделі** (`baselines.py`): SVM з ядром RBF (`C = 1`), DNN 256-128-64-32 (dropout 0,3), LSTM 128-64 з вікном 10 потоків, Isolation Forest (200 дерев), One-Class SVM (`nu = 0.1`).

## 6 Відтворюваність

Усі випадкові процеси використовують `random_state = 42`. Параметри нормалізації, відбір ознак і пороги оцінюються на навчальній і валідаційній частинах; тестова частина використовується лише після фіксації конфігурації.

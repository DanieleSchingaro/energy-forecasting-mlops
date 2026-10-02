## Risultati per orizzonte

Test: ultimi 12 mesi, da 2009-11-26 21:00:00. Ore valutate da 7697 a 7789 su 8760 (copertura 87.9%-88.9%; le ore scartate sono quelle con finestra di 168 ore incompleta).

| h | riferimento | MAE naive | Random Forest | XGBoost | migliore | miglioramento |
|---:|---|---:|---:|---:|---|---:|
| 1 | naive_1h | 0.4131 | 0.3430 | 0.3530 | Random Forest | +17.0% |
| 2 | naive_24h | 0.5730 | — | 0.4332 | XGBoost | +24.4% |
| 3 | naive_24h | 0.5731 | — | 0.4591 | XGBoost | +19.9% |
| 4 | naive_24h | 0.5730 | — | 0.4716 | XGBoost | +17.7% |
| 5 | naive_24h | 0.5729 | — | 0.4685 | XGBoost | +18.2% |
| 6 | naive_24h | 0.5728 | — | 0.4700 | XGBoost | +17.9% |
| 7 | naive_24h | 0.5727 | — | 0.4675 | XGBoost | +18.4% |
| 8 | naive_24h | 0.5726 | — | 0.4704 | XGBoost | +17.8% |
| 9 | naive_24h | 0.5727 | — | 0.4644 | XGBoost | +18.9% |
| 10 | naive_24h | 0.5729 | — | 0.4660 | XGBoost | +18.7% |
| 11 | naive_24h | 0.5730 | — | 0.4656 | XGBoost | +18.7% |
| 12 | naive_24h | 0.5731 | — | 0.4717 | XGBoost | +17.7% |
| 13 | naive_24h | 0.5730 | — | 0.4709 | XGBoost | +17.8% |
| 14 | naive_24h | 0.5731 | — | 0.4716 | XGBoost | +17.7% |
| 15 | naive_24h | 0.5732 | — | 0.4724 | XGBoost | +17.6% |
| 16 | naive_24h | 0.5734 | — | 0.4700 | XGBoost | +18.0% |
| 17 | naive_24h | 0.5733 | — | 0.4705 | XGBoost | +17.9% |
| 18 | naive_24h | 0.5732 | — | 0.4697 | XGBoost | +18.1% |
| 19 | naive_24h | 0.5731 | — | 0.4708 | XGBoost | +17.8% |
| 20 | naive_24h | 0.5728 | — | 0.4646 | XGBoost | +18.9% |
| 21 | naive_24h | 0.5727 | — | 0.4628 | XGBoost | +19.2% |
| 22 | naive_24h | 0.5728 | — | 0.4570 | XGBoost | +20.2% |
| 23 | naive_24h | 0.5728 | — | 0.4628 | XGBoost | +19.2% |
| 24 | naive_24h | 0.5729 | 0.4611 | 0.4605 | XGBoost | +19.6% |

MAE in kW. Il riferimento e' la baseline naive piu' forte dell'orizzonte: la persistenza a un'ora, la stagionalita' giornaliera piu' avanti, dove la persistenza non e' piu' disponibile al momento della previsione.

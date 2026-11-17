# final_metrica

**TỐT HƠN baseline**: spellF1 dev/select 0.6691 ± 0.0125; so baseline +0.0754 [+0.0701, +0.0806], thắng 5/5 bản

```
model          spellF1           macroF1           d spellF1 vs baseline [95% CI] wins      d macroF1
baseline       0.5938 ± 0.0105   0.8692 ± 0.0036   (mốc)                        -         -
final_metrica  0.6691 ± 0.0125   0.9065 ± 0.0019   +0.0754 [+0.0701, +0.0806]   5/5       +0.0373
smooth50       0.2113 ± 0.0062   0.4601 ± 0.0076   -0.3825 [-0.3990, -0.3660]   0/5       -0.4091
```

| bản | baseline | final_metrica | smooth50 |
|---|---|---|---|
| DFL-MAT-J03WN1_r00 | 0.6081 | 0.6878 | 0.2084 |
| DFL-MAT-J03WN1_r01 | 0.5937 | 0.6698 | 0.2060 |
| DFL-MAT-J03WN1_r02 | 0.5967 | 0.6656 | 0.2205 |
| DFL-MAT-J03WN1_r03 | 0.5789 | 0.6529 | 0.2149 |
| DFL-MAT-J03WN1_r04 | 0.5915 | 0.6695 | 0.2067 |

## Lỗi theo tình huống (trung bình các bản)

```
situation        % in-play   acc P(H)  acc codes   acc codes|H/A  % team errors
in_play             100.0%      89.6%      86.6%           90.1%         100.0%
ball_seen            64.7%      91.1%      90.5%           91.4%          57.4%
ball_unseen          35.3%      86.9%      79.4%           87.3%          42.6%
near_change          27.8%      77.8%      73.7%           76.6%          65.6%
far_from_change      72.2%      94.2%      91.6%           95.3%          34.4%
ball_fast            38.8%      92.1%      90.3%           92.6%          29.2%
ball_slow            61.2%      88.1%      84.2%           88.4%          70.8%
half_1               56.7%      91.9%      88.1%           92.0%          45.7%
half_2               43.3%      86.6%      84.6%           87.6%          54.3%
spellF1 0.6691  macroF1 0.9065  frame acc  92.5%
bóng sống -> D   3.9% | D -> bóng sống   1.9% | sai đội   9.6% (trên khung bóng sống)
lượt kiểm soát đoán / thật: 391.8 / 500.0 | lần đổi đội H<->A đoán / thật: 354.0 / 467.0
```

## Chạy

- commit `08bb3def1` (HEAD), 2026-11-16T00:51:47+00:00, thiết bị cuda:Tesla T4
- step: feature.nap1_dead, feature.nap1_pitch, loss.near_change, feature.contest_geometry, feature.ball_touch, decode.wide
- s1: 10 epoch, tốt nhất 0.6359, khởi tạo exp/baseline/ckpt/s1/ep005.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- s2: 10 epoch, tốt nhất 0.6481, khởi tạo exp/baseline/ckpt/s2/ep003.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- s3: 10 epoch, tốt nhất 0.6520, khởi tạo exp/baseline/ckpt/s3/ep010.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- gộp 9 checkpoint, decode_switch 0.6691, tham số {'scale': 0.02, 'floor': 1e-14, 'cap': 0.5, 'gate_scale': 0.02, 'window': 3, 'lambda_emit': 0.5, 'L_alive': 7.29, 'L_dead': 24.5, 'share_dead': 0.234, 'p_switch_after_dead': 0.693, 'eps_gate': 0.01, 'deadball_switch_gate': True}
- thời gian: train 67.4 phút, ensemble 11.7 phút, blend 0.0 phút

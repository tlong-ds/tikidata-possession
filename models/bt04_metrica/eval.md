# bt04_metrica

**TỐT HƠN baseline**: spellF1 dev/select 0.6605 ± 0.0102; so baseline +0.0667 [+0.0605, +0.0730], thắng 5/5 bản

```
model          spellF1           macroF1           d spellF1 vs baseline [95% CI] wins      d macroF1
baseline       0.5938 ± 0.0105   0.8692 ± 0.0036   (mốc)                        -         -
bt04_metrica   0.6605 ± 0.0102   0.9011 ± 0.0027   +0.0667 [+0.0605, +0.0730]   5/5       +0.0319
smooth50       0.2113 ± 0.0062   0.4601 ± 0.0076   -0.3825 [-0.3990, -0.3660]   0/5       -0.4091
```

| bản | baseline | bt04_metrica | smooth50 |
|---|---|---|---|
| DFL-MAT-J03WN1_r00 | 0.6081 | 0.6745 | 0.2084 |
| DFL-MAT-J03WN1_r01 | 0.5937 | 0.6646 | 0.2060 |
| DFL-MAT-J03WN1_r02 | 0.5967 | 0.6550 | 0.2205 |
| DFL-MAT-J03WN1_r03 | 0.5789 | 0.6474 | 0.2149 |
| DFL-MAT-J03WN1_r04 | 0.5915 | 0.6610 | 0.2067 |

## Lỗi theo tình huống (trung bình các bản)

```
situation        % in-play   acc P(H)  acc codes   acc codes|H/A  % team errors
in_play             100.0%      89.3%      85.7%           89.8%         100.0%
ball_seen            64.7%      90.6%      89.2%           91.1%          58.3%
ball_unseen          35.3%      86.9%      79.2%           87.3%          41.7%
near_change          27.8%      77.2%      73.1%           76.2%          65.4%
far_from_change      72.2%      93.9%      90.6%           95.1%          34.6%
ball_fast            38.8%      91.9%      90.1%           92.4%          29.4%
ball_slow            61.2%      87.6%      82.9%           88.1%          70.6%
half_1               56.7%      91.7%      87.5%           91.8%          45.6%
half_2               43.3%      86.1%      83.3%           87.2%          54.4%
spellF1 0.6605  macroF1 0.9011  frame acc  92.0%
bóng sống -> D   4.6% | D -> bóng sống   2.0% | sai đội   9.7% (trên khung bóng sống)
lượt kiểm soát đoán / thật: 390.4 / 500.0 | lần đổi đội H<->A đoán / thật: 349.8 / 467.0
```

## Chạy

- commit `64437f85a` (HEAD), 2026-11-15T12:23:38+00:00, thiết bị cuda:Tesla T4
- step: feature.nap1_dead, feature.nap1_pitch, loss.near_change, feature.contest_geometry, feature.ball_touch, decode.wide
- s1: 10 epoch, tốt nhất 0.6234, khởi tạo exp/baseline/ckpt/s1/ep005.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- s2: 10 epoch, tốt nhất 0.6350, khởi tạo exp/baseline/ckpt/s2/ep003.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- s3: 10 epoch, tốt nhất 0.6391, khởi tạo exp/baseline/ckpt/s3/ep010.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 2882/177546); gđ2 7 ep lr 0.0003 đóng băng không (học 177546/177546)
- gộp 9 checkpoint, decode_switch 0.6605, tham số {'scale': 0.02, 'floor': 1e-14, 'cap': 0.5, 'gate_scale': 0.02, 'window': 3, 'lambda_emit': 0.5, 'L_alive': 7.29, 'L_dead': 24.5, 'share_dead': 0.234, 'p_switch_after_dead': 0.693, 'eps_gate': 0.01, 'deadball_switch_gate': True}
- thời gian: train 58.6 phút, ensemble 15.9 phút, blend 0.0 phút

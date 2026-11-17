# baseline = nap1_pitch (đưa lên 2026-11-14)

# nap1_pitch

**TỐT HƠN baseline**: spellF1 dev/select 0.5938 ± 0.0105; so baseline +0.0075 [+0.0045, +0.0105], thắng 5/5 bản

```
model          spellF1           macroF1           d spellF1 vs baseline [95% CI] wins      d macroF1
baseline       0.5863 ± 0.0094   0.8652 ± 0.0035   (mốc)                        -         -
nap1_pitch     0.5938 ± 0.0105   0.8692 ± 0.0036   +0.0075 [+0.0045, +0.0105]   5/5       +0.0040
smooth50       0.2113 ± 0.0062   0.4601 ± 0.0076   -0.3750 [-0.3892, -0.3608]   0/5       -0.4051
```

| bản | baseline | nap1_pitch | smooth50 |
|---|---|---|---|
| DFL-MAT-J03WN1_r00 | 0.5996 | 0.6081 | 0.2084 |
| DFL-MAT-J03WN1_r01 | 0.5831 | 0.5937 | 0.2060 |
| DFL-MAT-J03WN1_r02 | 0.5908 | 0.5967 | 0.2205 |
| DFL-MAT-J03WN1_r03 | 0.5746 | 0.5789 | 0.2149 |
| DFL-MAT-J03WN1_r04 | 0.5834 | 0.5915 | 0.2067 |

## Lỗi theo tình huống (trung bình các bản)

```
situation        % in-play   acc P(H)  acc codes   acc codes|H/A  % team errors
in_play             100.0%      87.2%      80.5%           87.3%         100.0%
ball_seen            64.7%      87.6%      82.7%           87.7%          64.4%
ball_unseen          35.3%      86.5%      76.4%           86.6%          35.6%
near_change          27.8%      72.3%      67.9%           71.6%          64.1%
far_from_change      72.2%      93.0%      85.4%           93.6%          35.9%
ball_fast            38.8%      90.1%      87.1%           90.4%          30.7%
ball_slow            61.2%      85.4%      76.3%           85.2%          69.3%
half_1               56.7%      89.9%      82.8%           89.8%          45.6%
half_2               43.3%      83.7%      77.5%           84.1%          54.4%
spellF1 0.5938  macroF1 0.8692  frame acc  89.4%
bóng sống -> D   7.8% | D -> bóng sống   2.3% | sai đội  11.7% (trên khung bóng sống)
lượt kiểm soát đoán / thật: 418.6 / 500.0 | lần đổi đội H<->A đoán / thật: 382.4 / 467.0
```

## Chạy

- commit `a7a7252f8` (HEAD), 2026-11-13T18:08:19+00:00, thiết bị cuda:Tesla T4
- step: feature.nap1_dead, feature.nap1_pitch, loss.near_change
- s1: 9 epoch, tốt nhất 0.5733, khởi tạo exp/baseline/ckpt/s1/ep007.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 386/174922); gđ2 7 ep lr 0.0003 đóng băng không (học 174922/174922)
- s2: 7 epoch, tốt nhất 0.5786, khởi tạo exp/baseline/ckpt/s2/ep010.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 386/174922); gđ2 7 ep lr 0.0003 đóng băng không (học 174922/174922)
- s3: 10 epoch, tốt nhất 0.5812, khởi tạo exp/baseline/ckpt/s3/ep010.pt, gđ1 3 ep lr 0.001 đóng băng ['encoder.phi.2', 'temporal', 'had', 'touch', 'team', 'ball'] +cột cũ (học 386/174922); gđ2 7 ep lr 0.0003 đóng băng không (học 174922/174922)
- gộp 9 checkpoint, decode_switch 0.5938, tham số {'scale': 0.05, 'floor': 1e-06, 'cap': 0.5, 'gate_scale': 0.5, 'window': 3, 'lambda_emit': 1.0, 'L_alive': 7.29, 'L_dead': 24.5, 'share_dead': 0.234, 'p_switch_after_dead': 0.693, 'eps_gate': 0.01, 'deadball_switch_gate': True}
- thời gian: train 29.4 phút, ensemble 11.7 phút, blend 0.0 phút

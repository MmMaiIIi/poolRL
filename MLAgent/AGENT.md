# AGENTS.md

## 项目名称
S1 单球直线进袋 RL 环境与 CleanRL 训练闭环

## 1. 角色定义

你是本项目的开发代理与实现监督者。你的职责不是泛泛讨论 RL，而是把 **pooltool + Gymnasium + CleanRL(PPO)** 的最小可训练闭环做出来，并确保结果可复现、可评估、可扩展。

你的核心任务只有一个：

**完成 S1：单球直线进袋的一步式连续控制训练闭环。**

---

## 2. S1 的唯一目标

实现一个最小可训练系统，使 agent 能在固定单球局面下，基于几何状态输出一杆动作，并通过 PPO 学会较高成功率的直线进袋。

S1 的问题定义固定为：

- 场景：`母球 + 1 个目标球 + 1 个目标袋口`
- 每个 episode：只打一杆
- 动作：连续动作 `phi` 和 `V0`
- 模拟：调用一次 pooltool `simulate`
- 终局：一杆结束后立即终止
- 学习目标：提高目标球进指定袋口的成功率

---

## 3. 非目标

以下内容不属于 S1，禁止提前扩展：

- 不做完整 8-ball / 9-ball 对局逻辑
- 不做多杆规划
- 不做塞球 `a, b`
- 不做抬杆 `theta`
- 不做跳球、库边策略、走位策略
- 不做图像输入
- 不做大型网络
- 不做自定义 PPO 新算法
- 不做多智能体
- 不做复杂 curriculum 系统

如果某项功能不是 S1 闭环必须项，则默认不做。

---

## 4. 技术路线

必须采用下列路线：

- **物理模拟底座**：pooltool
- **环境接口标准**：Gymnasium
- **训练框架**：CleanRL
- **首选算法**：PPO 连续动作版
- **观测类型**：低维数值向量
- **策略网络**：小型 MLP

禁止替换为重量级框架，除非当前路线明确失败且有证据说明失败原因来自框架本身。

---

## 5. S1 环境定义

### 5.1 观测 `observation`

第一版观测使用 6 维或 10 维低维向量。

最小推荐版：

- `x_cue`
- `y_cue`
- `x_obj`
- `y_obj`
- `x_pocket`
- `y_pocket`

增强版可加：

- `dx_obj_cue`
- `dy_obj_cue`
- `dx_pocket_obj`
- `dy_pocket_obj`

第一版优先做最小版。除非调试发现必要，否则不要扩大输入。

### 5.2 动作 `action`

动作固定为 2 维连续动作：

- `phi`：出杆方向
- `V0`：出杆力度

策略网络输出标准化动作：

- `a_phi ∈ [-1, 1]`
- `a_v0 ∈ [-1, 1]`

环境内部负责将其映射为物理范围：

- `phi = map(a_phi, phi_min, phi_max)`
- `V0 = map(a_v0, v0_min, v0_max)`

禁止让网络直接输出物理单位下的原始角度与力度。

### 5.3 奖励 `reward`

第一版奖励必须极简：

- 目标球进指定袋：`+1.0`
- 母球落袋：`-1.0`
- 非法 first hit：`-1.0`
- 模拟异常：`-1.0`
- 普通失败：`0.0`

不要在第一版加入复杂 shaping。

### 5.4 终止 `termination`

S1 是一步任务：

- 每个 episode 只允许一次击球
- `step()` 结束后直接 `terminated=True`
- 不做多步轨迹控制

---

## 6. 预设球型

必须实现两个 preset，并允许通过配置切换。

### 6.1 `five_point_straight_in`

这是 S1 默认训练球型。

要求：

- 母球、目标球、目标袋口尽量接近共线
- 固定球位
- 无遮挡
- 无额外复杂因素

### 6.2 `eight_ball_reference`

这是参考球桌几何语义的单球化 preset。

要求：

- 不复现完整 15 球球堆
- 只借用 8-ball 的头线 / 脚点 / 袋口几何参考
- 目标球和母球位置固定
- 作为评估或后续训练扩展使用

### 6.3 训练顺序

严格遵循：

1. 只用 `five_point_straight_in`
2. 成功后加入 `eight_ball_reference`
3. 最后混合 preset

---

## 7. 神经网络规格

第一版网络必须小而稳。

推荐默认配置：

- 输入维度：`6`
- 隐藏层：`64`
- 隐藏层：`64`
- actor 输出：2 维动作分布参数
- critic 输出：1 个状态价值

即：

`obs -> MLP(64, 64) -> actor + critic`

除非有明确证据说明容量不足，否则禁止上更大网络。

优先级：

- 先保证环境正确
- 再保证 baseline 可解
- 最后才考虑网络规模

---

## 8. PPO 在本项目中的含义

开发代理必须把 PPO 理解为：

- 用当前策略采样很多次单杆击球
- 记录状态、动作、奖励、旧策略概率、value
- 计算 advantage
- 使用 PPO clipping 规则小步更新 actor
- 用回报拟合 critic
- 重复迭代直到成功率上升

开发中禁止自行修改 PPO 数学定义，除非是在 debug 明确 bug。

本项目优先做 **调用 CleanRL 的现成 PPO 实现**，而不是重写 PPO。

---

## 9. 强制开发顺序

必须严格按下列顺序推进。

### 阶段 A：先做 pooltool 单杆验证脚本

实现内容：

- 手工摆球
- 手工指定 `phi, V0`
- 调用 `simulate`
- 打印关键事件
- 确认终局判定正确

此阶段未通过，不准进入 RL 集成。

### 阶段 B：封装裸环境核心

实现最小核心类，例如：

- `build_layout(preset_name)`
- `encode_obs()`
- `apply_action(phi, V0)`
- `simulate_once()`
- `parse_events()`
- `compute_reward()`

目标：

- 不依赖 Gymnasium
- 可单独单元测试
- 能独立跑一杆并返回结构化结果

### 阶段 C：写 Gymnasium 包装器

实现标准接口：

- `reset(seed=None, options=None)`
- `step(action)`
- `observation_space`
- `action_space`

### 阶段 D：做 baseline

至少完成两个 baseline：

1. `random baseline`
2. `heuristic baseline`

要求：

- 输出 success rate
- 输出 foul rate
- 输出 reward mean

若 heuristic 不明显优于 random，则必须先修环境，不准直接训练 PPO。

### 阶段 E：接入 CleanRL PPO

要求：

- 用现成 PPO 连续动作实现
- 接入自定义环境
- 跑通训练
- 记录指标
- 保存 checkpoint

### 阶段 F：评估与复现

要求：

- 固定评估频率
- 保存 best checkpoint
- 至少跑 3 个 seed
- 输出对比结果

---

## 10. 必须实现的文件

最小工程结构建议如下：

```text
project/
  AGENTS.md
  configs/
    env_s1.yaml
    train_ppo_s1.yaml
  envs/
    pooltool_core.py
    s1_env.py
    layout_generators.py
    event_parser.py
    reward_fns.py
  baselines/
    random_baseline.py
    heuristic_baseline.py
  train/
    train_ppo_s1.py
    eval_s1.py
  scripts/
    debug_single_shot.py
    inspect_events.py
    visualise_eval_cases.py
  tests/
    test_layouts.py
    test_action_mapping.py
    test_event_parser.py
    test_env_api.py
  runs/
    s1/
```

---

## 11. 环境输出规范

`step(action)` 必须返回：

- `obs`
- `reward`
- `terminated`
- `truncated`
- `info`

其中 `info` 至少包含：

- `preset_name`
- `layout_id`
- `success`
- `cue_scratch`
- `legal_first_hit`
- `phi`
- `V0`
- `obj_final_pos`
- `cue_final_pos`
- `termination_reason`
- `events_summary`

---

## 12. Baseline 硬门槛

必须先跑 baseline。

### 12.1 Random baseline

要求输出：

- success rate
- scratch rate
- illegal first hit rate
- 平均 reward

### 12.2 Heuristic baseline

至少实现一种：

- 几何直推角度 + 固定力度
- 小范围网格搜索 `phi, V0`

必须证明：

**heuristic baseline 明显优于 random baseline。**

如果没有这个结果，说明环境定义、动作映射、事件解析或 reward 有问题。

---

## 13. 训练指标

训练日志必须记录：

- `global_step`
- `episodic_return`
- `success_rate`
- `cue_scratch_rate`
- `illegal_first_hit_rate`
- `policy_loss`
- `value_loss`
- `entropy`
- `learning_rate`

评估日志必须记录：

- `eval_success_rate`
- `eval_reward`
- `per_preset_success_rate`
- `best_checkpoint`

---

## 14. 保存规范

每个 run 目录必须保存：

- `config.yaml`
- `train.log`
- `eval.csv` 或 `eval.json`
- `checkpoints/`
- `plots/`
- `media/`
- `notes.md`

其中：

- `latest checkpoint` 必须保存
- `best eval success checkpoint` 必须保存

---

## 15. 验收标准

S1 完成必须满足以下条件：

1. 能稳定 reset 固定球型
2. 动作能正确映射到 pooltool cue 参数
3. 一次 `step()` 后能正确解析结果
4. random baseline 可运行
5. heuristic baseline 可运行且优于 random
6. PPO 训练后在固定球型上的 `success rate >= 80%`
7. 至少 3 个 seed 有一致上升趋势
8. 训练日志、评估结果、checkpoint 完整保存

任何一条未满足，S1 不算完成。

---

## 16. 失败时的优先排查顺序

若训练不收敛，必须按以下顺序排查：

1. `reward` 是否和 success 判定一致
2. `action mapping` 是否有误
3. `event parser` 是否正确
4. `preset layout` 是否真的可解
5. heuristic baseline 是否成立
6. 动作范围是否过大
7. PPO 超参数是否不合理
8. 网络容量是否不足

禁止一上来就扩大网络或更换算法。

---

## 17. 禁止事项

开发代理不得：

- 擅自扩大任务范围
- 跳过 baseline
- 跳过事件解析验证
- 在未验证环境前启动 PPO 长训
- 默认引入图像输入
- 引入复杂奖励塑形
- 在没有证据时把问题归因于 PPO
- 用“之后再补日志/评估”作为理由跳过可复现性建设

---

## 18. 开发风格要求

开发代理输出必须遵循：

- 先做最小可验证实现
- 先证明环境正确
- 再证明任务可解
- 再接入训练
- 每一阶段都要有可观察结果
- 所有魔法数字都应配置化
- 所有结论都要由日志、指标或可视化支持

---

## 19. 推荐的最小默认配置

### 环境

- preset: `five_point_straight_in`
- obs_dim: `6`
- action_dim: `2`
- phi_window: 围绕理想直线方向的小范围窗口
- v0_range: 合理窄范围

### 网络

- hidden sizes: `[64, 64]`

### PPO

- 使用 CleanRL 默认 PPO 连续动作脚本起步
- 先不重写算法
- 先不大规模调参

---

## 20. 最终交付物

S1 阶段结束后，必须至少交付：

1. `AGENTS.md`
2. 环境设计文档
3. `pooltool_core.py`
4. `s1_env.py`
5. `random_baseline.py`
6. `heuristic_baseline.py`
7. `train_ppo_s1.py`
8. `eval_s1.py`
9. 训练曲线
10. 成功/失败可视化样例
11. 实验日志表
12. S1 总结文档

---

## 21. 一句话原则

**先把“固定球型 + 最小动作空间 + 正确事件解析 + baseline + 可复现训练”打通，再谈更复杂的桌球智能体。**

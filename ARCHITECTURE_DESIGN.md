# BetaGoAgent 架构设计方案 (v1.0)

## 1. 项目愿景
打造一个**AI原生**的围棋智能体开发平台。它不仅仅是一个围棋程序，更是一个**可视化的强化学习实验室**。
- **智能 (Intelligent)**: 采用 AlphaZero 核心算法，具备自我进化能力。
- **可视化 (Visual)**: 实时展示 AI 的"思考过程"（MCTS 搜索树）、训练进度、胜率变化。
- **专业 (Professional)**: 采用异步 Batch 推理、经验回放等高阶 RL 技巧，对标开源界最强实现（如 KataGo/MiniGo 的架构思想）。
- **柔性 (Flexible)**: 模块化设计，轻松替换神经网络结构、棋盘尺寸或训练参数。

---

## 2. 核心架构图

系统采用 **生产者-消费者** 模型，分为三大独立进程组：

\\\mermaid
graph TD
    subgraph "Core Learning Loop (核心闭环)"
        SP[Self-Play Workers] -->|生成对局数据| RB(Replay Buffer)
        RB -->|采样 Batch| TR[Trainer Process]
        TR -->|更新权重| MS[Model Storage]
        MS -->|同步最新模型| SP
        MS -->|同步最新模型| EV[Evaluator]
    end

    subgraph "Inference Engine (推理引擎)"
        SP -- 局面 State --> INF[Inference Server]
        INF -- Policy/Value --> SP
        note[GPU Batch 推理\n极大提升吞吐量] -.-> INF
    end

    subgraph "Visualization & Interaction (前端交互)"
        UI[Web UI / Dashboard] <-->|WebSocket| API[API Server]
        API <--> MS
        API <--> SP
    end
\\\`r

## 3. 关键技术模块

### 3.1  Brain: 神经网络 (PyTorch)
不再使用简单的 CNN，而是采用现代架构：
- **Backbone**: ResNet-18 或 MobileNetV3 (兼顾性能与速度)。
- **Input Features**: 
  - \[Current Board]\ (当前盘面)
  - \[History -1]\ (上一步)
  - \[History -2]\ (上两步 - 用于识别打劫)
  - \[Color Plane]\ (黑白方指示)
- **Heads**: 
  - Policy Head (落子概率)
  - Value Head (胜率预估)

### 3.2  Engine: 异步 MCTS 推理
旧系统的最大瓶颈在于串行推理。新系统将实现 **Async Batch MCTS**：
1. **多线程搜索**: 多个 MCTS 搜索线程并行运行。
2. **虚拟损失 (Virtual Loss)**: 防止多线程探索同一分支。
3. **全局推理队列**: 所有线程遇到未展开节点时，不立即推理，而是将请求推入队列。
4. **Batch Execution**: 独立的 GPU 线程从队列取出 16/32 个请求，合并成一个 Tensor 进行一次推理，再分发结果。
   - **预期提升**: 10-50 倍推理速度。

### 3.3  Trainer: 持续进化机制
- **Experience Replay (经验回放)**: 维护一个 \maxlen=50000\ 的缓冲池，防止灾难性遗忘。
- **Pipelined Training**: 训练与自博弈完全解耦，训练进程不阻塞自博弈。
- **Snapshot & Evaluation**: 每训练 N 步自动保存快照，并启动后台评估进程与旧模型对战。

### 3.4  Dashboard: 可视化中控台
基于 **FastAPI + WebSocket + Modern Frontend (Vue/React)**：
- **实时监控**: Loss 曲线、Elo 分数趋势。
- **思考可视化**: 在棋盘上通过热力图显示 AI 的 Policy 分布，通过箭头显示 MCTS 的主要变例 (Principal Variation)。
- **人机对弈**: 随时可以打断训练，亲自上手测试 AI 棋力。

---

## 4. 目录结构规划

\\\`r
BetaGoAgent/
 core/               # 核心算法库 (不依赖框架)
    go_engine.py    # 纯粹的围棋规则 (Numpy优化)
   ├ mcts.py         # 蒙特卡洛树搜索
    datatype.py     # 基础数据结构
 brain/              # 神经网络相关
   ─ network.py      # PyTorch 模型定义
    inference.py    # 异步推理服务
 learning/           # 训练循环
    self_play.py    # 自博弈 Worker
    trainer.py      # 训练主循环
    buffer.py       # 经验回放池
 web/                # 可视化系统
    backend/        # FastAPI 服务端
    frontend/       # 前端界面
 config.py           # 全局配置
 main.py             # 统一入口
\\\`r

## 5. 实施路线图 (Roadmap)

1.  **Phase 1: 骨架搭建 (Skeleton)**
    - 实现高效的 \GoEngine\ (规则) 和 \Network\ (模型)。
    - 搭建 \Self-Play\ -> \Buffer\ -> \Trainer\ 的最小闭环。
2.  **Phase 2: 性能注入 (Performance)**
    - 实现 Batch MCTS 推理。
    - 引入多进程并行架构。
3.  **Phase 3: 赋予视觉 (Visualization)**
    - 搭建 Web Dashboard，实现实时训练监控。
4.  **Phase 4: 进化 (Evolution)**
    - 长期挂机训练，调整超参数，冲击业余段位。


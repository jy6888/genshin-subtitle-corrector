# subtitle-corrector

**POST-ASR 中文字幕纠错引擎** —— 面向游戏攻略视频的术语级字幕修正系统。

语音识别（ASR）经常将游戏专有名词（角色名、武器名、技能名）误识别为同音错别字。本系统接收 ASR 生成的 SRT/VTT/ASS 字幕，通过多层检测 + LLM 仲裁的管线，精准修正术语错误，同时保持句式、时间轴和说话风格完全不变。

## 核心亮点

### 多层级检测架构

系统采用 6 阶段流水线，大部分字幕行在前几阶段就被判定为无错，**仅约 15% 的候选需要调用 LLM**：

```
输入字幕 → 预处理(归一化+实体预激活)
  → Phase 1: jieba分词检测 + 统计语言模型异常发现 + 并发LLM语义过滤
  → Reducer: 语义聚类 + 话题标签
  → CandidateExpansion: 实体一致性 + 长实体变体 + 本地术语补全
  → Phase 2: 按cluster批处理LLM三步判决 + typed REQUERY二轮修正
  → Phase 6: 右到左多点替换 → 输出修正字幕
```

### 实体记忆与自适应衰减

系统维护一个动态实体记忆，追踪视频当前正在讨论的角色/武器。实体被 LLM 确认后激活，衰减速率随刷新次数递减（5次后锁定），无关实体在约5个cue后被遗忘。这让修正决策具备**上下文感知能力**。

### 冷热池惰性激活

9个热池类别（角色、武器、圣遗物等）始终活跃；9个冷池类别（食物、材料、成就等）默认休眠，每cue仅发送1个探测候选。当 LLM 选中冷池术语时，该类别被激活（带衰减）。实现了**按需加载**，避免候选池膨胀。

### typed REQUERY 二轮修正

Phase 2 LLM 可发出 3 种 REQUERY 请求：
- `terminology_phonetic` — 标准术语拼音匹配
- `asr_alias` — ASR 别名表查找
- `team_comp_alias` — 配队简称修复（代码层解析）

每种类型有独立的解析路径和约束，形成**代码层 + LLM 协作的闭环修正**。

### FN 分层诊断

评估模块自动将每个漏修（False Negative）分类到根因阶段：检测器漏检、候选错误、Phase2 拒绝、REQUERY 失败等。配合 per-cue trace 表，精确定位管线瓶颈。

## 技术栈

| 组件 | 技术 |
|------|------|
| 分词 | jieba |
| 模糊匹配 | RapidFuzz |
| 拼音 | pypinyin |
| 数据模型 | Pydantic v2 |
| 字幕解析 | pysubs2 |
| LLM | OpenAI 兼容协议 |
| 持久化 | SQLite |
| CLI | typer |
| 日志 | loguru |

## 快速开始

### 安装

```bash
conda create -n subtitle-corrector python=3.11 -y
conda activate subtitle-corrector
pip install -e ".[dev]"
```

### 配置

在项目根目录创建 `.env`：

```dotenv
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://你的API地址/v1
```

### 使用

```bash
# 初始化数据库 & 导入术语库
subtitle-corrector init-db
subtitle-corrector import-terms terms.csv

# 运行纠错（自动生成 Markdown 报告）
subtitle-corrector inspect input.srt

# 运行纠错 + 对比金标评估
subtitle-corrector evaluate input.srt -g ground_truth.srt --trace
```

## 项目结构

```
src/subtitle_corrector/
├── cli.py                     # CLI 入口 (typer)
├── schemas.py                 # Pydantic 数据模型
├── config/settings.py         # YAML 配置加载
├── subtitle/parser.py         # SRT/VTT/ASS 解析
├── normalize/text.py          # 文本归一化
├── pinyin/converter.py        # 拼音转换与相似度
├── memory/
│   ├── entity.py              # 实体记忆管理 (自适应衰减)
│   ├── category_activation.py # 冷热池分类激活
│   ├── correction_reuse.py    # 已确认修正规则复用
│   └── sqlite.py              # SQLite CRUD
├── matcher/
│   ├── terminology.py         # 模糊术语匹配 + 实体加成
│   └── retriever.py           # 术语检索
├── detector/
│   ├── jieba_span.py          # jieba 分词检测器 (主)
│   ├── language_model.py      # 统计语言模型异常检测
│   └── pinyin.py / entity.py / terminology.py
├── aliasing/
│   ├── populator.py           # 术语内部简称自动生成
│   ├── asr_alias.py           # ASR 别名审核态管理
│   └── runtime.py             # 别名运行时匹配
├── character_alias/
│   ├── lexicon.py             # 角色别名加载
│   └── team_comp.py           # 配队单字序列解析
├── spoken_alias/
│   └── lexicon.py             # 口语别名加载与匹配
├── normal_lm/
│   └── model.py               # 统计语言模型 + 近音别名
├── pipeline/
│   ├── discovery.py           # Phase 1: 检测 + 并发LLM过滤
│   ├── reducer.py             # 语义聚类
│   ├── candidate_expansion.py # 候选扩展 (3个expander)
│   ├── refinement.py          # Phase 2: 批处理LLM + REQUERY
│   ├── requery.py             # REQUERY 候选生成引擎
│   ├── commit.py              # Phase 6: EditPlan + 多点替换
│   └── factory.py             # 管线构建
├── resolver/
│   ├── llm.py                 # LLM 仲裁器
│   ├── pass1_prompt.py        # Phase 1 prompt (三步判决法)
│   └── pass2_prompt.py        # Phase 2 prompt (三步判决法 + REQUERY)
├── evaluation/
│   ├── fn_attribution.py      # FN 归因 (7-bucket分类)
│   ├── fn_trace.py            # FN 分层诊断
│   └── metrics.py             # F1/Precision/Recall
└── report/markdown.py         # Markdown 评估报告生成
```

## 配置

`configs/default.yaml` 控制所有管线参数，包括实体记忆衰减率、冷热池分类、匹配阈值、LLM 模型选择等。

## 许可

MIT

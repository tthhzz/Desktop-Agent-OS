# Embodied Agent OS — 具身智能体与多模态交互平台

基于 LangGraph 的多模态自主智能体系统。构建了"感知-记忆-决策-执行"的完整闭环，通过多节点编排、分层记忆与 MCP 工具链集成，实现具备系统级操控与持续学习能力的桌面级具身智能。

## 架构概览

```
用户输入 (语音/文字/视觉)
        │
        ▼
┌─────────────────────────────────────────────┐
│           LangGraph Multi-Agent              │
│                                              │
│  memory_retrieval → supervisor → chat/tools │
│       ↑                    │         │       │
│       │     ┌──────────────┘         │       │
│       │     ▼                        ▼       │
│  Memory System              MCP Tool Chain  │
│  ┌─────────────────┐    ┌─────────────────┐ │
│  │Sensory (感知)    │    │ Terminal (终端)  │ │
│  │Working (工作)    │    │ Computer (桌面) │ │
│  │Episodic (情节)   │    │ Playwright (浏览器)│ │
│  │Semantic (语义)   │    │ Memory (记忆)   │ │
│  │Skill (技能)      │    │ Search (搜索)   │ │
│  └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────┘
        │
        ▼
  Live2D + TTS 输出
```

## 核心特性

### 🧠 分层记忆与 Skill 沉淀
5 层认知记忆架构（Sensory → Working → Episodic → Semantic → Skill）：
- **反思层**：从 Episodic 提取 Semantic 知识，支持规则与 LLM 双模式
- **巩固层**：相似知识合并、冷记忆衰减、热记忆提升
- **Skill 挖掘**：高频工具链模式自动沉淀为可复用 Skill

### 🔧 系统级工具调度编排
基于 MCP 协议集成多种系统级控制能力：
- **Terminal**：Shell 命令执行、文件读写、目录浏览
- **Computer**：截屏、鼠标点击、键盘输入、拖拽操作
- **Playwright**：浏览器自动化（导航、截图、表单填写）
- **Memory**：记忆搜索、知识写入、技能查询
- Human-in-the-Loop 审批机制保障高风险操作安全

### 👁️ 多模态感知与具身链路
- VAD 全双工语音感知（silero_vad + sherpa-onnx ASR）
- 视觉语言模型支持图片输入
- Live2D 骨骼驱动 + 表情关键词映射 + 流式口型同步
- Edge-TTS 低延迟语音合成

## 技术栈

| 模块 | 技术 |
|---|---|
| Agent 编排 | LangGraph, LangChain |
| 记忆系统 | SQLite + FTS5, sentence-transformers |
| 工具协议 | MCP (FastMCP) |
| 语音 | sherpa-onnx, silero-vad, Edge-TTS |
| 视觉 | Qwen2.5-VL |
| 桌面操控 | pyautogui, Pillow |
| 浏览器 | Playwright MCP |
| 形象渲染 | Live2D (pixi-live2d-display) |
| Web 服务 | FastAPI, WebSocket, Uvicorn |

## 快速开始

```bash
# 1. 创建环境
conda create -n embodied-agent python=3.10 -y
conda activate embodied-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
# 编辑 conf.yaml 中的 llm_api_key

# 4. 启动
python run_server.py
```

浏览器访问 `http://localhost:12393` 即可使用。

## 项目结构

```
src/open_llm_vtuber/
├── agent/agents/langgraph_agent/   # LangGraph 多 Agent 核心
│   ├── agent.py                    # Agent 主类
│   ├── graph.py                    # StateGraph 拓扑定义
│   ├── state.py                    # AgentState 类型
│   ├── config.py                   # Agent 配置
│   └── nodes/
│       ├── supervisor.py           # Supervisor 路由节点
│       ├── chat_worker.py          # 闲聊 Worker
│       ├── tool_worker.py          # 工具 Worker
│       └── memory_retrieval.py     # 记忆检索节点
├── memory/                         # 5 层认知记忆系统
│   ├── memory_system.py            # 统一记忆管理器
│   ├── memory_mcp_server.py        # 记忆 MCP 服务
│   ├── layers/                     # 5 层记忆实现
│   ├── storage/                    # SQLite 持久化
│   └── evolution/                  # 反思/巩固/Skill挖掘
├── mcp_servers/                    # 自建 MCP 服务
│   ├── terminal_server/            # 终端执行服务
│   └── computer_use_server/        # 桌面自动化服务
└── ...                             # ASR/TTS/Live2D 等基础设施
```

## 致谢

本项目基于 [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)（MIT License）开发，保留了原项目的 ASR/TTS/Live2D/WebSocket 基础设施，新增了 LangGraph 多 Agent 架构、5 层认知记忆系统、MCP 工具编排等核心模块。

## License

MIT License — 详见 [LICENSE](LICENSE)

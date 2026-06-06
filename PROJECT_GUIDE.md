# Desktop-Agent OS — 项目技术详解 & 面试QA

## 一、项目概述

**Desktop-Agent OS** 是一个基于LangGraph的桌面级多模态自主智能体系统。核心能力是让AI Agent"看得懂屏幕、记得住习惯、拿得准工具"——通过VLM视觉链路理解屏幕内容，5层认知记忆实现跨会话知识沉淀，44+MCP工具实现系统级操控，Planner+Reflector闭环实现多步自主执行与结果验证。

**技术栈**：LangGraph, MCP(stdio/JSON-RPC), qwen2.5-vl-72b, pytesseract, sentence-transformers, SQLite+FTS5, FastMCP, pyautogui, Playwright, Silero VAD, Live2D

**基于**：Open-LLM-VTuber开源项目二次开发。原始项目提供对话框架和Live2D渲染，我实现了LangGraph多Agent拓扑、5层记忆系统、MCP工具链、VLM视觉链路、安全模块、SDD规约引擎。

---

## 二、重点模块详解

### 2.1 VLM视觉链路

**问题**：截图返回的base64数据，LLM只能看到乱码文本，无法理解图片内容。

**技术实现**：

截图数据流经3个关键节点，每个节点都需要正确处理多模态内容：

```
截图工具 → tool_worker → tool_planner → LLM(VLM)
```

**tool_worker.py**：执行MCP工具后，`_build_tool_message_content()`检测返回内容：
- 如果是直接base64图片（`data:image/jpeg;base64,...`）→ 构建 `[{"type":"text","text":"截图摘要"}, {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`
- 如果是JSON内嵌图片（screen_capture_and_parse返回的`{"annotated_image":"data:image/..."}`）→ `_extract_image_from_json()`提取图片，剩余JSON作为文本摘要
- 普通文本 → 原样返回

**tool_planner.py**：将ToolMessage转为HumanMessage给LLM时，`_tool_message_to_human_content()`：
- 如果content是list（多模态）→ 保留image_url块，VLM可以直接看到图片
- 如果content是字符串且以`data:image/`开头 → 包装成多模态格式
- 普通文本 → 截断到500字符

**截图优化**：`_encode_image_for_vlm()`在MCP Server端：
- `pyautogui.screenshot()` 截取1920x1080 PNG
- `img.resize((1280, 720), Image.LANCZOS)` 缩放
- `img.save(buf, format="JPEG", quality=75)` JPEG压缩
- base64编码：从2-5MB降到200-400KB

**chat_worker.py / supervisor.py / reflector.py** 也做了相应适配，确保多模态ToolMessage在各节点流转时不会将base64当作文本截断。

### 2.2 分层记忆系统

**5层存储架构**：

| 层 | 存储实现 | 数据结构 | 生命周期 | 写入时机 |
|---|---------|---------|---------|---------|
| Sensory | `deque(maxlen=10)` 内存 | `{source, data, mime_type}` | 当前帧 | 每次感知输入 |
| Working | `List[Dict]` 内存 | `{role, content, metadata}` | 当前会话 | 每轮对话 |
| Episodic | SQLite + FTS5 | `{id, user_input, ai_response, emotion, topics, importance}` | 持久化 | 每轮对话 `on_conversation_turn()` |
| Semantic | SQLite + FTS5 | `{id, content, categories, importance, confidence, access_count}` | 持久化 | Reflector提炼 / 显式写入 |
| Skill | SQLite + FTS5 | `{id, name, description, tools[], params[], template, frequency, validated}` | 持久化 | SkillMiner挖掘 / 手动创建 |

**SQLite统一表结构**：
```sql
CREATE TABLE {table} (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,          -- 完整JSON记录
    created_at TEXT,
    tags TEXT DEFAULT '',
    importance REAL DEFAULT 0.5
);
CREATE VIRTUAL TABLE {table}_fts USING fts5(id, data, tags, content={table});
```

**层间演化**：
- **Episodic → Semantic**：Reflector每N轮取最近10条Episodic，LLM提炼知识写入Semantic。无LLM时规则统计：主题出现≥2次存"User frequently discusses {topic}"，情绪出现≥2次存"User's conversations tend to be {emotion}"
- **Semantic内部**：Consolidator合并相似知识（Jaccard>0.7合并，取长文本+高importance），衰减低访问知识（access_count=0且importance>0.2则-0.1），提升高频知识（access_count≥3则importance+0.05）
- **工具模式 → Skill**：SkillMiner每2个consolidation周期扫描工具调用历史，精确序列出现≥3次→生成参数化Skill→validate→merge→prune

**检索流程**：
```
用户说话 → memory_retrieval_node
  → memory_system.retrieve_context(query)
  → 并行搜Episodic/Semantic/Skill三层
  → 每层：FTS5 MATCH query ORDER BY importance DESC LIMIT top_k
  → FTS5失败fallback到 LIKE '%query%'
  → 结果注入supervisor prompt
```

**FTS5 vs 向量检索**：FTS5是SQLite内置全文搜索引擎，建倒排索引，支持BM25排序。和`LIKE`的区别是FTS5有索引不需要全表扫描。但FTS5是关键词匹配（搜"高兴"找不到"开心"），向量检索能语义匹配。升级路径：PostgreSQL + pgvector，embedding模型(all-MiniLM-L6-v2)生成768维向量，向量召回+全文召回两路并行，RRF融合排序。

### 2.3 MCP工具链

**6个MCP Server，44+工具**：

**computer_use_server**（11工具）— pyautogui + pytesseract + PIL + ctypes：
| 工具 | 底层函数 | 说明 |
|------|---------|------|
| screenshot | `pyautogui.screenshot()` → `_encode_image_for_vlm()` | 截屏→缩放→JPEG→base64 |
| smart_screenshot | 同上 + `ctypes.windll.user32.GetForegroundWindow()` | 活跃窗口截屏+差异检测 |
| click | `pyautogui.click(x, y, button, clicks)` | 坐标点击 |
| type | ASCII: `typewrite()`; 中文: `pyperclip.copy()` + `hotkey('ctrl','v')` | 键盘输入 |
| hotkey | `pyautogui.hotkey(*keys)` | 快捷键组合 |
| scroll | `pyautogui.scroll(amount, x, y)` | 滚轮 |
| move | `pyautogui.moveTo(x, y, duration)` | 鼠标移动 |
| drag | `pyautogui.dragTo(x, y, duration, button)` | 拖拽 |
| screen_info | `pyautogui.size()`, `pyautogui.position()` | 分辨率+鼠标位置 |
| find_on_screen | `pytesseract.image_to_data(img, output_type=Output.DICT)` | OCR文字→坐标 |
| click_by_text | find_on_screen → `pyautogui.click()` | OCR定位→点击 |

**terminal_server**（5工具）— asyncio + Python内置：
| 工具 | 底层函数 | 说明 |
|------|---------|------|
| shell_exec | `asyncio.create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)` | 子进程执行，超时30s |
| shell_read_file | `open(path).readlines()[offset:offset+limit]` | 读文件指定行 |
| shell_write_file | `open(path,'w').write()` + `os.makedirs()` | 写文件+建目录 |
| shell_list_dir | `os.listdir()` + `os.path.getsize()` | 列目录+大小 |
| shell_cwd | `os.getcwd()` / `os.path.join()` | 工作目录 |

**screen_perception_server**（3工具）— pytesseract + PIL.ImageDraw：
| 工具 | 底层函数 | 说明 |
|------|---------|------|
| capture_and_parse | OCR → `ImageDraw.rectangle()` → JPEG | 截屏+SOM彩色标注 |
| find_element | OCR → 文字匹配 → 中心坐标 | 按描述找UI元素 |
| read_text | `pytesseract.image_to_data()` | OCR读屏 |

**memory_server**（7工具）— SQLiteStore：
| 工具 | 说明 |
|------|------|
| memory_search | FTS5搜三层记忆 |
| memory_write | 写入Semantic层 |
| skill_list | 按频率列出Skills |
| skill_find | 搜索匹配Skill |
| skill_validate | 验证Skill结构完整性 |
| skill_evolve | 执行validate+merge+prune |
| skill_create | 手动创建Skill |

**playwright**（社区包，~5-8工具）— Chromium CDP协议：navigate, click, fill, screenshot, evaluate等

**ddg-search**（1工具）：DuckDuckGo搜索

**time**（1工具）：获取当前时间

**MCP通信机制**：
- 每个Server是独立Python/Node子进程
- 通过stdio收发JSON-RPC 2.0消息
- MCPClient用`mcp.client.stdio.stdio_client()`启动子进程，建立`ClientSession`
- 工具名前缀：LLM调用`computer__click`，ToolExecutor去前缀为`click`，路由到computer server
- 启动时`list_tools()`发现所有工具，缓存到ToolManager

**LLM如何知道调什么工具**：
不是prompt注入，是LangChain的`bind_tools()`。44个工具的schema（name+description+parameters）通过OpenAI function calling协议绑到LLM，LLM看到用户消息后自己决定调哪个，返回`AIMessage(tool_calls=[{name, args}])`。

### 2.4 安全模块

**4级权限**（permission_manager.py）：

| 级别 | 值 | 工具示例 |
|------|---|---------|
| READ_ONLY | 0 | screenshot, search, read_file, screen_info, skill_list |
| STANDARD | 1 | memory_write |
| SENSITIVE | 2 | shell_exec, shell_write_file, playwright_navigate |
| DANGEROUS | 3 | click, type, hotkey, scroll, move, drag |

检查逻辑：`session_level >= tool_required_level`才允许。Session级别可通过API覆盖。

**硬屏蔽**（blocked_list.py）：50+危险命令模式永远拒绝，不经过权限判断：
- `rm -rf /`、`format`、`del /s`、`rmdir /s` — 文件系统破坏
- `reg delete`、`regedit` — 注册表操作
- `net user`、`net localgroup` — 用户管理
- `shutdown`、`reboot` — 系统控制
- `dd if=`、`mkfs` — 磁盘操作
- `:(){ :\|:& };:` — fork炸弹

**审计日志**（audit_logger.py）：SQLite存储，记录每次工具调用的tool_name、arguments、result_summary、permission_level、was_approved、was_blocked、duration_ms。

### 2.5 SDD规约引擎

**Spec Schema**（schema.py）：
```
Spec
├── spec_version: "1.0"
└── task: TaskSpec
    ├── goal: str
    ├── constraints: List[str]
    ├── interfaces: {input: InterfaceSpec, output: InterfaceSpec}
    ├── acceptance_criteria: List[AcceptanceCriterion]
    │   └── {description, check_type, expected, critical}
    └── decomposition: List[DecompositionStep]
        └── {step, action, description, with_args, expected_output, on_failure}
```

**Spec生成**（spec_generator.py）：LLM收到用户任务+工具列表→生成结构化Spec JSON→解析为Spec对象→decomposition转为plan_steps兼容后续节点→Spec存入state.metadata.current_spec

**Harness验证**（harness.py）：对每条acceptance_criterion检查：
- `contains`：expected文本是否出现在结果中；无expected时检查描述关键词
- `not_contains`：expected文本不出现在结果中；无expected时检查危险关键词
- `returns_success`：成功关键词存在且错误关键词不存在
- `matches_schema`：结果非空且不是错误
- `custom`：fallback，检查前100字符有无error

返回HarnessResult：`all_passed`、`pass_rate`、`critical_failures`列表。

**Reflector集成**：如果state.metadata.current_spec存在，Reflector先跑Harness验证，Harness说all_passed则直接判成功，有critical_failures则判失败。覆盖纯关键词判断。

### 2.6 上下文压缩

**context_compressor.py**：
- 阈值：消息数>20时触发
- 策略：前N-10条消息→LLM摘要成≤500字→替换为一条SystemMessage；最近10条原样保留
- Fallback：无LLM时取前后半段拼接截断
- 压缩前后打印字符数对比日志

### 2.7 Agent图拓扑

```
memory_retrieval → supervisor ──→ chat_worker → __end__
                     │
                     ├──→ planner → tool_planner → tool_worker → reflector
                     │                                              │
                     │                           ┌─────────────────┘
                     │                           ↓
                     │                       supervisor (loop)
                     
Entry: memory_retrieval
State: AgentState (messages用add_messages reducer, 其他字段last-writer-wins)
```

**各节点职责**：
- **memory_retrieval**：query搜三层记忆，结果注入state
- **supervisor**：轻量路由，只输出"chat"/"tools"/"__end__"，根据tool_inventory和对话决定
- **planner**（即spec_generator）：复杂任务生成结构化Spec，分解为plan_steps
- **tool_planner**：LLM + bind_tools生成具体tool_calls（AIMessage）
- **tool_worker**：执行MCP工具调用，构建多模态ToolMessage
- **reflector**：关键词+Harness验证结果，决定success/retry(≤3)/abort/chat

---

## 三、面试QA（20题）

### 记忆系统

**Q1：为什么分5层而不是3层？和传统RAG的区别？**

5层是按记忆的生命周期分的，不是拍脑袋定的数量。Sensory是帧级感知数据，看了就忘；Working是对话上下文，聊完就忘；Episodic是聊天记录，能搜但不能自动总结；Semantic是从记录提炼出的结构化知识，是Reflector从Episodic里抽出来的；Skill是反复操作沉淀的可复用模板。层间有evolution，不是5个独立数据库。

和传统RAG的区别是：传统RAG是"存文档→检索文档"，是静态的。我们是"对话→提炼知识→沉淀技能→下次复用"，记忆会自我演化。RAG只有检索没有evolution。

**Q2：FTS5和向量检索具体区别？为什么不用向量检索？**

FTS5是SQLite内置的全文搜索引擎，建倒排索引，支持BM25排序，搜"高兴"只能找到包含"高兴"的记录。向量检索是sentence-transformers把文本编码成768维向量，按余弦相似度检索，搜"高兴"能找到"开心"。

不用向量检索是因为：桌面应用单用户，记忆条目是短文本结构化知识（20-100字），FTS5+LLM rerank够用；向量检索需要embedding模型常驻内存（all-MiniLM-L6-v2约90MB），对桌面应用开销大。架构上SQLiteStore的search方法可以加一个向量召回分支，升级成本很低。

**Q3：记忆检索是怎么融入整个交互链路的？**

用户发消息后，第一个节点就是memory_retrieval。它调用memory_system.retrieve_context(query)，并行搜Episodic/Semantic/Skill三层。Episodic返回最近的对话记录，Semantic返回提炼的知识点，Skill返回匹配的操作模板。这些结果注入supervisor的prompt，supervisor根据记忆决定路由。比如用户说"帮我打开昨天的文档"，memory_retrieval搜到昨天打开文档的记录，supervisor就知道要调terminal工具而不是browser。

**Q4：Skill的自进化具体怎么做的？和简单的宏录制有什么区别？**

SkillMiner每2个consolidation周期运行一次。它维护一个工具调用序列历史deque，每次对话有工具调用就record_tool_sequence。mine时统计精确序列出现频率，≥3次的生成Skill。生成过程可以用LLM（给工具序列+频率+示例，让LLM生成name/description/params/template）或rule-based fallback。

和宏录制的区别：①参数化——Skill有params定义和template占位符，不是写死的；②验证——validate_skill检查参数类型、模板占位符匹配、必填项完整；③进化——evolve_skills会合并相似Skill（Jaccard工具集相似度>0.6）、淘汰低频Skill（频率<2且age>30天）；④检索——下次用户说话时skill_find能语义匹配到已有Skill，直接复用不用重新规划。

**Q5：上下文压缩怎么做的？会不会丢关键信息？**

滑窗+LLM摘要。对话超过20轮时，把前N-10条消息压缩成一段500字以内的摘要（LLM生成，保留任务目标/关键决策/重要事实，省略寒暄/重复/工具细节），替换为一条SystemMessage。最近10条原样保留。

信息丢失风险在于LLM摘要可能遗漏。缓解措施：①只压缩早期消息，最近10条不压缩；②摘要prompt强调"保留关键决策和事实"；③更可靠的方案是MemGPT的滚动检索——不是压缩，而是把旧消息存档，需要时再检索回来。但当前方案在LLM上下文窗口内够用。

### MCP与工具

**Q6：MCP和直接调函数有什么区别？为什么要用MCP？**

三个核心优势：①进程隔离——每个MCP Server是独立子进程，computer server崩溃不影响terminal server；②热插拔——工具配置在mcp_servers.json里，加一个server改配置重启就生效，不需要改主进程代码；③跨语言——社区MCP Server有Python的、Node的，MCP协议不限制语言，直接拿来用。

技术上是stdio + JSON-RPC 2.0通信。MCPClient启动子进程，通过stdin发请求，从stdout读响应。比如playwright server就是`npx @playwright/mcp@latest`启动的Node进程。

**Q7：LLM怎么知道要调哪个工具？工具太多会不会选错？**

通过LangChain的bind_tools()，44个工具的schema（name+description+parameters）绑到LLM上。LLM看到用户消息后自己决定调哪个，返回AIMessage(tool_calls=[{name, args}])。这是OpenAI function calling的标准用法，不是prompt注入。

选错的情况确实存在。缓解措施：①Supervisor先做粗粒度路由——只决定去chat还是tools；②ToolPlanner拿到绑了工具的LLM做细粒度选择，只传最近6条消息减少干扰；③工具描述写得明确（比如computer__click的description说"click at screen coordinates"），LLM根据语义匹配选工具。

**Q8：桌面操控的技术实现？OCR方案和UI Automation方案的区别？**

桌面操控分两种模式：①直接IO——terminal_server的read_file/write_file/shell_exec，不需要"看到"界面，Python直接操作文件和进程；②模拟操作——computer_use_server截屏→VLM理解或OCR定位→pyautogui坐标点击。

当前用OCR+pyautogui是1.0方案，优点是能操作任意应用不依赖系统API，缺点是精度受OCR和分辨率限制。更优方案是Windows UI Automation API获取辅助功能树（a11y tree），直接拿到结构化UI元素（按钮、输入框、菜单项），精度100%且延迟极低，这是AgentS和UI-TARS的3.0方案。实际部署应该分层：浏览器用Playwright（DOM级），桌面操作优先UI Automation，fallback到VLM+OCR。

**Q9：安全模块怎么防止误操作？**

三道防线：①blocked_list硬屏蔽——50+危险命令模式永远拒绝，`rm -rf /`、`format`、`reg delete`等，不经过权限判断直接拦截；②PermissionManager 4级权限——READ_ONLY(截图/搜索/读文件) → STANDARD(记忆写入) → SENSITIVE(shell执行/文件写入) → DANGEROUS(鼠标点击/键盘输入)，session级别低于工具要求级别就拒绝；③Human-in-the-Loop——DANGEROUS级别的工具首次调用时暂停，等用户确认才执行。加上audit_logger记录每次工具调用的完整信息（谁/什么时候/什么参数/什么结果/什么权限级别），出了事可追溯。

**Q10：MCP Server是怎么启动和通信的？**

启动流程：MCPClient收到第一个工具调用请求时，用`mcp.client.stdio.stdio_client()`启动子进程。参数包括command（如"python"）、args（如`["-m", "open_llm_vtuber.mcp_servers.computer_use_server"]`）、timeout（30s）。子进程启动后建立ClientSession，缓存在active_sessions字典里，后续调用复用同一个session。

通信协议是JSON-RPC 2.0 over stdio。客户端发`{"jsonrpc":"2.0","method":"tools/call","params":{"name":"click","arguments":{...}}}`，服务端返回`{"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"Clicked..."}]}}`。服务端用FastMCP的`@mcp.tool()`装饰器注册工具，启动时自动暴露给客户端。

### Agent架构

**Q11：你的系统和普通ChatGPT调用工具有什么本质区别？**

ChatGPT调用工具是单轮reactive：用户问→LLM选工具→返回结果→结束。我们的区别在三个能力：①多步规划——Planner先生成结构化执行计划（含goal/tool_hint/success_criteria），不是拿到就做；②结果验证——Reflector检查工具结果，error就重试，Harness验证acceptance_criteria，不是做完就报；③记忆演化——SkillMiner定期扫描操作模式发现规律自动生成Skill，不是每次从零开始。

本质是从"单轮工具调用"升级为"规划-执行-验证-重试"闭环。

**Q12：Planner和Supervisor职责有什么区别？不会重复吗？**

职责完全不同。Supervisor是轻量路由，只输出一个词（"chat"/"tools"/"__end__"），决定大方向。Planner是复杂任务分解，生成结构化Spec（goal+constraints+acceptance_criteria+decomposition_steps），这个Spec会贯穿后续所有步骤被Reflector用来验证。

类比：Supervisor是前台接待（"你要找哪个科室"），Planner是主治医师（"你的情况需要这三步治疗方案，每步有验收标准"）。

**Q13：SDD规约驱动具体是什么？Harness验证靠谱吗？**

SDD = Spec-Driven Development，借鉴软件工程的规约驱动开发。普通Planner生成的是自然语言步骤，无法验证。SDD生成的是结构化Spec，包含AcceptanceCriterion（验收标准），每条有check_type（contains/not_contains/returns_success/matches_schema）。

Harness验证在Reflector中运行，对每条acceptance_criterion检查工具结果。说靠谱也靠谱——`returns_success`检查成功关键词存在且错误关键词不存在，能正确判断大多数情况。说不靠谱——本质是关键词匹配，不是语义理解，"Error: success"这种边界case会误判。

更可靠的方案是让LLM做verification而不是关键词匹配，但成本高、延迟大。当前方案是性价比取舍：关键词快且免费，边界case有LLM fallback兜底。

**Q14：VLM视觉链路之前为什么不work？你是怎么修的？**

断裂点在两个地方：①tool_worker.py里`ToolMessage(content=str(base64_data))`，把base64图片强转成纯文本，LLM收到的是一坨乱码；②tool_planner.py里`msg.content[:300]`，base64被截断成300字符的乱码碎片。

修复方案：①tool_worker检测到base64图片→构建多模态content blocks：`[{"type":"text","text":"截图摘要"}, {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`；②tool_planner检测到多模态ToolMessage→保留image_url块传给VLM，文本截断上限提到500；③同步修了chat_worker/supervisor/reflector对多模态ToolMessage的处理。还做了截图优化：1920x1080 PNG(2-5MB)缩放为1280x720 JPEG 75%质量(~200KB)，VLM推理速度提升5-10倍。

**Q15：LangGraph的StateGraph是什么？和普通if-else有什么区别？**

StateGraph是有限状态机，节点是函数，边是转移条件。和if-else的区别是：①声明式——你定义节点和边，框架管理执行顺序和状态传递，不需要手写调度逻辑；②状态管理——AgentState用TypedDict定义，messages字段用add_messages reducer实现追加而非覆盖，每个节点返回partial state，LangGraph自动merge；③可视化——编译后的图可以导出Mermaid图；④持久化——LangGraph支持checkpointing，中断后可以恢复。

if-else也能实现同样的逻辑，但需要手动管理状态传递、错误处理、重试逻辑。StateGraph是工程化工具不是能力提升工具。

### 综合与深度

**Q16：如果让你改进，优先改什么？**

四个方向：①换Gemini Flash做视觉链路——当前qwen2.5-vl-72b API延迟3-5秒，Gemini Flash 1-2秒，匹配商用体验；②加向量检索层——当前FTS5无法语义匹配，加sentence-transformers+pgvector，两路并行RRF融合；③加UI Automation——Windows a11y tree替代OCR定位，精度100%延迟毫秒级；④异步记忆写入——当前on_conversation_turn同步写三层，大量对话时可能卡顿，改为后台任务异步写入。

**Q17：和Open Interpreter(龙虾)比，优劣势分别是什么？**

优势：①5层记忆+Skill演化——龙虾没有记忆系统，每次对话从零开始；②SDD规约+Harness验收——龙虾做完就报不管对不对；③4级权限+审计——龙虾只有safe_mode开关；④中文输入支持——pyperclip+Ctrl+V。

劣势：①VLM推理速度——72B模型3-5秒，龙虾用GPT-4V 1-2秒；②图标语义搜索——龙虾用CLIP+sentence-transformers匹配图标描述，我们只能OCR找文字；③CV元素检测——龙虾用OpenCV自适应阈值+轮廓检测UI元素，我们没有；④多显示器——龙虾支持screeninfo检测多屏。

**Q18：Playwright和pyautogui的操控有什么区别？什么时候用哪个？**

Playwright操控浏览器：通过CDP协议直连Chromium，操作DOM节点（`page.click('button#submit')`），100%精准，毫秒级延迟，但只能操控浏览器。pyautogui操控桌面：通过模拟鼠标键盘事件（`pyautogui.click(832, 456)`），能操作任意应用，但依赖坐标精度，受分辨率影响。

使用策略：浏览器内的操作（搜索、填表、爬数据）用Playwright；桌面应用的操作（关窗口、拖文件、操作本地软件）用pyautogui+VLM/OCR。两者都能做时Playwright优先（更精准更快）。

**Q19：这个项目里你做了哪些？哪些是开源项目的？**

基于Open-LLM-VTuber开源项目二次开发。原始项目提供了：对话框架（WebSocket handler、conversation pipeline）、Live2D渲染（pixi-live2d-display）、ASR（sherpa-onnx）、TTS（edge-tts）、VAD（silero）、基础LLM调用。

我从零实现的：①LangGraph多Agent拓扑（Supervisor/Planner/ToolPlanner/Reflector全部重写）；②5层记忆系统+Skill自进化；③3个MCP Server（computer_use_server/terminal_server/screen_perception_server）；④VLM视觉链路打通（多模态消息处理）；⑤安全模块（权限/屏蔽/审计）；⑥SDD规约引擎（Spec/Harness/Validator）。

**Q20：多模态感知"整合屏幕视觉与VAD语音"具体怎么做的？**

屏幕视觉：截屏→JPEG压缩→base64→通过image_url传给VLM，LLM理解屏幕内容输出操作决策。OCR+SOM标注是辅助方案，不需要VLM时用pytesseract做文字定位。

VAD语音：Silero VAD神经网络模型实时检测语音活动，状态机（IDLE→ACTIVE→INACTIVE），概率阈值判断是否在说话。检测到用户说话就中断当前TTS播放，实现全双工交互（barge-in打断）。

两者的整合在single_conversation.py：WebSocket handler同时接收文本消息和语音流，语音流经ASR转文本后和文本消息走同一个agent.chat pipeline。语音场景下VAD检测到停顿就触发LLM响应，不需要等用户按"发送"。

这两个模态是输入侧的并行感知，输出侧是Live2D骨骼驱动（TTS音量→嘴型同步）+情绪渲染（LLM emotion标签→表情参数），通过WebSocket实时推送。

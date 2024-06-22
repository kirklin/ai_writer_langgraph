# LangGraph Study

LangGraph Study 是一个基于 LangChain 和 OpenAI 的项目，展示了如何使用语言模型和工具来生成大纲并根据大纲撰写文章。项目通过定义多个 Agent（大纲生成器和文章写手），并利用一个主管（Supervisor）来协调这些 Agent 之间的工作流。

## 项目结构

- `config.py`: 包含 API 相关的配置，如 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。
- `main.py`: 项目的主代码文件，包含工作流的定义和执行逻辑。

## 环境配置

在运行此项目之前，请确保已正确配置环境变量。你可以在 `config.py` 文件中设置 API 密钥和 Base URL：

```python
# config.py
OPENAI_API_KEY = 'your_openai_api_key'
OPENAI_BASE_URL = 'your_openai_base_url'
```

## 安装依赖

确保你已安装所需的 Python 库。你可以使用以下命令来安装所有依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件内容如下：

```text
langchain
langchain_core
langchain_openai
langchain_community
langgraph
```

## 运行项目

你可以通过运行 `main.py` 来启动项目：

```bash
python main.py
```

## 项目概述

本项目展示了如何通过多个 Agent 协同工作来完成复杂任务。主要包括以下几个部分：

1. **大纲生成器 (Outline Generator)**: 接受一个主题，并生成相应的大纲。
2. **文章写手 (Article Writer)**: 根据生成的大纲撰写一篇文章。
3. **主管 (Supervisor)**: 协调各个 Agent 之间的工作流，根据任务的完成情况决定下一个执行任务的 Agent。

## 工作流程

1. 用户输入一个主题请求。
2. 大纲生成器生成该主题的大纲。
3. 文章写手根据大纲撰写文章。
4. 主管根据任务进展协调各个 Agent 的工作，直至任务完成。

# Chinese Jailbreak

[English](README.en.md) | [中文](README.zh-CN.md)

本仓库提供面向中文大语言模型安全评测研究的核心提示生成代码。

## 安装

创建并激活 Conda 环境：

```bash
conda env create -f environment.yml
conda activate jail
```

也可以直接安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 配置

运行前设置本地模型路径：

```bash
export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
```

Windows PowerShell：

```powershell
$env:LOCAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
```

仓库中提供了 `.env.example` 作为配置参考。

## 使用

```bash
python examples/run_single.py
```

也可以传入自定义输入：

```bash
python examples/run_single.py "请将这句话改写为安全评测提示"
```

## 数据集

评测数据请参考 JailBench：

https://github.com/PKU-Alignment/JailBench

## 伦理使用

本代码仅用于授权安全评测、红队测试和模型护栏改进研究。

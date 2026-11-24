# EM_stu 1.0.25 使用指南

EM_stu 是一个 VS Code 插件。你可以从一个机器学习预制项目开始，设置实验目标，然后在编辑器里查看训练进度、日志和结果。

## 在 VS Code 中安装

1. 使用 VS Code 桌面版 1.99 或更高的 1.x 版本。先把 [em_stu_1.0.25.vsix 下载到本机](./em_stu_1.0.25.vsix?raw=true)，保留文件的 `.vsix` 扩展名。
2. 打开 VS Code，点击左侧活动栏的「扩展」（Extensions）图标。在**扩展视图**右上角点击「…」，选择「从 VSIX 安装…」（Install from VSIX…），选中刚下载的文件。也可以打开命令面板，搜索并执行 `Extensions: Install from VSIX...`。
3. 安装完成后，在命令面板执行 `Developer: Reload Window` 重新加载窗口。回到「扩展」，在已安装项目中搜索 **EM_stu**，打开详情页，核对版本为 **1.0.25**。

如果已经配置了 VS Code 的 `code` 命令，也可以在下载目录执行以下命令。

```sh
code --install-extension ./em_stu_1.0.25.vsix
```

安装后同样需要重新加载窗口。

## 首次使用

在 macOS 或 Linux 上，用 VS Code 打开用于实验的本地文件夹。确认信任该工作区后，点击左侧的 **EM_stu** 图标，或在命令面板执行 `EM_stu: Open Task Panel`。执行 `EM_stu: Check Environment`，按提示准备 Python 3.9 或更高版本及运行依赖。在插件设置中选择 Qoder 或 Codex，完成对应 CLI 的安装和认证。

## 跑第一个实验

1. 在「项目准备」中选择「新手科研入门」，第一次可以选 MNIST。阅读预填的任务提示，按需修改，点击「确认下载」。代码和数据分别下载，校验完成后自动进入「任务配置」。已有本地项目也可以从「从本地项目开始」进入。
2. 确认项目目录、主指标、基线和目标值。预制项目会填入一个 epoch 的基线结果。选择本次运行的 Agent 和模型，点击「校验并继续」。
3. 点击「启动任务」并确认执行。在「运行流程」查看当前节点和指标，点击「日志」或节点产物查看详情。离开面板后，可在「任务历史」中重新打开运行记录。

## 许可与校验

插件原创代码及文档采用 Apache License 2.0。安装包内的 LICENSE.txt、NOTICE 和 THIRD_PARTY_NOTICES.md 分别列出许可条款、软件和数据归属、第三方组件声明。预制数据集和模型权重适用各自的条款。

安装包 SHA-256 为 `adb0d8e16798d57cb853a9eccfe87f99f9560165bfe26bdf0b6ddc40c6516ae8`。

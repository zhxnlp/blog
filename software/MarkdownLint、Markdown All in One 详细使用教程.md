# 文章目录
<!-- no toc -->
- [一、 Markdown简介](#一-markdown简介)
  - [1.1 简介](#11-简介)
  - [1.2 熟悉VS Code的Markdown功能](#12-熟悉vs-code的markdown功能)
  - [1.3 尝试Markdown预览器](#13-尝试markdown预览器)
  - [1.4 Markdown Snippets（代码补全）](#14-markdown-snippets代码补全)
- [二、MarkdownLint教程](#二markdownlint教程)
  - [2.1 MarkdownLint简介与安装](#21-markdownlint简介与安装)
  - [2.2 使用方法](#22-使用方法)
    - [2.2.1 基本使用](#221-基本使用)
    - [2.2.2 启用自动修复](#222-启用自动修复)
    - [2.2.3 批量检查当前工作区中的所有 Markdown 文件](#223-批量检查当前工作区中的所有-markdown-文件)
    - [2.2.4 临时禁用或重启 MarkdownLint 的检查功能](#224-临时禁用或重启-markdownlint-的检查功能)
  - [2.3 代码片段](#23-代码片段)
  - [2.4 规则说明](#24-规则说明)
  - [2.5 配置](#25-配置)
    - [2.5.1 配置文件](#251-配置文件)
    - [2.5.2 继承配置](#252-继承配置)
    - [2.5.3 配置优先级](#253-配置优先级)
    - [2.5.4 配置详情](#254-配置详情)
      - [2.5.4.1 `markdownlint.config`（配置规则）](#2541-markdownlintconfig配置规则)
      - [2.5.4.2 `markdownlint.configFile`（配置文件路径）](#2542-markdownlintconfigfile配置文件路径)
      - [2.5.4.3. `markdownlint.focusMode`（聚焦模式）](#2543-markdownlintfocusmode聚焦模式)
      - [2.5.4.4  `markdownlint.run`（运行时机）](#2544--markdownlintrun运行时机)
      - [2.5.4.5  `markdownlint.customRules`（自定义规则）](#2545--markdownlintcustomrules自定义规则)
      - [2.5.4.6 `markdownlint.lintWorkspaceGlobs`（工作区检查模式）](#2546-markdownlintlintworkspaceglobs工作区检查模式)
  - [2.6 抑制警告](#26-抑制警告)
  - [2.7 安全性](#27-安全性)
- [三、Markdown All in One](#三markdown-all-in-one)
  - [3.1 功能介绍](#31-功能介绍)
    - [3.1.1 键盘快捷键](#311-键盘快捷键)
    - [3.1.2 目录生成（及所有Markdown All in One命令）](#312-目录生成及所有markdown-all-in-one命令)
  - [3.1.3 列表编辑](#313-列表编辑)
  - [3.1.4 将 Markdown 文档转换为 HTML 格式](#314-将-markdown-文档转换为-html-格式)
    - [3.1.5 其他功能](#315-其他功能)
  - [3.2 支持的设置](#32-支持的设置)
  - [3.3 常见问题解答](#33-常见问题解答)
  - [3.4 更新日志](#34-更新日志)

## 一、 Markdown简介

>参考[《Build an Amazing Markdown Editor Using Visual Studio Code and Pandoc》](https://thisdavej.com/build-an-amazing-markdown-editor-using-visual-studio-code-and-pandoc/)

### 1.1 简介

[Markdown](https://en.wikipedia.org/wiki/Markdown)是一种轻量级标记语言，允许用户使用易读易写的纯文本格式编写文档，并可将其转换为多种不同格式。它特别适合用于撰写技术文档，因为Markdown文档可以与Git或您选择的源代码控制系统一起检查和版本控制。

&#8195;&#8195;如果你不熟悉Markdown语法，请查看Adam Pritchard的[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)，其中包括标准Markdown语法以及我们将在编辑器中使用的扩展GFM（GitHub Flavor Markdown）。

### 1.2 熟悉VS Code的Markdown功能

1. 创建一个名为“md”（或您选择的名称）的文件夹，用于存储您的Markdown文件。我们稍后还会向此文件夹中添加一些额外的VS Code配置文件。

2. 右键单击此文件夹并选择“用Code打开”。这将打开整个文件夹，而不是仅仅打开单个文件，这对于在给定文件夹中同时创建和编辑多个文件非常方便。

3. 创建一个名为“test.md”的文件，并添加以下内容：

```markup
# Heading 1
## Heading 2 text

Hello world!

We will output Markdown to:
- HTML
- docx
- PDF
```

### 1.3 尝试Markdown预览器

1. 点击`Ctrl+Shift+V`，您将看到Markdown代码的HTML格式预览。再次点击`Ctrl+Shift+V`可返回Markdown代码。

2. 您还可以创建一个单独的窗口窗格来预览Markdown。为此：

    - 按F1打开VS Code命令面板，输入“Markdown: Open Preview to the Side”。
        ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/acf67f968f7c44a6b01cb7e51ad337b3.png)

    - 或者，您可以按快捷键`Ctrl+K`后跟`V`（`Ctrl+K V`）来创建侧边预览窗格。

&#8195;&#8195;您现在可以在左侧窗格中的Markdown文档中添加文本，并在右侧的HTML预览窗格中看到这些更改。现在就试试这个功能，非常不错！
>&#8195;&#8195;按下CTRL K,松开之后按CTRL S,可以打开vscode的快捷键设置界面,查看是否有冲突.

### 1.4 Markdown Snippets（代码补全）

1. 在Markdown文档中，尝试按`Ctrl+Space`，VS Code将提供一个上下文敏感的Markdown命令列表供您使用。例如，您可以按`Ctrl+Space`，输入“link”，然后按`Enter`来插入一个超链接。您可以添加超链接文本，按`Tab`，然后输入URL。在许多这些片段中，`Tab`键是很有用的。
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/4932803bec534179b5b16f66405be48e.png#pic_center> =800x)
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/c695dccd9a45464da0fc059ee94ec187.png#pic_center> =800x)

    >&#8195;&#8195;win10自带的微软输入法中，切换中英文的快捷键是`Ctrl+Space`或`shift`，第一个一个快捷功能与这里重复，将其取消就OK了。
    >&#8195;&#8195;这也太好用了，写了几年md博客都没发现。不过CSDN中超链接添加也很方便啊。

2. 虽然超出本教程范围，但您还可以通过导航到VS Code菜单中的“文件”|“首选项”|“用户片段”|“Markdown”来创建自己的Markdown片段。有关更多信息，请参阅此文章。

## 二、MarkdownLint教程

### 2.1 MarkdownLint简介与安装

&#8195;&#8195;[MarkdownLint](https://github.com/DavidAnson/vscode-markdownlint?tab=readme-ov-file) 是 Visual Studio Code 的一个扩展，用于对 Markdown 文件进行语法检查和风格检查。它包含一系列规则，以鼓励 Markdown 文件的标准和一致性。这些规则由 Node.js 的 [markdownlint 库](https://github.com/DavidAnson/markdownlint)提供支持，并通过 [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) 引擎执行检查，该引擎也可用于命令行脚本和持续集成场景。

1. 通过 Visual Studio Code 内部安装

   - 按下 `Ctrl+Shift+X`（Windows/Linux）或 `⇧⌘X`（Mac）打开扩展标签页。
   - 输入 `markdownlint` 查找扩展。
   - 点击安装按钮，然后点击启用按钮。

2. 通过命令行安装:打开命令行提示符,运行以下命令：`code --install-extension DavidAnson.vscode-markdownlint`

### 2.2 使用方法

#### 2.2.1 基本使用

1. **显示警告**：安装之后，在 Visual Studio Code 中打开一个 Markdown 文件时，如果文件中有任何违反 MarkdownLint 规则的地方，编辑器会显示警告。警告以绿色波浪下划线表示，也可以通过按下 `Ctrl+Shift+M`（Windows/Linux）或 `⇧⌘M`（Mac）打开错误和警告对话框查看。

2. **查看警告**：将鼠标指针悬停在绿色下划线处，可以看到警告信息，或者按下 `F8` 和 `Shift+F8`（Windows/Linux）或 `⇧F8`（Mac）循环浏览所有警告（所有 MarkdownLint 警告都以 `MD###` 开头）。
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/bb96857944424151a7483fbfa83da21a.png#pic_center> =800x)

3. **查看警告详情**：有关 MarkdownLint 警告的更多信息，将光标放在行上，点击灯泡图标或按下 `Ctrl+.`（Windows/Linux）或 `⌘.`（Mac）打开快速修复对话框。点击对话框中的一个警告，即可进行快速修复
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/636848e6a05045a7b2c5d9619ce0f91f.png#pic_center> =800x)

>&#8195;&#8195;MarkdownLint 默认会检查并报告 VSCode 中被识别为 Markdown 文件的问题。 如果你有一些特殊的文件类型（例如自定义的扩展名），并且希望 VSCode 将它们视为 Markdown 文件进行检查，你可以通过设置将这些文件扩展名与 Markdown 语言模式[关联](https://code.visualstudio.com/docs/languages/overview#_add-a-file-extension-to-a-language)起来。

#### 2.2.2 启用自动修复

1. MarkdownLint 注册自身为 Markdown 文件的源代码格式化程序，可以通过以下命令调用：
   - 格式化文档：`Format Document` 或 `editor.action.formatDocument`，可以通过命令面板（`View|Command Palette...` 或 `Ctrl+Shift+P`/ `Ctrl+Shift+P`/ `⇧⌘P`）或默认键盘快捷键 `Shift+Alt+F`（Windows）/ `Ctrl+Shift+I`（Linux）/ `⇧⌥F`（Mac）来执行。
   - 格式化选定内容：`Format Selection` 或 `editor.action.formatSelection`，可以通过命令面板或默认键盘快捷键 `Ctrl+K Ctrl+F`/ `Ctrl+K Ctrl+F`/ `⌘K ⌘F` 来执行。
2. 为了在保存或粘贴到 Markdown 文档时自动格式化，可以按以下方式配置 Visual Studio Code 的 `editor.formatOnSave` 或 `editor.formatOnPaste` 设置：

   ```json
   "[markdown]": {
       "editor.formatOnSave": true,
       "editor.formatOnPaste": true
   },
   ```

3. MarkdownLint 还提供了 `markdownlint.fixAll` 命令，可以在一步中修复文档的所有违规项，并可以从命令面板运行或绑定到键盘快捷键（我将其设置为`CTRL + K L`）。
4. 为了在保存 Markdown 文档时自动修复违规项，可以按以下方式配置 Visual Studio Code 的 `editor.codeActionsOnSave` 设置：

   ```json
   "editor.codeActionsOnSave": {
       "source.fixAll.markdownlint": true
   }
   ```

>任何一种方法自动应用的修复程序都可以通过Edit撤销，或者`Ctrl+Z`。
>
#### 2.2.3 批量检查当前工作区中的所有 Markdown 文件

1. MarkdownLint 提供了一个命令 `markdownlint.lintWorkspace`，用于检查当前工作区中的所有 Markdown 文件。这个命令使用了与扩展相同的底层引擎 markdownlint-cli2 来执行检查。检查结果会输出到 VSCode 的“终端”面板（Terminal panel）中的一个新终端窗口。
。
2. 结果还会出现在“问题”面板中（通过`Ctrl+Shift+M`/ `Ctrl+Shift+M`/ `⇧⌘M`打开）中，你可以点击具体的条目，直接打开对应的文件并定位到有问题的行。

3. 如果你希望自定义哪些文件被包含或排除在检查之外，可以通过配置 `markdownlint.lintWorkspaceGlobs` 设置来实现（详见下文）。

#### 2.2.4 临时禁用或重启 MarkdownLint 的检查功能

1. 要临时禁用 Markdown 文档的检查，可以运行 `markdownlint.toggleLinting` 命令。当你打开一个新的工作区时，`markdownlint.toggleLinting` 的效果会被重置
2. 要重新启用检查，再次运行 `markdownlint.toggleLinting` 命令。

### 2.3 代码片段

&#8195;&#8195;在编辑 Markdown 文档时，以下代码片段可用（按下 `Ctrl+Space`/ `Ctrl+Space`/ `⌃Space` 获取 IntelliSense 建议）：

| 代码片段                | 描述                                                                 | 示例                                      |
|-------------------------|----------------------------------------------------------------------|-------------------------------------------|
| `markdownlint-disable`   | 禁用所有规则，直到遇到 `markdownlint-enable` 或文档结束。                     | `<!-- markdownlint-disable -->`           |
| `markdownlint-enable`    | 重新启用所有规则。                                                   | `<!-- markdownlint-enable -->`            |
| `markdownlint-disable-line` | 禁用当前行的规则。                                               | `<!-- markdownlint-disable-line -->`      |
| `markdownlint-disable-next-line` | 禁用下一行的规则。                                     | `<!-- markdownlint-disable-next-line -->` |
| `markdownlint-capture`   | 捕获当前的 MarkdownLint 配置状态，以便后续可以恢复。                           | `<!-- markdownlint-capture -->`           |
| `markdownlint-restore`   | 恢复之前捕获的 MarkdownLint 配置状态。                                   | `<!-- markdownlint-restore -->`           |
| `markdownlint-disable-file` | 禁用整个文件的规则。                                       | `<!-- markdownlint-disable-file -->`      |
| `markdownlint-enable-file`  | 重新启用整个文件的规则。                                         | `<!-- markdownlint-enable-file -->`       |
| `markdownlint-configure-file` | 为当前文件配置 MarkdownLint 规则。                                 | `<!-- markdownlint-configure-file -->`    |

### 2.4 规则说明

MarkdownLint 包含一系列规则，以下是一些常见的规则：

| 规则编号 | 描述 | 规则编号 | 描述 |
|---------|---------|---------|---------|
| MD001    | heading-increment：标题级别应一次只增加一个级别。 | MD003    | heading-style：标题风格。 |
| MD004    | ul-style：无序列表风格。 | MD005    | list-indent：同一级别的列表项缩进不一致。 |
| MD007    | ul-indent：无序列表缩进。 | MD009    | no-trailing-spaces：尾随空格。 |
| MD010    | no-hard-tabs：硬制表符。 | MD011    | no-reversed-links：反向链接语法。 |
| MD012    | no-multiple-blanks：多个连续的空行。 | MD013    | line-length：行长度。 |
| MD014    | commands-show-output：在命令前使用美元符号但未显示输出。 | MD018    | no-missing-space-atx：ATX 风格标题的哈希符号后没有空格。 |
| MD019    | no-multiple-space-atx：ATX 风格标题的哈希符号后有多个空格。 | MD020    | no-missing-space-closed-atx：封闭的 ATX 风格标题的哈希符号内没有空格。 |
| MD021    | no-multiple-space-closed-atx：封闭的 ATX 风格标题的哈希符号内有多个空格。 | MD022    | blanks-around-headings：标题周围应有空行。 |
| MD023    | heading-start-left：标题必须从行首开始。 | MD024    | no-duplicate-heading：具有相同内容的多个标题。 |
| MD025    | single-title/single-h1：在同一文档中存在多个顶级标题。 | MD026    | no-trailing-punctuation：标题中的尾随标点符号。 |
| MD027    | no-multiple-space-blockquote：块引用符号后有多个空格。 | MD028    | no-blanks-blockquote：块引用内有空行。 |
| MD029    | ol-prefix：有序列表项前缀。 | MD030    | list-marker-space：列表标记后的空格。 |
| MD031    | blanks-around-fences：围栏代码块周围应有空行。 | MD032    | blanks-around-lists：列表周围应有空行。 |
| MD033    | no-inline-html：内联 HTML。 | MD034    | no-bare-urls：裸露的 URL。 |
| MD035    | hr-style：水平线风格。 | MD036    | no-emphasis-as-heading：使用强调而不是标题。 |
| MD037    | no-space-in-emphasis：强调标记内的空格。 | MD038    | no-space-in-code：代码段元素内的空格。 |
| MD039    | no-space-in-links：链接文本内的空格。 | MD040    | fenced-code-language：围栏代码块应指定语言。 |
| MD041    | first-line-heading/first-line-h1：文件的第一行应为顶级标题。 | MD042    | no-empty-links：空链接。 |
| MD043    | required-headings：必需的标题结构。 | MD044    | proper-names：专有名词应有正确的大小写。 |
| MD045    | no-alt-text：图像应有替代文本（alt 文本）。 | MD046    | code-block-style：代码块风格。 |
| MD047    | single-trailing-newline：文件应以单个换行符结尾。 | MD048    | code-fence-style：代码围栏风格。 |
| MD049    | emphasis-style：强调风格应一致。 | MD050    | strong-style：强强调风格应一致。 |
| MD051    | link-fragments：链接片段应有效。 | MD052    | reference-links-images：引用链接和图像应使用已定义的标签。 |
| MD053    | link-image-reference-definitions：链接和图像引用定义应需要。 | MD054    | link-image-style：链接和图像风格。 |
| MD055    | table-pipe-style：表格管道风格。 | MD056    | table-column-count：表格列数。 |
| MD058    | blanks-around-tables：表格周围应有空行。 |         |         |

### 2.5 配置

#### 2.5.1 配置文件

1. 默认情况下，所有规则都启用，除了 `MD013`/ `line-length`，因为许多文件的行长度超过常规的 80 个字符限制。

    ```
    {
        "MD013": false
    }
    ```

2. 可以通过在项目的任何目录中创建一个名为 `.markdownlint.jsonc`/ `.markdownlint.json`的JSON文件； 或 名为`.markdownlint.yaml`/ `.markdownlint.yml` 的YAML文件；或 `.markdownlint.cjs` 的 JSON 文件来启用、禁用和自定义规则。
3. 此外，还可以通过在项目的任何目录中创建一个名为 `.markdownlint-cli2.jsonc` 或 `.markdownlint-cli2.yaml` 或 名为`.markdownlint-cli2.cjs` 的JavaScript 来配置选项（包括规则和其他设置）。
4. 也可以通过 VS Code 的[User and workspace settings](https://code.visualstudio.com/docs/editor/settings)来进行配置。

>&#8195;&#8195;有关配置文件优先级和完整示例的详细信息，请参阅[markdownlint-cli 2 README.md](https://github.com/DavidAnson/markdownlint-cli2#configuration)的配置部分。

#### 2.5.2 继承配置

1. 自定义规则配置通常由项目根目录中的 `.markdownlint.json` 文件定义。

    ```json
    {
        "MD003": { "style": "atx_closed" },
        "MD007": { "indent": 4 },
        "no-hard-tabs": false
    }
    ```

    - `MD003`：定义标题风格，这里设置为 `atx_closed`（即闭合的 ATX 标题风格，例如 # Title #）。
   - `MD007`：定义无序列表的缩进，这里设置为 4 个空格。
   - `no-hard-tabs`：禁用硬制表符（false 表示允许使用硬制表符）。
2. 如果你希望在当前配置文件中继承另一个配置文件的规则，可以使用 `extends` 属性提供相对路径：

   ```json
   {
       "extends": "../.markdownlint.json",
       "no-hard-tabs": true
   }
   ```

>- 文件位置：通过 `extends` 引用的文件不需要位于当前项目中，但它通常会是项目的一部分。例如，你可以在一个共享的配置文件中定义通用规则，然后在多个项目中通过 `extends` 引用它。
>- 继承规则：继承的配置文件中的规则会被加载，但你可以通过在当前文件中覆盖或添加新的规则来调整行为。

#### 2.5.3 配置优先级

配置源的优先级如下（按优先级递减顺序）：

1. 位于当前目录或父目录中的 `.markdownlint-cli2.jsonc,.markdownlint-cli2.yaml,markdownlint-cli2.cjs` 文件。这是最高优先级的配置文件，通常用于更复杂的配置或扩展功能。
2. 位于当前目录或父目录中的 `.markdownlint.jsonc,.markdownlint.json,.markdownlint.yaml,.markdownlint.yml,.markdownlint.cjs` 文件。这些文件用于常规的 MarkdownLint 配置。
3. 通过 VSCode 的用户设置或工作区设置中的 `markdownlint.config` 和 `markdownlint.configFile` 配置项。这些设置可以在 VSCode 的 `settings.json` 文件中定义。
4. 默认配置：如果没有找到任何自定义配置文件或设置，MarkdownLint 将使用默认的规则配置（请参阅上面）。

- **即时生效**：保存到任何位置的配置文件更改都会立即生效。
- **extends 引用的文件**：通过 extends 属性引用的配置文件不会被 VSCode 监控更改。如果需要应用更改，可能需要手动重新加载或重启 VSCode。
- **显式禁用或启用继承的配置**：可以在任何配置文件中显式禁用或重新启用继承的配置

#### 2.5.4 配置详情

这段内容详细介绍了 **MarkdownLint** 在 VSCode 中的配置选项和使用方法，包括如何自定义规则、配置文件的优先级、如何启用或禁用某些功能，以及如何扩展和自定义规则。以下是逐段解析：

---

##### 2.5.4.1 `markdownlint.config`（配置规则）

- **项目本地配置文件**：建议使用项目本地的配置文件（如 `.markdownlint.json`），因为它可以与命令行工具兼容，并且便于团队协作。
- **VSCode 用户设置中的配置示例**：

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.config": {
          "MD003": { "style": "atx_closed" },  // 设置标题风格为闭合的 ATX 标题
          "MD007": { "indent": 4 },           // 设置无序列表缩进为 4 个空格
          "no-hard-tabs": false               // 允许使用硬制表符
      }
  }
  ```

- **`extends` 的使用**：
  - 如果配置文件中使用了 `extends` 属性，文件路径的解析规则如下：
    - **工作区内的配置文件**：相对路径基于当前配置文件的位置。
    - **用户设置中的 `extends`**：文件路径基于用户主目录（例如 Windows 的 `%USERPROFILE%` 或 macOS/Linux 的 `$HOME`）。
    - **工作区设置中的 `extends`**：文件路径基于工作区文件夹。
    - 可以使用 VSCode 的预定义变量 `${userHome}` 和 `${workspaceFolder}` 来覆盖默认行为。

---

##### 2.5.4.2 `markdownlint.configFile`（配置文件路径）

- **默认行为**：配置文件通常存储在项目的根目录中。
- **自定义配置文件路径**：如果需要将配置文件存储在其他位置，可以通过设置 `configFile` 来指定相对路径。

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.configFile": "./config/.markdownlint.jsonc"
  }
  ```

- **优先级**：如果同时设置了 `markdownlint.config` 和 `markdownlint.configFile`，`configFile` 中的设置优先。

---

##### 2.5.4.3. `markdownlint.focusMode`（聚焦模式）

- **默认行为**：MarkdownLint 会在你输入或编辑文档时实时记录和高亮显示所有问题，包括一些“临时性”问题（如行尾的多余空格）。
- **聚焦模式**：如果觉得实时检查过于干扰，可以启用聚焦模式，忽略光标所在行的问题。

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.focusMode": true
  }
  ```

- **忽略多行**：还可以设置一个正整数，忽略光标上下若干行的问题。

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.focusMode": 2  // 忽略光标上下各 2 行
  }
  ```

- **注意**：这是一个应用级设置，仅在用户设置中有效，不适用于工作区设置。

---

##### 2.5.4.4  `markdownlint.run`（运行时机）

- **默认行为**：MarkdownLint 会在你输入或编辑文档时实时运行。
- **仅在保存时运行**：如果觉得实时检查干扰太大，可以配置为仅在保存文档时运行。

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.run": "onSave"
  }
  ```

- **注意**：在这种情况下，编辑文档时报告的问题列表可能会过时，直到保存文档时才会更新。

---

##### 2.5.4.5  `markdownlint.customRules`（自定义规则）

- **自定义规则**：可以在 VSCode 的用户或工作区设置中指定额外的自定义规则，这些规则可以是 JavaScript 文件或 npm 包。

  ```json
  {
      "editor.someSetting": true,
      "markdownlint.customRules": [
          "./.vscode/my-custom-rule.js",                    // 相对路径
          "./.vscode/my-custom-rule-array.js",
          "./.vscode/npm-package-for-custom-rule",
          "/absolute/path/to/custom/rule.js",               // 绝对路径（不推荐）
          "{publisher.extension-name}/custom-rule.js",     // 扩展路径
          "{publisher.extension-name}/npm/rule/package"
      ]
  }
  ```

- **路径规则**：
  - 通常使用相对路径，并以 `./` 开头。
  - 绝对路径以 `/` 开头，但不推荐，因为它在不同机器上可能不可靠。
  - `{extension}/path` 格式的路径基于已安装的 VSCode 扩展目录。
- **编写自定义规则**：可以参考 [markdownlint 文档](https://github.com/DavidAnson/markdownlint)了解如何编写自定义规则。
- **模块路径**：在 `.markdownlint-cli2.{jsonc,yaml,cjs}` 配置文件中，可以通过 `modulePaths` 属性指定额外的模块解析路径，以解决全局安装的 Node 模块不可用的问题。

---

##### 2.5.4.6 `markdownlint.lintWorkspaceGlobs`（工作区检查模式）

- **默认模式**：在运行 `markdownlint.lintWorkspace` 命令时，MarkdownLint 会检查工作区中所有重要的 Markdown 文件，这些文件的扩展名包括：

  ```json
  [
      "**/*.{md,mkd,mdwn,mdown,markdown,markdn,mdtxt,mdtext,workbook}",  // 匹配所有常见的 Markdown 文件扩展名
      "!**/*.code-search",                                               // 排除特定文件
      "!**/bower_components",
      "!**/node_modules",
      "!**/.git",
      "!**/vendor"
  ]
  ```

- **自定义模式**：可以在工作区或用户设置中自定义这个列表，以包含或排除额外的文件和目录。
- **语法参考**：更多关于语法的信息可以参考 [markdownlint-cli2 文档](https://github.com/DavidAnson/markdownlint-cli2)的“命令行”部分。

### 2.6 抑制警告

&#8195;&#8195;可以通过在 Markdown 文件中添加注释来抑制个别警告，有关内联抑制的更多信息，请参阅 `markdownlint` [README.md 的配置部分](https://github.com/DavidAnson/markdownlint#configuration)。

   ```markdown
   <!-- markdownlint-disable MD037 -->
   deliberate space * in * emphasis
   <!-- markdownlint-enable MD037 -->
   ```

### 2.7 安全性

1. 从自定义规则、`markdown-it` 插件或配置文件（如 `.markdownlint.cjs`/ `.markdownlint-cli2.cjs`）运行 JavaScript 可能存在安全风险，因此 VS Code 的工作区信任设置将被尊重，以阻止未受信任的工作区的 JavaScript。

## 三、Markdown All in One

>参考：[Markdown All in One文档](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)

Markdown All in One 是一款为 Visual Studio Code 提供的强大 Markdown 支持扩展。它提供了许多实用功能，如键盘快捷键、目录生成、自动预览等。以下是详细的使用教程：

### 3.1 功能介绍

#### 3.1.1 键盘快捷键

| 快捷键                | 功能描述                             | 快捷键                | 功能描述                              | 快捷键                | 功能描述                             |
|-----------------------|--------------------------------------|-------------------------------------|-------------------------------------|-------------------------------------|---------------------------------
| `Ctrl/Cmd + B`        | 切换粗体                             | `Ctrl/Cmd + I`        | 切换斜体                             | `Alt + S`（Windows）  | 切换删除线  
| `Ctrl + Shift + ]`    | 提升标题级别                         | `Ctrl + Shift + [`    | 降低标题级别                         | `Ctrl/Cmd + M`        | 切换数学环境                         |
| `Alt + C`             | 切换任务列表项的选中状态             | `Ctrl/Cmd + Shift + V`| 切换预览                             |`Ctrl/Cmd + K V`      | 将预览显示在侧边栏                   |
`Ctrl + V`|自动添加链接（先复制链接，再选中文本，然后按下此快捷键）

![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/5b7d0dfee27d4379af1704f12e2416b7.gif#pic_center> =400x)
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/fd921bd466254096baafd1790bfb0676.gif#pic_center> =400x)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96d3597109af49b6ad2cdf56a697020e.gif#pic_center)

#### 3.1.2 目录生成（及所有Markdown All in One命令）

&#8195;&#8195;Markdown All in One中有以下命令，可对md目录进行操作，可设置对应快捷键。

| 命令名称                                    | 描述                                       |
|---------------------------------------------|------------------------------------------|
| `Create Table of Contents`                    | 创建目录索引 。目录会在文件保存时自动更新。如果需要禁用此默认功能，可以修改 `toc.updateOnSave` 设置。            |
| `Update Table of Contents`                   | 更新目录索引                               |
| `Add/Update Section numbers`                 | 添加/更新章节编号。添加后按CTRL +S ，将编号应用到标题                     |
| `Remove section numbers`                     | 移除章节编号。删除后按CTRL +S ，将更改应用到标题                           |
| `Markdown All in One: Toggle code span`          | 切换代码段                                                   |
| `Markdown All in One: Toggle code block`         | 切换代码块                                                   |
| `Markdown All in One: Print current document to HTML` | 将当前文档打印为 HTML                                        |
| `Markdown All in One: Print documents to HTML`    | 将多个文档打印为 HTML（批量模式）                             |
| `Markdown All in One: Toggle math environment`   | 切换数学环境                                                 |
| `Markdown All in One: Toggle list`               | 切换列表，循环通过不同的列表标记（默认为 `-, *, +, 1.,1` ）  |

![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/ab8905e3496a4fe580ae4564eb20bc40.gif#pic_center> =700x)

2. **目录缩进**：可以根据文件配置目录的缩进类型（使用制表符或空格），在 VSCode 的状态栏右下角即可找到相关设置。注意检查 `list.indentationSize` 设置，以确保缩进大小符合你的需求。
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/0cf53be38f8e42119fc2da42e965bb97.png#pic_center> =800x)
3. **与 GitHub 或 GitLab 兼容**：为了使目录与 GitHub 或 GitLab 的渲染方式兼容，需要设置 `slugifyMode` 选项。

- Slugify 是一个将标题转换为 URL 友好格式的过程，不同的平台（如 GitHub、GitLab）对标题的转换方式可能不同，因此需要通过 `slugifyMode` 来指定兼容的转换方式，否则生成的目录链接可能在 GitHub 或 GitLab 上无法正确跳转到对应的标题。
- 打开 VSCode 的设置（`Ctrl +,` ），搜索`markdown.toc.slugifyMode`，即可设置：
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/7d9854ae3e634ded97a9c9cb93a876f2.png#pic_center> =800x)

4. **自定义标题**：有三种方法可以自定义在目录中显示哪些标题。

- **忽略标题**：在标题后添加 `<!-- omit from toc -->` 注释，可以将该标题从目录中排除。该注释也可以放在标题的上方。
- **设置生成目录的起始位置**：当你的文档包含一些非目录性质的标题，或者你希望从文档的某个特定部分开始生成目录时，在对应位置添加 `<!-- no toc -->` 注释，将不会在该注释以上的部分识别和生成目录。
- **设置 toc.levels**：打开 VSCode 的设置（`Ctrl +,` ），搜索 `markdown.extension.toc.levels`，可通过 toc.levels 设置哪些级别的标题应该出现在目录中（默认为1到6级标题）。
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/6bce46924beb4c1fafe8c4067297635d.png#pic_center> =800x)

- **设置toc.omittedFromToc**：在 settings.json 文件中，可以通过 `markdown.extension.toc.omittedFromToc` 设置来排除某些标题（及其子标题）。示例如下：

 ```json
 "markdown.extension.toc.omittedFromToc": {
     "README.md": [  // 指定 README.md 文件中需要从目录中排除的标题
         "# Introduction",  // 排除一级标题 "Introduction" 及其所有子标题
         "## Also omitted",  // 排除二级标题 "Also omitted" 及其所有子标题
     ],
     "/home/foo/Documents/todo-list.md": [  // 指定位于指定路径下的 todo-list.md 文件中需要排除的标题
         "## Shame list (I'll never do these)",  // 排除二级标题 "Shame list (I'll never do these)" 及其所有子标题
     ]
 }
 ```

- 在排除标题时，确保文档中的标题是唯一的。重复的标题可能导致不可预测的行为。

- 如果你使用的是用 `===` 或 `---` 下划线表示的标题Setext 标题，也可以排除，只需在设置中添加它们的 # 和 ## 版本。例如，你有以下标题：

 ```markup
 Introduction
 =============
 ```

你可以在 toc.omittedFromToc 设置中这样写：

```json
"markdown.extension.toc.omittedFromToc": {
    "README.md": [
        "# Introduction",  // 排除 Setext 标题
    ]
}
```

### 3.1.3 列表编辑

1. **自动继续列表（`Enter` ）**：
当你在列表项中按下 `Enter` 键时，将在新行上放置一个正确的列表标记。如果是有序列表，数字也会相应更新。同时，支持“复制/移动行上/下”的操作。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7d98853b0e4449aca556bf7d729751f2.gif#pic_center)

2. **自适应缩进**：默认情况下，插件会根据 CommonMark 规范自动确定不同列表上下文中的缩进大小。列表项与其父级内容左对齐。通常意味着：

- 无序列表项每级缩进两个空格； 有序列表项缩进的宽度是其父级列表标记的宽度加上后面的空格。
- 如果你更喜欢固定大小，可以通过设置 `list.indentationSize` 来更改。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ebc65d3e1845423ba90df5c40d3537eb.gif#pic_center)

3. **增加或减少列表级别（`Tab,Backspace`）**：
  在列表标记后的第一个空格后按 `Tab` 键可以增加列表级别，按 `Backspace` 键可以减少列表级别。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0db1987804204390aa4c7d755bf1d5a4.gif#pic_center)
4. **任务列表（ `Alt + C` ）**：
  GitHub Flavored Markdown 风格的任务列表是受支持的。你可以使用 `Alt + C` 快捷键（`markdown.extension.checkTaskList` 命令）来勾选或取消勾选任务列表项。例如：

- [ ] 未完成的任务 1
- [x] 已完成的任务 2
- [X] 已完成的任务 3

### 3.1.4 将 Markdown 文档转换为 HTML 格式

| 命令名称                                    | 描述                                       |
|---------------------------------------------|------------------------------------------|
`Markdown: Print current document to HTML`|将当前打开的单个 Markdown 文档转换为 HTML 格式。
`Markdown: Print documents to HTML`|将多个 Markdown 文档批量转换为 HTML 格式。

1. **兼容性**：以上命令兼容其他 Markdown 插件（如 Markdown Footnotes），转换后的 HTML 文件在外观上应该与在 VS Code 中的显示效果相同，除了一些由于 API 限制导致的主题颜色可能会有所不同。

2. **指定标题**：你可以在 Markdown 文档的第一行使用特定的注释 `<!-- title: Your Title -->` 来指定导出的 HTML 文件的标题：
3. **文件链接转换**：文档中指向 .md 文件的普通链接将自动转换为指向 .html 文件的链接。这使得在 HTML 文件中点击链接时，可以直接打开相应的 HTML 页面。

4. 建议使用浏览器（如 Chrome）将导出的 HTML 转换为 PDF 以便共享。

#### 3.1.5 其他功能

- 数学公式：建议使用 Markdown+Math 扩展，并禁用本扩展的 `math.enabled` 选项。
![在这里插入图片描述](<https://i-blog.csdnimg.cn/direct/415c3aa05ef9441da8ea3119ed9be298.png#pic_center> =800x)
- 自动补全：支持文件路径补全、数学公式补全、链接补全
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b559b701894a4d41b12fdda6f2a2b458.png)

### 3.2 支持的设置

|类别| 设置名称                                      | 描述                                                         |
|-----------------------------------------------|--------------------------------------------------------------|-----------------------------------------------
补全设置| `markdown.extension.completion.respectVscodeSearchExclude` | 是否考虑 `search.exclude` 选项。                           |
|| `markdown.extension.completion.root`            | 提供文件路径补全时的根文件夹。                                 |
文本设置| `markdown.extension.italic.indicator`           | 斜体文本的包裹符号（`*` 或 `_`）。                          |
|| `markdown.extension.bold.indicator`             | 粗体文本的包裹符号（`**` 或 `__`）。                      |
KaTeX 设置| `markdown.extension.katex.macros`              | KaTeX 宏定义。                                             |
 列表设置| `markdown.extension.list.indentationSize`        | 有序和无序列表的缩进大小。                                  |
|| `markdown.extension.list.toggle.candidate-markers` | 切换有序列表标记的候选符号。                           |
|| `markdown.extension.orderedList.autoRenumber`   | 自动修复列表标记。                                         |
|| `markdown.extension.orderedList.marker`         | 有序列表的标记类型。                                       |
预览设置| `markdown.extension.preview.autoShowPreviewToSide` | 打开 Markdown 文件时自动显示侧边预览。                 |
|| `markdown.extension.print.absoluteImgPath`      | 将图片路径转换为绝对路径。                                |
|| `markdown.extension.print.imgToBase64`          | 将图片转换为 Base64 编码。                               |
|| `markdown.extension.print.includeVscodeStylesheets` | 是否包含 VS Code 的默认样式。                         |
|| `markdown.extension.print.onFileSave`            | 保存文件时打印到 HTML。                                   |
|| `markdown.extension.print.theme`                | 导出 HTML 的主题。                                       |
|| `markdown.extension.print.validateUrls`          | 启用/禁用 URL 验证。                                     |
格式化设置| `markdown.extension.tableFormatter.enabled`      | 启用 GFM 表格格式化。                                     |
|| `markdown.extension.toc.slugifyMode`            | TOC 链接生成的 slugify 模式（`vscode`、`github`、`gitlab` 或 `gitea`）。 |
目录设置| `markdown.extension.toc.omittedFromToc`        | 按项目文件忽略的标题列表。                               |
|| `markdown.extension.toc.levels`                 | 目录中显示的标题级别。                                    |
|| `markdown.extension.toc.orderedList`            | 目录中使用有序列表。                                     |
|| `markdown.extension.toc.plaintext`              | 纯文本模式。                                               |
|| `markdown.extension.toc.unorderedList.marker`    | 目录中无序列表的标记符号。                                |
|| `markdown.extension.toc.updateOnSave`           | 保存时自动更新目录。                                     |

### 3.3 常见问题解答

1. 命令 "command 'markdown.extension.onXXXKey' not found"
大多数情况下，这是因为 VS Code 在首次打开 Markdown 文件时需要几秒钟加载此扩展。如果长时间后仍出现此错误，请尝试重启 VS Code。如果需要，卸载并重新安装此扩展。

2. 哪些 Markdown 语法受支持？
 对于其他 Markdown 语法，需要从 VS Code 市场安装相应的扩展（如 Mermaid 图表、表情符号、脚注和上标）。安装后，它们将在 VS Code 和导出的 HTML 文件中生效。

3. 性能问题
性能问题可能是由于其他扩展（如某些拼写检查扩展）引起的。可以通过禁用所有其他扩展来验证。要查找根本原因，可以安装开发版本并创建 CPU 分析文件，然后在 GitHub 上提交问题报告。

### 3.4 更新日志

&#8195;&#8195;最新开发版本可从[指定链接](https://github.com/yzhang-gh/vscode-markdown/actions/workflows/main.yml?query=event:push%20is:success)下载。有两个版本：常规构建`markdown-all-in-one-*.vsix`和用于创建详细 CPU 分析文件的调试版本`debug.vsix`。安装时，请在 VS Code 命令面板中执行相应命令。

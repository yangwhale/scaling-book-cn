# 📊 如何转换成 Google Slides

本文档提供三种方法将 Markdown 演讲稿转换成 Google Slides。

---

## 方法一：使用 Marp CLI（推荐）⭐

### 1. 安装 Marp CLI

```bash
# 使用 npm
npm install -g @marp-team/marp-cli

# 或使用 brew (macOS)
brew install marp-cli
```

### 2. 生成 PPTX 文件

```bash
# 在项目根目录执行
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.pptx
```

### 3. 上传到 Google Slides

1. 打开 [Google Drive](https://drive.google.com)
2. 点击 **新建** → **文件上传**
3. 选择生成的 `chapter-01.pptx`
4. 右键点击上传的文件 → **打开方式** → **Google 幻灯片**
5. 会自动转换成 Google Slides 格式！

### 4. 可选：导出其他格式

```bash
# 导出 PDF
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.pdf

# 导出 HTML（可在浏览器中演示）
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.html
```

---

## 方法二：使用 VS Code 插件

### 1. 安装 Marp for VS Code

1. 打开 VS Code
2. 按 `Cmd+Shift+X` 打开扩展商店
3. 搜索 "Marp for VS Code"
4. 点击安装

### 2. 导出 PPTX

1. 打开 `chapter-01.marp.md` 文件
2. 点击右上角的 Marp 图标 
3. 选择 **Export slide deck...**
4. 选择 **PPTX** 格式
5. 保存文件

### 3. 上传到 Google Slides

同方法一的步骤 3。

---

## 方法三：手动复制到 Google Slides

如果你想更精细地控制布局，可以手动操作：

### 1. 创建新的 Google Slides

访问 [slides.new](https://slides.new) 创建新演示文稿

### 2. 参考 slides.md 文件

打开 `slides.md` 文件，按照以下结构复制内容：

- **📊 幻灯片内容** 部分 → 复制到幻灯片
- **🎤 演讲脚本** 部分 → 添加到 **演讲者备注**（点击幻灯片底部的备注区域）

### 3. 推荐的 Google Slides 模板

- 使用 **Simple Light** 主题
- 标题字体：Google Sans 或 Roboto
- 正文字体：Noto Sans SC（支持中文）

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `slides.md` | 完整演讲稿（含脚本），适合备课 |
| `chapter-01.marp.md` | Marp 格式，可直接转换成 PPT |
| `chapter-01.pptx` | 生成的 PowerPoint 文件 |
| `chapter-01.pdf` | 生成的 PDF 文件 |

---

## ⚠️ 注意事项

1. **数学公式**：Google Slides 不原生支持 LaTeX。建议：
   - 安装 [Auto-LaTeX Equations](https://workspace.google.com/marketplace/app/autolatex_equations/850293439076) 插件
   - 或将公式截图后插入

2. **中文字体**：确保选择支持中文的字体（如 Noto Sans SC）

3. **图片**：Marp 中的网络图片需要在导出后检查是否正确显示

---

## 🚀 快速命令

```bash
# 一键生成所有格式
cd /path/to/scaling-book-cn
npm install -g @marp-team/marp-cli  # 首次安装

# 生成 PPTX + PDF + HTML
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.pptx
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.pdf  
marp slides/chapter-01-roofline/chapter-01.marp.md -o slides/chapter-01-roofline/chapter-01.html
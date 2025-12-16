# 如何扩展你的模型

本书旨在揭开在 TPU 上扩展大语言模型（LLM）的神秘面纱。我们尝试解释 TPU 的工作原理、LLM 如何在大规模下实际运行，以及如何在训练和推理过程中选择并行方案以避免通信瓶颈。本书可在 https://jax-ml.github.io/scaling-book 上阅读。

**中文版说明：** 这是 Scaling Book 的中文翻译版本，由社区贡献。如有翻译问题或建议，欢迎提出。

### 致谢

本书由 Google DeepMind 的 Jacob Austin、Sholto Douglas、Roy Frostig、Anselm Levskaya、Charlie Chen、Sharad Vikram、Federico Lebron、Peter Choy、Vinay Ramasesh 和 Albert Webson 撰写。许多核心思想最初由 James Bradbury 和 Reiner Pope 提出。

网站使用了由 https://github.com/alshedivat/al-folio 和 Distill 团队创建的 Distill 风格 Jekyll 主题。特此感谢！

### 本地运行

要在本地构建此仓库，你需要安装 Ruby、ImageMagick 和 Jupyter。在 MacOS 上可以使用 Homebrew 安装：

```
brew install imagemagick ruby
pip install jupyter
```

安装完成后，你需要确保正确版本的 Ruby 在 PATH 中。你应该至少安装 ruby 3.4.5。你可能需要在 .bashrc 中添加以下内容来获取正确版本：

```
if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  export PATH=/opt/homebrew/opt/ruby/bin:$PATH
  export PATH=`gem environment gemdir`/bin:$PATH
fi
```

之后，你应该可以克隆并运行仓库了：

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

成功运行 jekyll serve 后，本书将在 `http://127.0.0.1:4000/scaling-book` 上可用。

要部署到 GitHub Pages 站点（需要仓库写权限），运行 `sh bin/deploy`，大约需要 3 分钟完成。

### 贡献与联系

如果你发现任何问题或有疑问，请在网站本身（由 Giscus 提供支持）或 GitHub 讨论区留言。如果你想贡献，欢迎提交 PR。你也可以发邮件至 jaaustin [at] google [dot] com。

在 GitHub 上贡献需要签署 Google "贡献者许可协议"（CLA）。你可以在这里签署：https://cla.developers.google.com/clas。

### 引用

在学术场景中进行引用时，请按以下格式引用本作品：

```Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.```

BibTeX 引用格式：

```
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

![dragon](assets/img/dragon.png)

*本书最初名为 "How To Scale Your Dragon"（如何训练你的龙），取自梦工厂电影，因此使用了龙的意象。*

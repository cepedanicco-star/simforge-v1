# 用户手动上传与公开

本轮只创建本地干净仓库与tag，不推送、不改远程、不创建GitHub Release。**本地历史干净不代表原私有simforge远程的历史干净。** 旧分支/提交可能含本机路径或实验资料，本轮保留它们私有，不建议直接翻转旧仓可见性。

## 推荐：新的空私有仓

1. 登录GitHub，创建一个新的空私有仓（例如simforge-v1），不要自动初始化README/许可证。
2. 在本地交付的ue-isaac-pipeline仓库目录执行以下命令，把占位符替换为刚创建的仓库地址：

```bash
git remote add origin <YOUR_NEW_EMPTY_PRIVATE_REPOSITORY_URL>
git push -u origin main
git push origin v1.0.0
```

3. 检查网页文件树、README和Releases旁的tag。可下载自动生成的源码ZIP，或把本地打包ZIP附到手动Release。
4. 确认检查对象是这个新仓，再手动打开仓库 **Settings → General → Danger Zone → Change visibility → Public**，按GitHub当时的确认步骤操作。
5. 手动进入 **Releases → Draft a new release**，选择已推送的v1.0.0，标题填写首版名称，正文复制RELEASE_v1.md。需要时附源码ZIP与SHA256SUMS.txt；本版不提供权重，视频用README中的外部链接。
6. 用未登录/隐私浏览窗口验证README、下载和视频链接。公开视频不代表私有源码仓已公开。

## 用Git bundle恢复

若只移动了bundle文件，可在空目录执行`git clone <LOCAL_BUNDLE_FILE> ue-isaac-pipeline`，再`git tag --list`确认v1.0.0。clone后origin可能仍指向本地bundle，上传前用`git remote set-url origin <YOUR_NEW_EMPTY_PRIVATE_REPOSITORY_URL>`替换，然后推main和tag。

源代码ZIP不含.git；若从ZIP开始，需要自行git init/commit并标记v1.0.0。推荐直接使用带干净历史的仓库目录或bundle，保留已核验tag。

## 不要把旧历史合回来

不要将旧实验仓设为新发布仓的upstream后merge全部历史，也不要把旧branches/tags整体push过来。不要把之前的bonus-report JSON、auto-run文档、原始数据、checkpoint、USD、个人路径或密钥拖进网页上传。对原仓执行force-push清洗不是本轮操作；需要原仓同名公开时，应另做完整迁移和远程历史清理审查，不能只替换main当前文件。

# 发布到 npm（@jxhs/vla-mini）

## 403：需要 2FA 或 Granular Token

npm 官方要求 **发布包** 必须满足其一：

1. 账号开启 **双因素认证（2FA）**  
   - 打开 https://www.npmjs.com/settings/~/account  
   - 启用 2FA（推荐 Authenticator App）  
   - 再执行 `npm publish --access public`

2. 或使用 **Granular Access Token**（带 Publish 权限）  
   - https://www.npmjs.com/settings/~/tokens → Generate New Token → Granular  
   - Packages: Read and write  
   - 勾选 bypass 2FA（若页面有该选项）  
   - 本机登录：
     ```bash
     npm config set registry https://registry.npmjs.org/
     npm login
     # 或: npm config set //registry.npmjs.org/:_authToken=你的token
     ```

## 发布步骤

```bash
npm config set registry https://registry.npmjs.org/
cd d:\vla\npm\vla-mini
# 确认 package.json 里是 "name": "@jxhs/vla-mini"
npm run bundle
npm publish --access public
```

## 别人如何安装

```bash
npm install -g @jxhs/vla-mini
vla-mini install
vla-mini demo --dry-run
```

或一次性：`npx @jxhs/vla-mini install`

## 注意

- `"name": "@jxhs/vla-mini"` 必须写在 **package.json** 里，不能在 cmd 里直接输入。
- 下载依赖仍可用国内镜像；**只有 publish 要用** `registry.npmjs.org`。
- 已登录 npmmirror 与登录 npmjs 是两套账号，互不相通。

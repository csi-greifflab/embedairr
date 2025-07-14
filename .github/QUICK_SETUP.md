# Quick Setup Guide: GitHub Environment + Trusted Publishing

## 🎯 Quick Steps

### 1. Create GitHub Environment (2 minutes)
```
GitHub Repository → Settings → Environments → New environment
Name: testpypi
Protection rules: 
  - Deployment branches: test
  - Save protection rules
```

### 2. Configure TestPyPI Trusted Publishing (3 minutes)
```
TestPyPI → Account Settings → Publishing → Add trusted publisher
PyPI Project Name: pypepe
Owner: your-github-username
Repository: pypepe
Workflow: publish-test-branch-trusted.yml
Environment: testpypi
```

### 3. Test It! (30 seconds)
```bash
git checkout test
git commit --allow-empty -m "Test publishing"
git push origin test
```

## 🔧 Exact Configuration Values

**GitHub Environment:**
- Name: `testpypi`
- Allowed branches: `test`

**TestPyPI Trusted Publisher:**
- Project Name: `pypepe`
- Owner: `[YOUR_GITHUB_USERNAME]`
- Repository: `pypepe`
- Workflow: `publish-test-branch-trusted.yml`
- Environment: `testpypi`

## 📁 Workflow File

Only one workflow file is needed:
- `publish-test-branch-trusted.yml` - Trusted publishing with verification

## 🚀 What Happens Next

1. Push to `test` branch → Workflow triggers
2. GitHub requests deployment approval (if configured)
3. Workflow builds package with unique version
4. Publishes to TestPyPI via trusted publishing
5. Package available at: https://test.pypi.org/project/pypepe/

## 🛠️ Install Test Package
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pypepe
```

## ❓ Need Help?
- Check workflow logs in Actions tab
- Verify environment in Settings → Environments
- See full guide in `GITHUB_ENVIRONMENT_SETUP.md`

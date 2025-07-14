# Setting Up GitHub Environment for Trusted Publishing

This guide will walk you through setting up a GitHub environment for secure publishing to TestPyPI using trusted publishing.

## Step 1: Create GitHub Environment

1. **Go to your GitHub repository** (https://github.com/your-username/pypepe)

2. **Navigate to Settings**:
   - Click on the "Settings" tab in your repository
   - Scroll down to the "Environments" section in the left sidebar
   - Click on "Environments"

3. **Create new environment**:
   - Click "New environment"
   - Name: `testpypi`
   - Click "Configure environment"

4. **Configure environment protection rules** (optional but recommended):
   - **Deployment branches**: Select "Selected branches" and add `test` branch
   - **Environment secrets**: We won't need any for trusted publishing
   - **Reviewers**: You can add yourself or team members who should approve deployments
   - Click "Save protection rules"

## Step 2: Configure Trusted Publishing on TestPyPI

1. **Go to TestPyPI** (https://test.pypi.org)

2. **Log in or create an account**

3. **Navigate to Account Settings**:
   - Click on your username in the top right
   - Select "Account settings" from the dropdown

4. **Set up Trusted Publishing**:
   - Scroll down to the "API tokens" section
   - Click on "Add API token" or look for "Trusted publishing"
   - If you see "Trusted publishing", click "Add a new pending publisher"
   - If not, look for "Publishing" in the left sidebar

5. **Configure the trusted publisher**:
   - **PyPI Project Name**: `pypepe` (or your desired package name)
   - **Owner**: Your GitHub username (e.g., `your-username`)
   - **Repository name**: `pypepe`
   - **Workflow filename**: `publish-test-branch-trusted.yml`
   - **Environment name**: `testpypi`
   - Click "Add"

## Step 3: Test the Setup

1. **Push to test branch**:
   ```bash
   git checkout test
   # Make some changes or just trigger the workflow
   git commit --allow-empty -m "Test trusted publishing setup"
   git push origin test
   ```

2. **Monitor the workflow**:
   - Go to the "Actions" tab in your GitHub repository
   - You should see the workflow running
   - If configured with reviewers, you'll need to approve the deployment

3. **Check the deployment**:
   - Go to the "Environments" section in your repository settings
   - You should see deployment history under the `testpypi` environment

## Step 4: Verify Publication

After successful workflow completion:

1. **Check TestPyPI**:
   - Go to https://test.pypi.org/project/pypepe/
   - You should see your package listed

2. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pypepe
   ```

## Troubleshooting

### Common Issues

1. **"No valid trusted publisher found"**:
   - Double-check the configuration on TestPyPI
   - Ensure the repository name, workflow filename, and environment name match exactly

2. **"Environment protection rules prevent deployment"**:
   - Check your environment settings
   - Make sure the `test` branch is allowed for deployment
   - Approve the deployment if reviewers are required

3. **"Package does not exist"**:
   - For the first publication, you might need to create the package on TestPyPI first
   - Or remove the environment restriction for the first run

4. **"Workflow doesn't trigger"**:
   - Ensure you're pushing to the `test` branch
   - Check that the workflow file is in the correct location

### Debugging Steps

1. **Check workflow logs**:
   - Go to Actions tab → Select the failed workflow run
   - Click on the job to see detailed logs

2. **Verify environment configuration**:
   - Settings → Environments → testpypi
   - Check deployment history and logs

3. **Test without environment first**:
   - Temporarily remove the `environment: testpypi` line from the workflow
   - Push to test branch to see if trusted publishing works
   - Add the environment back once confirmed working

## Common TestPyPI Version Issues

### "Local versions not allowed" Error

**Problem**: Error message like `The use of local versions in <Version('1.0.0.dev20250714131355+9372b3f')> is not allowed`

**Solution**: This happens when the version contains a local identifier (the `+commit_hash` part). The workflows have been updated to use only development versions without local identifiers.

**Fixed version format**: `1.0.0.dev20250714131355` (no `+commit_hash`)

If you see this error, ensure your workflow uses the corrected versioning scheme without the `+${COMMIT_SHA}` part.

## Security Benefits

- ✅ No API tokens to manage or rotate
- ✅ Automatic authentication via OpenID Connect
- ✅ Scoped permissions (only this repo can publish)
- ✅ Audit trail of all publications
- ✅ Environment protection rules for additional security

## Next Steps

Once TestPyPI is working, you can set up production PyPI publishing:

1. Create a `pypi` environment in GitHub
2. Configure trusted publishing on PyPI (not TestPyPI)
3. Create a production workflow for releases

This setup provides a secure, maintainable way to publish your Python packages!

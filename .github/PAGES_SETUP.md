# GitHub Pages Setup for Test Results

This repository is configured to automatically deploy test visual outputs to GitHub Pages after each test run.

## One-Time Setup Required

To enable GitHub Pages, you need to configure it once in your repository settings:

1. **Go to your repository on GitHub**
2. **Click Settings** (top menu)
3. **Click Pages** (left sidebar under "Code and automation")
4. **Under "Source"**:
   - Select: **GitHub Actions**

That's it! No need to select a branch or folder.

## How It Works

After setup:

1. Every time tests run in GitHub Actions, the workflow:
   - Generates test output images (PNG files)
   - Creates an HTML gallery with thumbnails
   - Deploys to GitHub Pages

2. **View test results** at:
   ```
   https://<your-username>.github.io/ComfyUI-Grounding/<run-number>/
   ```

   Replace:
   - `<your-username>` with your GitHub username
   - `<run-number>` with the workflow run number (shown in Actions tab)

3. **Direct link** is also shown in the GitHub Actions summary after tests complete

## Features

- **Visual Gallery**: Browse all test output images with thumbnails
- **Lightbox View**: Click any image to view full-size
- **Multiple Runs**: Each test run gets its own folder (by run number)
- **History**: Old test results are kept (controlled by `keep_files: true`)
- **Dark Theme**: GitHub-style dark theme matching the Actions UI

## Example URL Structure

```
https://yourname.github.io/ComfyUI-Grounding/
├── 123/                          # Run #123
│   ├── index.html               # Gallery page
│   ├── sa2va_segmentation.png
│   ├── grounding_dino_detection.png
│   └── ...
├── 124/                          # Run #124
│   ├── index.html
│   └── ...
```

## Troubleshooting

### "404 Page Not Found"
- Make sure you've enabled Pages in Settings → Pages → Source: GitHub Actions
- Wait a few minutes after the first deployment
- Check the Actions tab for any deployment errors

### "No images showing"
- Verify test outputs were created in `tests/test_outputs/`
- Check the workflow logs for the "Deploy to GitHub Pages" step
- Make sure tests actually ran and generated images

### "Old runs not showing"
- Pages only deploys when tests run
- Manual test runs or scheduled runs will create new folders
- Check https://yourname.github.io/ComfyUI-Grounding/ for the list

## Disabling

To disable GitHub Pages deployment:

1. Go to Settings → Pages
2. Select "None" under Source

Or remove the "Deploy to GitHub Pages" step from `.github/workflows/test-real-models.yml`

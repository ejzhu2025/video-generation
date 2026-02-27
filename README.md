---
title: Video Generation
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Video Generation

AI-powered ad video generator with two modes:

- **Text → Video**: Paste a brand brief, Claude generates a storyboard, fal.ai renders the video
- **Image → Video**: Upload a product photo, Claude reads brand tonality and writes a motion prompt, fal.ai animates it

## Models supported
- Wan 2.2 14B (I2V) · Wan 2.1 T2V 14B
- Kling 1.6 Pro (T2V + I2V)
- MiniMax Video-01

## Setup (HuggingFace Spaces Secrets)

Set the following in your Space → Settings → Variables and secrets:

| Secret | Value |
|--------|-------|
| `ANTHROPIC_API_KEY` | from console.anthropic.com |
| `FAL_API_KEY` | from fal.ai |
| `VIDEO_BACKEND` | `fal` |
| `FAL_T2V_MODEL` | `fal-ai/wan/v2.1/t2v-14b` |
| `FAL_I2V_MODEL` | `fal-ai/wan/v2.2-a14b/image-to-video` |
